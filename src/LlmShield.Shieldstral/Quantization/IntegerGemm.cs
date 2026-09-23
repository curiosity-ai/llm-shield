// Register-blocked integer GEMM between block-quantized weights and Q8
// activations.
//
// Derived in approach from TensorSharp's ManagedQuantizedOps and
// TensorComputePrimitives.Dot4 (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause — register blocking so one load feeds several dot products, and
// the vectorised nibble unpack. See third-party/TensorSharp-LICENSE.
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Quantization;

/// <summary>
/// The integer matmul as a GEMM rather than as a pile of dot products.
///
/// <see cref="IntegerDot"/> computes one (row, token) dot at a time, and every one
/// of them ends in the same overhead: a float conversion and scale per 32-weight
/// block, and a horizontal reduction per row. Those are per-<em>output</em> costs,
/// and in a prefill there are hundreds of thousands of outputs per matmul.
///
/// Here eight weight rows are unpacked together and interleaved, four bytes at a
/// time, so that lane <c>j</c> of a 256-bit vector holds four consecutive weights
/// of row <c>j</c>. Broadcasting four activation bytes against that vector makes
/// one <c>vpdpbusd</c> (or <c>vpmaddubsw</c>+<c>vpmaddwd</c>) produce partial sums
/// for eight rows at once, already in the lanes they belong to. The block's scale
/// is then one vector multiply for eight rows, there is no horizontal reduction
/// at all — the accumulator <em>is</em> the eight outputs — and each packed weight
/// vector is reused across a tile of four tokens.
///
/// The weights are unpacked to <em>unsigned</em> bytes, with the type's zero point
/// (8 for Q4_0, 16 for Q5_0, 128 for Q8_0) subtracted afterwards in the integer
/// domain as <c>z·Σa</c>; the unsigned operand is what <c>vpmaddubsw</c> and
/// <c>vpdpbusd</c> want. Q4_1/Q5_1 carry a float offset instead, which contributes
/// <c>m·d_a·Σa</c> per block. Without VNNI the pairwise sums pass through int16:
/// 31·127·2 fits, 255·127·2 does not, so Q8_0 needs <c>vpdpbusd</c> here and falls
/// back to <see cref="IntegerDot"/> on hardware without it.
/// </summary>
public static unsafe class IntegerGemm
{
    /// <summary>Rows unpacked and interleaved together — one per 32-bit lane of a 256-bit vector.</summary>
    public const int GroupRows = 8;

    /// <summary>Tokens each packed weight vector is reused across.</summary>
    public const int TokenTile = 4;

    /// <summary>Weights per quantization block, for every type this handles.</summary>
    public const int BlockSize = 32;

    /// <summary>Bytes one block of a packed group occupies: eight rows of 32 unsigned weights.</summary>
    public const int PackedBlockBytes = GroupRows * BlockSize;

    /// <summary>Whether this machine has the instructions the kernel is written in.</summary>
    public static bool IsHardwareAccelerated => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>Weight types this kernel can multiply on the current machine.</summary>
    public static bool Supports(GgmlType type) => IsHardwareAccelerated && type switch
    {
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 => true,
        // 255·127·2 overflows the int16 pair sums of vpmaddubsw; only vpdpbusd,
        // which accumulates straight into int32, can take Q8_0's full byte range.
        GgmlType.Q8_0 => AvxVnni.IsSupported,
        _ => false,
    };

    /// <summary>True when the type carries a per-block float offset rather than an integer zero point.</summary>
    public static bool HasOffset(GgmlType type) => type is GgmlType.Q4_1 or GgmlType.Q5_1;

    /// <summary>
    /// What the unsigned bytes <see cref="PackGroup"/> writes sit above the signed
    /// weight: <c>w = u − z</c>. Zero for the offset types, whose payload is
    /// unsigned to begin with.
    /// </summary>
    public static int ZeroPoint(GgmlType type) => type switch
    {
        GgmlType.Q4_0 => 8,
        GgmlType.Q5_0 => 16,
        GgmlType.Q8_0 => 128,
        _ => 0,
    };

    // ------------------------------------------------------------ activations

    /// <summary>
    /// Quantizes one row of activations to int8 with one float scale per 32
    /// values, and records each block's integer sum.
    /// </summary>
    /// <remarks>
    /// The scale stays float32 rather than being rounded to f16 as a Q8_0 block
    /// would store it — nothing here is serialised, and the rounding would only
    /// add error. Values round to nearest-even (<c>vcvtps2dq</c>), as ggml's own
    /// SIMD quantizer does; the scalar reference in <see cref="IntegerDot"/> rounds
    /// half away from zero, and the two differ only on exact ties.
    /// </remarks>
    /// <param name="sums">Σq per block: what the zero point and the offset are multiplied by.</param>
    public static void QuantizeActivations(float* x, int count, sbyte* values, float* scales, int* sums)
    {
        Vector256<float> absMask = Vector256.Create(0x7FFFFFFF).AsSingle();
        Vector256<int> order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        int blocks = count / BlockSize;

        for (int b = 0; b < blocks; b++, x += BlockSize, values += BlockSize)
        {
            Vector256<float> v0 = Avx.LoadVector256(x);
            Vector256<float> v1 = Avx.LoadVector256(x + 8);
            Vector256<float> v2 = Avx.LoadVector256(x + 16);
            Vector256<float> v3 = Avx.LoadVector256(x + 24);

            Vector256<float> m = Avx.Max(
                Avx.Max(Avx.And(v0, absMask), Avx.And(v1, absMask)),
                Avx.Max(Avx.And(v2, absMask), Avx.And(v3, absMask)));
            Vector128<float> m4 = Sse.Max(m.GetLower(), m.GetUpper());
            m4 = Sse.Max(m4, Sse.Shuffle(m4, m4, 0b_01_00_11_10));
            m4 = Sse.Max(m4, Sse.Shuffle(m4, m4, 0b_10_11_00_01));
            float max = m4.ToScalar();

            float d = max / 127f;
            Vector256<float> inverse = Vector256.Create(d != 0f ? 1f / d : 0f);

            Vector256<int> i0 = Avx.ConvertToVector256Int32(Avx.Multiply(v0, inverse));
            Vector256<int> i1 = Avx.ConvertToVector256Int32(Avx.Multiply(v1, inverse));
            Vector256<int> i2 = Avx.ConvertToVector256Int32(Avx.Multiply(v2, inverse));
            Vector256<int> i3 = Avx.ConvertToVector256Int32(Avx.Multiply(v3, inverse));

            // |x·inverse| <= 127 by construction, so the saturating packs never
            // saturate; they only narrow. The packs interleave 128-bit lanes, which
            // the final dword permutation puts back in element order.
            Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
            Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);
            Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);
            packed = Avx2.PermuteVar8x32(packed.AsInt32(), order).AsSByte();
            Vector256.Store(packed, values);

            scales[b] = d;
            sums[b] = Vector256.Sum(Avx2.Add(Avx2.Add(i0, i1), Avx2.Add(i2, i3)));
        }
    }

    // ---------------------------------------------------------------- packing

    /// <summary>
    /// Unpacks up to eight weight rows into the interleaved layout the kernel reads:
    /// for block <c>b</c>, eight 32-byte vectors, vector <c>k</c> holding weights
    /// <c>4k..4k+3</c> of each row in turn. Rows beyond <paramref name="rowCount"/>
    /// read as an all-zero block, so a short final group computes (and the caller
    /// discards) zeros.
    /// </summary>
    /// <param name="rows">Pointers to each row's packed GGUF blocks.</param>
    /// <param name="packed"><c>blocks × <see cref="PackedBlockBytes"/></c> bytes.</param>
    /// <param name="scales"><c>blocks × 8</c> floats: the rows' block scales, lane-ordered.</param>
    /// <param name="offsets"><c>blocks × 8</c> floats for the offset types; may be null otherwise.</param>
    public static void PackGroup(GgmlType type, byte** rows, int rowCount, int blocks,
        byte* packed, float* scales, float* offsets)
    {
        switch (type)
        {
            case GgmlType.Q4_0: PackGroup<Q4_0Block>(rows, rowCount, blocks, packed, scales, offsets); return;
            case GgmlType.Q4_1: PackGroup<Q4_1Block>(rows, rowCount, blocks, packed, scales, offsets); return;
            case GgmlType.Q5_0: PackGroup<Q5_0Block>(rows, rowCount, blocks, packed, scales, offsets); return;
            case GgmlType.Q5_1: PackGroup<Q5_1Block>(rows, rowCount, blocks, packed, scales, offsets); return;
            case GgmlType.Q8_0: PackGroup<Q8_0Block>(rows, rowCount, blocks, packed, scales, offsets); return;
            default: throw new NotSupportedException($"{type} has no integer GEMM kernel.");
        }
    }

    /// <remarks>
    /// Specialised per block format, so the type dispatch happens once per group
    /// rather than once per (row, block), and the eight unpacked rows stay in
    /// registers through the transpose. Packing is per-weight work, and at a few
    /// tokens it is most of the matmul.
    /// </remarks>
    private static void PackGroup<TBlock>(byte** rows, int rowCount, int blocks,
        byte* packed, float* scales, float* offsets) where TBlock : struct, IBlockFormat
    {
        byte* zero = stackalloc byte[TBlock.Size];
        new Span<byte>(zero, TBlock.Size).Clear();

        // Missing rows keep re-reading the one zero block.
        byte** p = stackalloc byte*[GroupRows];
        nint* step = stackalloc nint[GroupRows];
        for (int r = 0; r < GroupRows; r++)
        {
            bool real = r < rowCount;
            p[r] = real ? rows[r] : zero;
            step[r] = real ? TBlock.Size : 0;
        }

        for (int b = 0; b < blocks; b++, packed += PackedBlockBytes)
        {
            Vector256<int> v0 = TBlock.Unpack(p[0]).AsInt32(), v1 = TBlock.Unpack(p[1]).AsInt32();
            Vector256<int> v2 = TBlock.Unpack(p[2]).AsInt32(), v3 = TBlock.Unpack(p[3]).AsInt32();
            Vector256<int> v4 = TBlock.Unpack(p[4]).AsInt32(), v5 = TBlock.Unpack(p[5]).AsInt32();
            Vector256<int> v6 = TBlock.Unpack(p[6]).AsInt32(), v7 = TBlock.Unpack(p[7]).AsInt32();

            Transpose8x8(ref v0, ref v1, ref v2, ref v3, ref v4, ref v5, ref v6, ref v7);
            Vector256.Store(v0.AsByte(), packed);
            Vector256.Store(v1.AsByte(), packed + 32);
            Vector256.Store(v2.AsByte(), packed + 64);
            Vector256.Store(v3.AsByte(), packed + 96);
            Vector256.Store(v4.AsByte(), packed + 128);
            Vector256.Store(v5.AsByte(), packed + 160);
            Vector256.Store(v6.AsByte(), packed + 192);
            Vector256.Store(v7.AsByte(), packed + 224);

            Vector256.Store(GatherHalves(p, 0), scales + b * GroupRows);
            if (offsets is not null)
                Vector256.Store(TBlock.HasOffset ? GatherHalves(p, 2) : Vector256<float>.Zero, offsets + b * GroupRows);

            for (int r = 0; r < GroupRows; r++) p[r] += step[r];
        }
    }

    /// <summary>The f16 at byte <paramref name="offset"/> of each of the eight current blocks, as float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> GatherHalves(byte** p, int offset) => HalfToSingle(Vector256.Create(
        Unsafe.ReadUnaligned<ushort>(p[0] + offset), Unsafe.ReadUnaligned<ushort>(p[1] + offset),
        Unsafe.ReadUnaligned<ushort>(p[2] + offset), Unsafe.ReadUnaligned<ushort>(p[3] + offset),
        Unsafe.ReadUnaligned<ushort>(p[4] + offset), Unsafe.ReadUnaligned<ushort>(p[5] + offset),
        Unsafe.ReadUnaligned<ushort>(p[6] + offset), Unsafe.ReadUnaligned<ushort>(p[7] + offset)));

    /// <summary>
    /// Eight f16 bit patterns (zero-extended) to float32, exactly. .NET exposes no
    /// <c>vcvtph2ps</c>, and eight scalar <see cref="Half"/> conversions per block
    /// were a measurable part of packing.
    /// </summary>
    /// <remarks>
    /// Shifting the exponent and mantissa into float position and multiplying by
    /// 2^112 re-biases the exponent from 15 to 127; the same multiply scales a
    /// subnormal half, which lands as a float subnormal, to its exact value.
    /// Infinities and NaNs get their exponent forced to all-ones afterwards, and
    /// NaNs are quieted, as both <see cref="Half"/> and <c>vcvtph2ps</c> do.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> HalfToSingle(Vector256<int> bits)
    {
        Vector256<int> magnitude = Avx2.And(bits, Vector256.Create(0x7FFF));
        Vector256<int> sign = Avx2.ShiftLeftLogical(Avx2.And(bits, Vector256.Create(0x8000)), 16);
        Vector256<float> value = Avx.Multiply(
            Avx2.ShiftLeftLogical(magnitude, 13).AsSingle(), Vector256.Create(0x77800000).AsSingle());
        Vector256<int> special = Avx2.Or(
            Avx2.And(Avx2.CompareGreaterThan(magnitude, Vector256.Create(0x7BFF)), Vector256.Create(0x7F800000)),
            Avx2.And(Avx2.CompareGreaterThan(magnitude, Vector256.Create(0x7C00)), Vector256.Create(0x00400000)));
        return Avx2.Or(Avx2.Or(value.AsInt32(), sign), special).AsSingle();
    }

    /// <summary>Where a block format keeps its fields, and how its payload becomes 32 unsigned bytes.</summary>
    private interface IBlockFormat
    {
        /// <summary>Bytes per 32-weight block.</summary>
        static abstract int Size { get; }
        /// <summary>Whether an f16 offset follows the f16 scale.</summary>
        static abstract bool HasOffset { get; }
        /// <summary>The block's 32 weights as unsigned bytes, in activation order.</summary>
        static abstract Vector256<byte> Unpack(byte* block);
    }

    private struct Q4_0Block : IBlockFormat
    {
        public static int Size => 18;
        public static bool HasOffset => false;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Unpack(byte* block) => Nibbles(block + 2);
    }

    private struct Q4_1Block : IBlockFormat
    {
        public static int Size => 20;
        public static bool HasOffset => true;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Unpack(byte* block) => Nibbles(block + 4);
    }

    private struct Q5_0Block : IBlockFormat
    {
        public static int Size => 22;
        public static bool HasOffset => false;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Unpack(byte* block)
            => Avx2.Or(Nibbles(block + 6), FifthBits(Unsafe.ReadUnaligned<uint>(block + 2)));
    }

    private struct Q5_1Block : IBlockFormat
    {
        public static int Size => 24;
        public static bool HasOffset => true;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Unpack(byte* block)
            => Avx2.Or(Nibbles(block + 8), FifthBits(Unsafe.ReadUnaligned<uint>(block + 4)));
    }

    private struct Q8_0Block : IBlockFormat
    {
        public static int Size => 34;
        public static bool HasOffset => false;
        // w + 128 as an unsigned byte is w with its sign bit flipped.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<byte> Unpack(byte* block)
            => Avx2.Xor(Vector256.Load(block + 2), Vector256.Create((byte)0x80));
    }

    /// <summary>16 packed bytes → 32 nibbles in ggml's order: every low nibble, then every high one.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Nibbles(byte* qs)
    {
        Vector128<byte> packed = Vector128.Load(qs);
        Vector128<byte> mask = Vector128.Create((byte)0x0F);
        Vector128<byte> low = Sse2.And(packed, mask);
        Vector128<byte> high = Sse2.And(Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), mask);
        return Vector256.Create(low, high);
    }

    /// <summary>
    /// Q5's 32 high bits → <c>0x10</c> in byte <c>j</c> where bit <c>j</c> is set.
    /// Each byte picks up the qh byte holding its bit, is OR-ed with a mask that
    /// has every bit set but that one, and so compares equal to 0xFF exactly when
    /// its bit was set — the expansion ggml's <c>bytes_from_bits_32</c> uses.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> FifthBits(uint qh)
    {
        Vector256<byte> select = Vector256.Create(
            (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        Vector256<byte> bytes = Avx2.Shuffle(Vector256.Create(qh).AsByte(), select);
        bytes = Avx2.Or(bytes, Vector256.Create(0x7FBFDFEFF7FBFDFEUL).AsByte());
        Vector256<byte> set = Avx2.CompareEqual(bytes, Vector256<byte>.AllBitsSet);
        return Avx2.And(set, Vector256.Create((byte)0x10));
    }

    /// <summary>In-place transpose of an 8×8 matrix of 32-bit elements, one row per vector.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8x8(
        ref Vector256<int> m0, ref Vector256<int> m1, ref Vector256<int> m2, ref Vector256<int> m3,
        ref Vector256<int> m4, ref Vector256<int> m5, ref Vector256<int> m6, ref Vector256<int> m7)
    {
        Vector256<int> t0 = Avx2.UnpackLow(m0, m1), t1 = Avx2.UnpackHigh(m0, m1);
        Vector256<int> t2 = Avx2.UnpackLow(m2, m3), t3 = Avx2.UnpackHigh(m2, m3);
        Vector256<int> t4 = Avx2.UnpackLow(m4, m5), t5 = Avx2.UnpackHigh(m4, m5);
        Vector256<int> t6 = Avx2.UnpackLow(m6, m7), t7 = Avx2.UnpackHigh(m6, m7);

        Vector256<long> u0 = Avx2.UnpackLow(t0.AsInt64(), t2.AsInt64()), u1 = Avx2.UnpackHigh(t0.AsInt64(), t2.AsInt64());
        Vector256<long> u2 = Avx2.UnpackLow(t1.AsInt64(), t3.AsInt64()), u3 = Avx2.UnpackHigh(t1.AsInt64(), t3.AsInt64());
        Vector256<long> u4 = Avx2.UnpackLow(t4.AsInt64(), t6.AsInt64()), u5 = Avx2.UnpackHigh(t4.AsInt64(), t6.AsInt64());
        Vector256<long> u6 = Avx2.UnpackLow(t5.AsInt64(), t7.AsInt64()), u7 = Avx2.UnpackHigh(t5.AsInt64(), t7.AsInt64());

        m0 = Avx2.Permute2x128(u0, u4, 0x20).AsInt32();
        m1 = Avx2.Permute2x128(u1, u5, 0x20).AsInt32();
        m2 = Avx2.Permute2x128(u2, u6, 0x20).AsInt32();
        m3 = Avx2.Permute2x128(u3, u7, 0x20).AsInt32();
        m4 = Avx2.Permute2x128(u0, u4, 0x31).AsInt32();
        m5 = Avx2.Permute2x128(u1, u5, 0x31).AsInt32();
        m6 = Avx2.Permute2x128(u2, u6, 0x31).AsInt32();
        m7 = Avx2.Permute2x128(u3, u7, 0x31).AsInt32();
    }

    // ----------------------------------------------------------------- kernel

    /// <summary>
    /// Four tokens against one packed group: writes eight outputs per token,
    /// <c>y[t·yStride + 0..7]</c>.
    /// </summary>
    /// <remarks>
    /// Every output lane goes through exactly the same sequence of operations as
    /// in <see cref="Multiply1"/>, so a token's result does not depend on which
    /// tile it landed in — the prefix cache relies on that to be bit-exact.
    /// </remarks>
    /// <param name="values">Token 0's int8 activations; the others follow at a stride of <paramref name="cols"/>.</param>
    /// <param name="activationScales">Token 0's block scales; the others follow at a stride of <paramref name="blocks"/>.</param>
    /// <param name="activationSums">Token 0's Σq per block, same stride.</param>
    public static void Multiply4(
        byte* packed, float* scales, float* offsets, int blocks, int zeroPoint,
        sbyte* values, float* activationScales, int* activationSums, int cols,
        float* y, int yStride)
    {
        Vector256<float> f0 = Vector256<float>.Zero, f1 = f0, f2 = f0, f3 = f0;
        Vector256<short> ones = Vector256.Create((short)1);
        sbyte* a0 = values, a1 = a0 + cols, a2 = a1 + cols, a3 = a2 + cols;
        float* d0 = activationScales, d1 = d0 + blocks, d2 = d1 + blocks, d3 = d2 + blocks;
        int* s0 = activationSums, s1 = s0 + blocks, s2 = s1 + blocks, s3 = s2 + blocks;

        for (int b = 0; b < blocks; b++, packed += PackedBlockBytes, a0 += 32, a1 += 32, a2 += 32, a3 += 32)
        {
            // Starting from −z·Σa folds the zero point in for free; integer
            // addition is exact, so where it happens does not change the result.
            Vector256<int> i0 = Vector256.Create(-zeroPoint * s0[b]), i1 = Vector256.Create(-zeroPoint * s1[b]);
            Vector256<int> i2 = Vector256.Create(-zeroPoint * s2[b]), i3 = Vector256.Create(-zeroPoint * s3[b]);
            // Unrolled by hand: the JIT keeps a constant-trip loop as a loop, and
            // its counter and branch are a fifth of the instructions otherwise.
            Step(packed, 0, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 4, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 8, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 12, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 16, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 20, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 24, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);
            Step(packed, 28, a0, a1, a2, a3, ones, ref i0, ref i1, ref i2, ref i3);

            Vector256<float> scale = Vector256.Load(scales + b * GroupRows);
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Avx.Multiply(scale, Vector256.Create(d0[b])), f0);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Avx.Multiply(scale, Vector256.Create(d1[b])), f1);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), Avx.Multiply(scale, Vector256.Create(d2[b])), f2);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), Avx.Multiply(scale, Vector256.Create(d3[b])), f3);

            if (offsets is not null)
            {
                Vector256<float> offset = Vector256.Load(offsets + b * GroupRows);
                f0 = Fma.MultiplyAdd(offset, Vector256.Create(d0[b] * s0[b]), f0);
                f1 = Fma.MultiplyAdd(offset, Vector256.Create(d1[b] * s1[b]), f1);
                f2 = Fma.MultiplyAdd(offset, Vector256.Create(d2[b] * s2[b]), f2);
                f3 = Fma.MultiplyAdd(offset, Vector256.Create(d3[b] * s3[b]), f3);
            }
        }

        Vector256.Store(f0, y);
        Vector256.Store(f1, y + yStride);
        Vector256.Store(f2, y + 2 * yStride);
        Vector256.Store(f3, y + 3 * yStride);
    }

    /// <summary>Weights <c>k..k+3</c> of the block for all eight rows, against four tokens.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Step(byte* packed, int k, sbyte* a0, sbyte* a1, sbyte* a2, sbyte* a3, Vector256<short> ones,
        ref Vector256<int> i0, ref Vector256<int> i1, ref Vector256<int> i2, ref Vector256<int> i3)
    {
        Vector256<byte> w = Vector256.Load(packed + k * 8);
        i0 = MultiplyAdd(i0, w, Broadcast(a0 + k), ones);
        i1 = MultiplyAdd(i1, w, Broadcast(a1 + k), ones);
        i2 = MultiplyAdd(i2, w, Broadcast(a2 + k), ones);
        i3 = MultiplyAdd(i3, w, Broadcast(a3 + k), ones);
    }

    /// <summary>One token against one packed group — the tail of a tile, and the whole of a decode step.</summary>
    public static void Multiply1(
        byte* packed, float* scales, float* offsets, int blocks, int zeroPoint,
        sbyte* values, float* activationScales, int* activationSums, float* y)
    {
        Vector256<float> f0 = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        sbyte* a0 = values;

        for (int b = 0; b < blocks; b++, packed += PackedBlockBytes, a0 += 32)
        {
            Vector256<int> i0 = Vector256.Create(-zeroPoint * activationSums[b]);
            for (int k = 0; k < 32; k += 4)
                i0 = MultiplyAdd(i0, Vector256.Load(packed + k * 8), Broadcast(a0 + k), ones);

            Vector256<float> scale = Vector256.Load(scales + b * GroupRows);
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Avx.Multiply(scale, Vector256.Create(activationScales[b])), f0);

            if (offsets is not null)
                f0 = Fma.MultiplyAdd(Vector256.Load(offsets + b * GroupRows),
                    Vector256.Create(activationScales[b] * activationSums[b]), f0);
        }

        Vector256.Store(f0, y);
    }

    /// <summary>Four activation bytes, repeated into every 32-bit lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<sbyte> Broadcast(sbyte* p) => Vector256.Create(Unsafe.ReadUnaligned<int>(p)).AsSByte();

    /// <summary>
    /// <c>acc[j] += Σ_{i<4} w[4j+i]·a[4j+i]</c>, unsigned by signed. One
    /// <c>vpdpbusd</c> with VNNI; otherwise the two-step <c>vpmaddubsw</c> →
    /// <c>vpmaddwd</c>, whose int16 intermediate is why Q8_0 needs VNNI here.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> MultiplyAdd(Vector256<int> acc, Vector256<byte> w, Vector256<sbyte> a, Vector256<short> ones)
    {
        if (AvxVnni.IsSupported) return AvxVnni.MultiplyWideningAndAdd(acc, w, a);
        return Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(w, a), ones));
    }
}
