// Integer dot products between a quantized weight row and a Q8-quantized
// activation row — the arithmetic ggml uses on CPU.
//
// Derived in approach from TensorSharp's ManagedQuantizedOps
// (https://github.com/zhongkaifu/TensorSharp), BSD-3-Clause.
// See third-party/TensorSharp-LICENSE.
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Quantization;

/// <summary>
/// Dots a quantized weight row against activations quantized to Q8_0, without
/// materialising either side as float32.
///
/// The generic path in <see cref="Dequantizer"/> reconstructs each weight to
/// float and multiply-adds; that costs a widen, a scale and an FMA per weight.
/// Here the same block is multiplied in 8-bit and accumulated in 32-bit, and the
/// two block scales are applied once per 32 weights instead of once per weight.
///
/// The catch is that it only works where the weight's quantization has a single
/// multiplicative scale per block (optionally plus an offset that factors out of
/// the sum). That covers the legacy families; the k-quants and i-quants carry
/// per-sub-block scales that would need their own kernels, and they fall back to
/// the generic path — <see cref="Supports"/> is the predicate.
/// </summary>
public static unsafe class IntegerDot
{
    /// <summary>Bytes one Q8_0-quantized activation block occupies: an f16 scale plus 32 int8.</summary>
    public const int ActivationBlockBytes = 34;
    public const int BlockSize = 32;

    /// <summary>Weight types this can dot directly against Q8_0 activations.</summary>
    public static bool Supports(GgmlType type) => type switch
    {
        GgmlType.Q8_0 or GgmlType.Q4_0 or GgmlType.Q4_1 or
        GgmlType.Q5_0 or GgmlType.Q5_1 => true,
        _ => false,
    };

    /// <summary>
    /// Quantizes <paramref name="count"/> activations into Q8_0 blocks.
    /// <paramref name="count"/> must be a multiple of 32.
    /// </summary>
    /// <remarks>
    /// Round-half-away-from-zero, matching ggml's <c>quantize_row_q8_0_ref</c>. The
    /// activations are re-quantized once per matmul call and then reused across
    /// every output row, so this cost is amortised over thousands of dots.
    /// </remarks>
    public static void QuantizeActivations(ReadOnlySpan<float> source, Span<byte> destination)
        => QuantizeActivations(source, destination, default);

    /// <param name="blockSums">
    /// Optional, one entry per block: the block's scale times the sum of its
    /// quantized values. Q4_1 and Q5_1 need exactly this to apply their per-block
    /// offset, and it costs nothing to accumulate while the values are in registers.
    /// </param>
    public static void QuantizeActivations(ReadOnlySpan<float> source, Span<byte> destination,
        Span<float> blockSums)
    {
        int blocks = source.Length / BlockSize;
        fixed (float* src = source)
        fixed (byte* dst = destination)
        fixed (float* sums = blockSums)
        {
            for (int b = 0; b < blocks; b++)
            {
                float* x = src + b * BlockSize;
                byte* out_ = dst + b * ActivationBlockBytes;

                float max = 0;
                for (int i = 0; i < BlockSize; i++)
                {
                    float a = MathF.Abs(x[i]);
                    if (a > max) max = a;
                }

                float d = max / 127f;
                float inverse = d != 0f ? 1f / d : 0f;
                Unsafe.WriteUnaligned(out_, (Half)d);

                sbyte* q = (sbyte*)(out_ + 2);
                int sum = 0;
                for (int i = 0; i < BlockSize; i++)
                {
                    float scaled = x[i] * inverse;
                    // MathF.Round(.., AwayFromZero) is what ggml's roundf does.
                    // Clamping to 127 (not 128) keeps every product inside int16.
                    sbyte v = (sbyte)Math.Clamp((int)MathF.Round(scaled, MidpointRounding.AwayFromZero), -127, 127);
                    q[i] = v;
                    sum += v;
                }
                if (sums is not null) sums[b] = d * sum;
            }
        }
    }

    /// <summary>True when the type carries a per-block offset as well as a scale.</summary>
    public static bool HasOffset(GgmlType type) => type is GgmlType.Q4_1 or GgmlType.Q5_1;

    /// <summary>
    /// Unpacks one weight row into plain int8 plus per-block scales (and offsets,
    /// where the type has them).
    ///
    /// Splitting this from the dot is the whole point. Unpacking nibbles is per
    /// *weight* work; dotting is per (weight, token). Doing them together means a
    /// 64-token prefill unpacks the same row 64 times — which is how an integer
    /// kernel ends up losing to a float one that decodes once and reuses.
    /// </summary>
    /// <param name="offsets">May be null for the types with no offset.</param>
    public static void UnpackRow(GgmlType type, byte* weights, sbyte* values,
        float* scales, float* offsets, int blocks)
    {
        switch (type)
        {
            case GgmlType.Q8_0:
                for (int b = 0; b < blocks; b++, weights += 34, values += BlockSize)
                {
                    scales[b] = (float)Read<Half>(weights);
                    // CopyBlockUnaligned is one IL instruction; Buffer.MemoryCopy is
                    // a call with overlap checks, and at 32 bytes a time that call
                    // overhead is most of the work.
                    Unsafe.CopyBlockUnaligned(values, weights + 2, BlockSize);
                }
                return;

            case GgmlType.Q4_0:
                for (int b = 0; b < blocks; b++, weights += 18, values += BlockSize)
                {
                    scales[b] = (float)Read<Half>(weights);
                    byte* qs = weights + 2;
                    for (int j = 0; j < 16; j++)
                    {
                        values[j] = (sbyte)((qs[j] & 0x0F) - 8);
                        values[16 + j] = (sbyte)((qs[j] >> 4) - 8);
                    }
                }
                return;

            case GgmlType.Q4_1:
                for (int b = 0; b < blocks; b++, weights += 20, values += BlockSize)
                {
                    scales[b] = (float)Read<Half>(weights);
                    offsets[b] = (float)Read<Half>(weights + 2);
                    byte* qs = weights + 4;
                    for (int j = 0; j < 16; j++)
                    {
                        values[j] = (sbyte)(qs[j] & 0x0F);
                        values[16 + j] = (sbyte)(qs[j] >> 4);
                    }
                }
                return;

            case GgmlType.Q5_0:
                for (int b = 0; b < blocks; b++, weights += 22, values += BlockSize)
                {
                    scales[b] = (float)Read<Half>(weights);
                    uint qh = Read<uint>(weights + 2);
                    byte* qs = weights + 6;
                    for (int j = 0; j < 16; j++)
                    {
                        values[j] = (sbyte)(((qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4)) - 16);
                        values[16 + j] = (sbyte)(((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4)) - 16);
                    }
                }
                return;

            case GgmlType.Q5_1:
                for (int b = 0; b < blocks; b++, weights += 24, values += BlockSize)
                {
                    scales[b] = (float)Read<Half>(weights);
                    offsets[b] = (float)Read<Half>(weights + 2);
                    uint qh = Read<uint>(weights + 4);
                    byte* qs = weights + 8;
                    for (int j = 0; j < 16; j++)
                    {
                        values[j] = (sbyte)((qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4));
                        values[16 + j] = (sbyte)((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4));
                    }
                }
                return;

            default:
                throw new NotSupportedException($"{type} has no integer unpack kernel.");
        }
    }

    /// <summary>
    /// Dots an unpacked weight row against one Q8_0-quantized activation row.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The accumulator is a float vector carried across all blocks, with a single
    /// horizontal reduction at the end. Reducing per block instead — the obvious
    /// way to write this — costs a shuffle chain every 32 weights and is enough on
    /// its own to make the integer path lose to the float one.
    /// </para>
    /// <para>
    /// The per-block offset, where there is one, is constant across the block, so
    /// it contributes <c>offset · Σactivations</c> and is folded in separately by
    /// the caller rather than added to every weight.
    /// </para>
    /// </remarks>
    public static float DotUnpacked(sbyte* values, float* scales, byte* activations, int blocks)
    {
        if (Avx2.IsSupported)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);
            for (int b = 0; b < blocks; b++, values += BlockSize, activations += ActivationBlockBytes)
            {
                Vector256<sbyte> w = Vector256.Load(values);
                Vector256<sbyte> a = Vector256.Load((sbyte*)(activations + 2));

                // vpmaddubsw wants an unsigned left operand, so move the weight's
                // sign onto the activation: |w| · (sign(w)·a) == w · a. Products
                // reach 127·127 and are summed in pairs, so 16 bits is exactly
                // enough — which is why Q8_0 clamps to ±127 rather than ±128.
                Vector256<short> pairs = Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(a, w));
                Vector256<int> sums = Avx2.MultiplyAddAdjacent(pairs, ones);

                float scale = scales[b] * (float)Read<Half>(activations);
                acc = Avx.Add(acc, Avx.Multiply(Vector256.ConvertToSingle(sums), Vector256.Create(scale)));
            }
            return Vector256.Sum(acc);
        }

        float total = 0;
        for (int b = 0; b < blocks; b++, values += BlockSize, activations += ActivationBlockBytes)
            total += scales[b] * (float)Read<Half>(activations)
                   * SumProducts(values, (sbyte*)(activations + 2));
        return total;
    }

    /// <summary>
    /// Q8_0 against Q8_0, straight out of both packed rows.
    ///
    /// Q8_0's payload is already plain int8, so unpacking it only copies bytes
    /// around. Reading the interleaved <c>{scale, 32 values}</c> layout in place
    /// skips that entirely — which matters most at one token, where there is no
    /// second token to amortise a copy over.
    /// </summary>
    public static float DotPackedQ8_0(byte* weights, byte* activations, int blocks)
    {
        if (Avx2.IsSupported)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);
            for (int b = 0; b < blocks; b++, weights += 34, activations += ActivationBlockBytes)
            {
                Vector256<sbyte> w = Vector256.Load((sbyte*)(weights + 2));
                Vector256<sbyte> a = Vector256.Load((sbyte*)(activations + 2));
                Vector256<int> sums = Avx2.MultiplyAddAdjacent(
                    Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(a, w)), ones);
                float scale = (float)Read<Half>(weights) * (float)Read<Half>(activations);
                acc = Avx.Add(acc, Avx.Multiply(Vector256.ConvertToSingle(sums), Vector256.Create(scale)));
            }
            return Vector256.Sum(acc);
        }

        float total = 0;
        for (int b = 0; b < blocks; b++, weights += 34, activations += ActivationBlockBytes)
            total += (float)Read<Half>(weights) * (float)Read<Half>(activations)
                   * SumProducts((sbyte*)(weights + 2), (sbyte*)(activations + 2));
        return total;
    }

    /// <summary>
    /// <c>Σ_b offsets[b] · activationSums[b]</c> — the contribution of a weight
    /// type's per-block offset, which does not depend on the weight payload at all.
    /// </summary>
    public static float OffsetContribution(float* offsets, float* activationSums, int blocks)
    {
        float total = 0;
        for (int b = 0; b < blocks; b++) total += offsets[b] * activationSums[b];
        return total;
    }

    /// <summary>
    /// Convenience path that unpacks and dots in one call — for a single token,
    /// where there is nothing to amortise the unpack over.
    /// </summary>
    public static float Dot(GgmlType type, byte* weights, byte* activations, int count)
    {
        int blocks = count / BlockSize;
        sbyte* values = stackalloc sbyte[count];
        float* scales = stackalloc float[blocks];
        float* offsetStorage = stackalloc float[blocks];
        float* offsets = HasOffset(type) ? offsetStorage : null;
        UnpackRow(type, weights, values, scales, offsets, blocks);

        float total = DotUnpacked(values, scales, activations, blocks);
        if (offsets is not null)
        {
            float* sums = stackalloc float[blocks];
            for (int b = 0; b < blocks; b++)
                sums[b] = (float)Read<Half>(activations + b * ActivationBlockBytes)
                        * SumInt8((sbyte*)(activations + b * ActivationBlockBytes + 2));
            total += OffsetContribution(offsets, sums, blocks);
        }
        return total;
    }

    // ------------------------------------------------------------ primitives

    /// <summary>
    /// Σ a[i]·b[i] over one 32-element block, in 32-bit integers.
    /// </summary>
    /// <remarks>
    /// Each product is at most 127·127 = 16129, so the multiply is exact in 16 bits
    /// and the 32-element sum cannot exceed 516128 — no saturation anywhere, which
    /// is why this needs no intermediate rescaling.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SumProducts(sbyte* a, sbyte* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<sbyte> va = Vector256.Load(a);
            Vector256<sbyte> vb = Vector256.Load(b);
            (Vector256<short> aLo, Vector256<short> aHi) = Vector256.Widen(va);
            (Vector256<short> bLo, Vector256<short> bHi) = Vector256.Widen(vb);
            (Vector256<int> p0, Vector256<int> p1) = Vector256.Widen(aLo * bLo);
            (Vector256<int> p2, Vector256<int> p3) = Vector256.Widen(aHi * bHi);
            return Vector256.Sum(p0 + p1 + p2 + p3);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<int> acc = Vector128<int>.Zero;
            for (int i = 0; i < BlockSize; i += 16)
            {
                Vector128<sbyte> va = Vector128.Load(a + i);
                Vector128<sbyte> vb = Vector128.Load(b + i);
                (Vector128<short> aLo, Vector128<short> aHi) = Vector128.Widen(va);
                (Vector128<short> bLo, Vector128<short> bHi) = Vector128.Widen(vb);
                (Vector128<int> p0, Vector128<int> p1) = Vector128.Widen(aLo * bLo);
                (Vector128<int> p2, Vector128<int> p3) = Vector128.Widen(aHi * bHi);
                acc += p0 + p1 + p2 + p3;
            }
            return Vector128.Sum(acc);
        }

        int sum = 0;
        for (int i = 0; i < BlockSize; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Σ b[i] over one block — the activation sum the Q4_1/Q5_1 offset needs.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SumInt8(sbyte* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            (Vector256<short> lo, Vector256<short> hi) = Vector256.Widen(Vector256.Load(b));
            (Vector256<int> a0, Vector256<int> a1) = Vector256.Widen(lo);
            (Vector256<int> a2, Vector256<int> a3) = Vector256.Widen(hi);
            return Vector256.Sum(a0 + a1 + a2 + a3);
        }
        int sum = 0;
        for (int i = 0; i < BlockSize; i++) sum += b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Read<T>(byte* p) where T : unmanaged => Unsafe.ReadUnaligned<T>(p);
}
