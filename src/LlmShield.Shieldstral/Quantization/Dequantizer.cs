// Derived from TensorSharp's ManagedQuantizedOps (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
//
// Block layouts follow ggml-quants.c / ggml-common.h. Every kernel here is
// checked byte-for-byte against the reference `gguf` Python package by
// DequantizerParityTests, which is what keeps a hand-unpacked bitfield honest.
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Quantization;

/// <summary>
/// Turns a run of GGML blocks back into float32. Every type GGUF can carry is
/// supported, including the i-quant codebook families and the ternary types, so
/// a checkpoint quantized by any llama.cpp build loads without a native helper.
/// </summary>
public static unsafe class Dequantizer
{
    /// <summary>True for every <see cref="GgmlType"/> this class can decode.</summary>
    public static bool Supports(GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16 or GgmlType.F64 or
        GgmlType.I8 or GgmlType.I16 or GgmlType.I32 or GgmlType.I64 or
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 or
        GgmlType.Q8_0 or GgmlType.Q8_1 or
        GgmlType.Q2_K or GgmlType.Q3_K or GgmlType.Q4_K or GgmlType.Q5_K or
        GgmlType.Q6_K or GgmlType.Q8_K or
        GgmlType.IQ2_XXS or GgmlType.IQ2_XS or GgmlType.IQ2_S or
        GgmlType.IQ3_XXS or GgmlType.IQ3_S or
        GgmlType.IQ1_S or GgmlType.IQ1_M or
        GgmlType.IQ4_NL or GgmlType.IQ4_XS or
        GgmlType.TQ1_0 or GgmlType.TQ2_0 or GgmlType.MXFP4 => true,
        _ => false,
    };

    /// <summary>
    /// Decodes <paramref name="count"/> consecutive elements starting at the
    /// beginning of a block. <paramref name="count"/> must be a whole number of
    /// blocks for the quantized types.
    /// </summary>
    public static void Dequantize(GgmlType type, byte* src, float* dst, int count)
    {
        int blockSize = GgmlTypeInfo.BlockSize(type);
        if (count % blockSize != 0)
            throw new ArgumentException(
                $"{type} decodes whole blocks of {blockSize}; asked for {count} elements.", nameof(count));

        switch (type)
        {
            case GgmlType.F32: Buffer.MemoryCopy(src, dst, (long)count * 4, (long)count * 4); return;
            case GgmlType.F16: F16(src, dst, count); return;
            case GgmlType.BF16: Bf16(src, dst, count); return;
            case GgmlType.F64: for (int i = 0; i < count; i++) dst[i] = (float)Read<double>(src + i * 8); return;
            case GgmlType.I8: for (int i = 0; i < count; i++) dst[i] = (sbyte)src[i]; return;
            case GgmlType.I16: for (int i = 0; i < count; i++) dst[i] = Read<short>(src + i * 2); return;
            case GgmlType.I32: for (int i = 0; i < count; i++) dst[i] = Read<int>(src + i * 4); return;
            case GgmlType.I64: for (int i = 0; i < count; i++) dst[i] = Read<long>(src + i * 8); return;

            case GgmlType.Q4_0: Q4_0(src, dst, count); return;
            case GgmlType.Q4_1: Q4_1(src, dst, count); return;
            case GgmlType.Q5_0: Q5_0(src, dst, count); return;
            case GgmlType.Q5_1: Q5_1(src, dst, count); return;
            case GgmlType.Q8_0: Q8_0(src, dst, count); return;
            case GgmlType.Q8_1: Q8_1(src, dst, count); return;

            case GgmlType.Q2_K: Q2_K(src, dst, count); return;
            case GgmlType.Q3_K: Q3_K(src, dst, count); return;
            case GgmlType.Q4_K: Q4_K(src, dst, count); return;
            case GgmlType.Q5_K: Q5_K(src, dst, count); return;
            case GgmlType.Q6_K: Q6_K(src, dst, count); return;
            case GgmlType.Q8_K: Q8_K(src, dst, count); return;

            case GgmlType.IQ2_XXS: Iq2Xxs(src, dst, count); return;
            case GgmlType.IQ2_XS: Iq2Xs(src, dst, count); return;
            case GgmlType.IQ2_S: Iq2S(src, dst, count); return;
            case GgmlType.IQ3_XXS: Iq3Xxs(src, dst, count); return;
            case GgmlType.IQ3_S: Iq3S(src, dst, count); return;
            case GgmlType.IQ1_S: Iq1S(src, dst, count); return;
            case GgmlType.IQ1_M: Iq1M(src, dst, count); return;
            case GgmlType.IQ4_NL: Iq4Nl(src, dst, count); return;
            case GgmlType.IQ4_XS: Iq4Xs(src, dst, count); return;

            case GgmlType.TQ1_0: Tq1_0(src, dst, count); return;
            case GgmlType.TQ2_0: Tq2_0(src, dst, count); return;
            case GgmlType.MXFP4: Mxfp4(src, dst, count); return;

            default: throw new NotSupportedException($"No dequantizer for GGML type {type}.");
        }
    }

    public static void Dequantize(GgmlType type, ReadOnlySpan<byte> src, Span<float> dst)
    {
        int count = dst.Length;
        long need = GgmlTypeInfo.RowBytes(type, count);
        if (src.Length < need)
            throw new ArgumentException($"{type}: {count} elements need {need} bytes, got {src.Length}.", nameof(src));
        fixed (byte* s = src)
        fixed (float* d = dst)
            Dequantize(type, s, d, count);
    }

    // ------------------------------------------------------------ scalar types

    private static void F16(byte* src, float* dst, int n)
    {
        // (float)Half is a single hardware conversion on every target this ships
        // on; the JIT keeps it that way, and F16 weights are not the hot path
        // (they get converted once at load, not per token).
        for (int i = 0; i < n; i++)
            dst[i] = (float)Read<Half>(src + i * 2);
    }

    private static void Bf16(byte* src, float* dst, int n)
    {
        for (int i = 0; i < n; i++)
        {
            uint bits = (uint)Read<ushort>(src + i * 2) << 16;
            dst[i] = BitConverter.UInt32BitsToSingle(bits);
        }
    }

    // ----------------------------------------------------------- legacy 32-blocks

    private static void Q4_0(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / 32; b++, src += 18, dst += 32)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            for (int j = 0; j < 16; j++)
            {
                dst[j] = d * ((qs[j] & 0x0F) - 8);
                dst[16 + j] = d * ((qs[j] >> 4) - 8);
            }
        }
    }

    private static void Q4_1(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / 32; b++, src += 20, dst += 32)
        {
            float d = (float)Read<Half>(src);
            float m = (float)Read<Half>(src + 2);
            byte* qs = src + 4;
            for (int j = 0; j < 16; j++)
            {
                dst[j] = d * (qs[j] & 0x0F) + m;
                dst[16 + j] = d * (qs[j] >> 4) + m;
            }
        }
    }

    private static void Q5_0(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / 32; b++, src += 22, dst += 32)
        {
            float d = (float)Read<Half>(src);
            uint qh = Read<uint>(src + 2);
            byte* qs = src + 6;
            for (int j = 0; j < 16; j++)
            {
                int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
                int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
                dst[j] = d * (lo - 16);
                dst[16 + j] = d * (hi - 16);
            }
        }
    }

    private static void Q5_1(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / 32; b++, src += 24, dst += 32)
        {
            float d = (float)Read<Half>(src);
            float m = (float)Read<Half>(src + 2);
            uint qh = Read<uint>(src + 4);
            byte* qs = src + 8;
            for (int j = 0; j < 16; j++)
            {
                int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
                int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
                dst[j] = d * lo + m;
                dst[16 + j] = d * hi + m;
            }
        }
    }

    private static void Q8_0(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / 32; b++, src += 34, dst += 32)
        {
            float d = (float)Read<Half>(src);
            sbyte* qs = (sbyte*)(src + 2);
            if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vd = Vector256.Create(d);
                for (int j = 0; j < 32; j += 8)
                {
                    var q = Vector256.Create(
                        (float)qs[j], qs[j + 1], qs[j + 2], qs[j + 3],
                        qs[j + 4], qs[j + 5], qs[j + 6], qs[j + 7]);
                    Vector256.Store(q * vd, dst + j);
                }
            }
            else
            {
                for (int j = 0; j < 32; j++) dst[j] = d * qs[j];
            }
        }
    }

    private static void Q8_1(byte* src, float* dst, int n)
    {
        // block_q8_1 stores {d, s} as two halves; s is the block sum used by the
        // dot kernels and plays no part in reconstructing the values.
        for (int b = 0; b < n / 32; b++, src += 36, dst += 32)
        {
            float d = (float)Read<Half>(src);
            sbyte* qs = (sbyte*)(src + 4);
            for (int j = 0; j < 32; j++) dst[j] = d * qs[j];
        }
    }

    // ------------------------------------------------------------- k-quants

    private const int QK_K = 256;

    private static void Q2_K(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / QK_K; b++, src += 84, dst += QK_K)
        {
            byte* scales = src;
            byte* qs = src + 16;
            float d = (float)Read<Half>(src + 80);
            float dmin = (float)Read<Half>(src + 82);

            int outIdx = 0;
            for (int g = 0; g < 2; g++)
                for (int s = 0; s < 4; s++)
                    for (int j = 0; j < 32; j++, outIdx++)
                    {
                        int sb = outIdx >> 4;
                        float dl = d * (scales[sb] & 0x0F);
                        float ml = dmin * (scales[sb] >> 4);
                        int q = (qs[g * 32 + j] >> (2 * s)) & 3;
                        dst[outIdx] = dl * q - ml;
                    }
        }
    }

    private static void Q3_K(byte* src, float* dst, int n)
    {
        sbyte* sc = stackalloc sbyte[16];
        for (int b = 0; b < n / QK_K; b++, src += 110, dst += QK_K)
        {
            byte* hmask = src;
            byte* qs = src + 32;
            byte* scales = src + 96;
            float d = (float)Read<Half>(src + 108);

            // 6-bit scales packed as 8 low nibbles-pairs + 4 bytes of high bits.
            for (int k = 0; k < 16; k++)
            {
                int low = scales[k & 7] >> (k < 8 ? 0 : 4);
                int high = scales[8 + (k & 3)] >> (2 * (k >> 2));
                sc[k] = (sbyte)(((low & 0x0F) | ((high & 0x03) << 4)) - 32);
            }

            int outIdx = 0;
            for (int g = 0; g < 2; g++)
                for (int s = 0; s < 4; s++)
                    for (int j = 0; j < 32; j++, outIdx++)
                    {
                        int ql = (qs[g * 32 + j] >> (2 * s)) & 3;
                        // The high bit is stored inverted: a set mask bit means "no -4 offset".
                        int qh = ((hmask[j] >> (g * 4 + s)) & 1) ^ 1;
                        dst[outIdx] = d * sc[outIdx >> 4] * (ql - (qh << 2));
                    }
        }
    }

    /// <summary>
    /// Unpacks Q4_K/Q5_K's 12-byte packed 6-bit scale/min pairs into eight of each.
    /// Layout (llama.cpp <c>get_scale_min_k4</c>):
    /// bytes 0-3 hold scale[0..3] in the low 6 bits, bytes 4-7 hold min[0..3],
    /// and bytes 8-11 carry the low nibbles of scale[4..7] and min[4..7] with the
    /// remaining two bits borrowed from the top of bytes 0-7.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackK4Scales(byte* packed, byte* sc, byte* min)
    {
        for (int i = 0; i < 4; i++)
        {
            sc[i] = (byte)(packed[i] & 0x3F);
            min[i] = (byte)(packed[4 + i] & 0x3F);
            sc[4 + i] = (byte)((packed[8 + i] & 0x0F) | ((packed[i] >> 2) & 0x30));
            min[4 + i] = (byte)((packed[8 + i] >> 4) | ((packed[4 + i] >> 2) & 0x30));
        }
    }

    private static void Q4_K(byte* src, float* dst, int n)
    {
        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        for (int b = 0; b < n / QK_K; b++, src += 144, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            float dmin = (float)Read<Half>(src + 2);
            UnpackK4Scales(src + 4, sc, mn);
            byte* qs = src + 16;

            int outIdx = 0;
            for (int g = 0; g < 4; g++)
                for (int s = 0; s < 2; s++)
                {
                    int sb = g * 2 + s;
                    float dl = d * sc[sb];
                    float ml = dmin * mn[sb];
                    int shift = 4 * s;
                    for (int j = 0; j < 32; j++, outIdx++)
                        dst[outIdx] = dl * ((qs[g * 32 + j] >> shift) & 0x0F) - ml;
                }
        }
    }

    private static void Q5_K(byte* src, float* dst, int n)
    {
        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        for (int b = 0; b < n / QK_K; b++, src += 176, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            float dmin = (float)Read<Half>(src + 2);
            UnpackK4Scales(src + 4, sc, mn);
            byte* qh = src + 16;
            byte* qs = src + 48;

            int outIdx = 0;
            for (int g = 0; g < 4; g++)
                for (int s = 0; s < 2; s++)
                {
                    int sb = g * 2 + s;
                    float dl = d * sc[sb];
                    float ml = dmin * mn[sb];
                    int shift = 4 * s;
                    for (int j = 0; j < 32; j++, outIdx++)
                    {
                        int q = ((qs[g * 32 + j] >> shift) & 0x0F) | (((qh[j] >> sb) & 1) << 4);
                        dst[outIdx] = dl * q - ml;
                    }
                }
        }
    }

    private static void Q6_K(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / QK_K; b++, src += 210, dst += QK_K)
        {
            byte* ql = src;
            byte* qh = src + 128;
            sbyte* scales = (sbyte*)(src + 192);
            float d = (float)Read<Half>(src + 208);

            int outIdx = 0;
            for (int g = 0; g < 2; g++)
                for (int s = 0; s < 2; s++)
                    for (int j = 0; j < 64; j++, outIdx++)
                    {
                        int lo = (ql[g * 64 + j] >> (4 * s)) & 0x0F;
                        // ql walks 64 outputs per (g, s) step but qh only 32, so the
                        // qh shift advances halfway through each ql half-block.
                        int hi = (qh[g * 32 + (j & 31)] >> (2 * (s * 2 + (j >> 5)))) & 0x03;
                        dst[outIdx] = d * scales[outIdx >> 4] * ((lo | (hi << 4)) - 32);
                    }
        }
    }

    private static void Q8_K(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / QK_K; b++, src += 292, dst += QK_K)
        {
            float d = Read<float>(src);
            sbyte* qs = (sbyte*)(src + 4);
            for (int j = 0; j < QK_K; j++) dst[j] = d * qs[j];
        }
    }

    // ---------------------------------------------------------------- i-quants

    /// <summary>Expands one of the 128 sign patterns into eight +1/-1 multipliers.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySigns(byte signMask, float* values)
    {
        for (int t = 0; t < 8; t++)
            if (((signMask >> t) & 1) != 0) values[t] = -values[t];
    }

    private static void Iq2Xxs(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq2XxsGrid;
        ReadOnlySpan<byte> ksigns = QuantGrids.KSigns;
        for (int b = 0; b < n / QK_K; b++, src += 66, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            for (int g = 0; g < 8; g++)
            {
                uint w0 = Read<uint>(qs + g * 8);
                uint w1 = Read<uint>(qs + g * 8 + 4);
                float db = d * (0.5f + (w1 >> 28)) * 0.25f;
                for (int k = 0; k < 4; k++)
                {
                    int gridIdx = (int)((w0 >> (8 * k)) & 0xFF);
                    byte sign = ksigns[(int)((w1 >> (7 * k)) & 0x7F)];
                    float* o = dst + g * 32 + k * 8;
                    for (int t = 0; t < 8; t++) o[t] = db * grid[gridIdx * 8 + t];
                    ApplySigns(sign, o);
                }
            }
        }
    }

    private static void Iq2Xs(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq2XsGrid;
        ReadOnlySpan<byte> ksigns = QuantGrids.KSigns;
        for (int b = 0; b < n / QK_K; b++, src += 74, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            byte* scales = src + 66;
            for (int i = 0; i < 16; i++)
            {
                int scale = (scales[i >> 1] >> (4 * (i & 1))) & 0x0F;
                float db = d * (0.5f + scale) * 0.25f;
                for (int k = 0; k < 2; k++)
                {
                    ushort q = Read<ushort>(qs + (i * 2 + k) * 2);
                    int gridIdx = q & 511;
                    byte sign = ksigns[q >> 9];
                    float* o = dst + i * 16 + k * 8;
                    for (int t = 0; t < 8; t++) o[t] = db * grid[gridIdx * 8 + t];
                    ApplySigns(sign, o);
                }
            }
        }
    }

    private static void Iq2S(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq2SGrid;
        for (int b = 0; b < n / QK_K; b++, src += 82, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            byte* signs = src + 34;
            byte* qh = src + 66;
            byte* scales = src + 74;
            for (int i = 0; i < 16; i++)
            {
                int scale = (scales[i >> 1] >> (4 * (i & 1))) & 0x0F;
                float db = d * (0.5f + scale) * 0.25f;
                for (int k = 0; k < 2; k++)
                {
                    int j = i * 2 + k;
                    int gridIdx = qs[j] | (((qh[j >> 2] >> (2 * (j & 3))) & 3) << 8);
                    float* o = dst + i * 16 + k * 8;
                    for (int t = 0; t < 8; t++) o[t] = db * grid[gridIdx * 8 + t];
                    ApplySigns(signs[j], o);
                }
            }
        }
    }

    private static void Iq3Xxs(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq3XxsGrid;
        ReadOnlySpan<byte> ksigns = QuantGrids.KSigns;
        for (int b = 0; b < n / QK_K; b++, src += 98, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            byte* scales = src + 66;
            for (int g = 0; g < 8; g++)
            {
                uint w = Read<uint>(scales + g * 4);
                float db = d * (0.5f + (w >> 28)) * 0.5f;
                for (int k = 0; k < 4; k++)
                {
                    byte sign = ksigns[(int)((w >> (7 * k)) & 0x7F)];
                    float* o = dst + g * 32 + k * 8;
                    // Each grid entry is 4 values, so an 8-wide slot spans two entries.
                    for (int t = 0; t < 8; t++)
                    {
                        int m = g * 32 + k * 8 + t;
                        o[t] = db * grid[qs[m >> 2] * 4 + (m & 3)];
                    }
                    ApplySigns(sign, o);
                }
            }
        }
    }

    private static void Iq3S(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq3SGrid;
        for (int b = 0; b < n / QK_K; b++, src += 110, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            byte* qh = src + 66;
            byte* signs = src + 74;
            byte* scales = src + 106;
            for (int g = 0; g < 8; g++)
            {
                int scale = (scales[g >> 1] >> (4 * (g & 1))) & 0x0F;
                float db = d * (1 + 2 * scale);
                for (int k = 0; k < 4; k++)
                {
                    float* o = dst + g * 32 + k * 8;
                    for (int t = 0; t < 8; t++)
                    {
                        int m = g * 32 + k * 8 + t;
                        int e = m >> 2;
                        int gridIdx = qs[e] | (((qh[e >> 3] >> (e & 7)) & 1) << 8);
                        o[t] = db * grid[gridIdx * 4 + (m & 3)];
                    }
                    ApplySigns(signs[g * 4 + k], o);
                }
            }
        }
    }

    private const float Iq1Delta = 0.125f;

    private static void Iq1S(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq1SGrid;
        for (int b = 0; b < n / QK_K; b++, src += 50, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            byte* qhBytes = src + 34;
            for (int g = 0; g < 8; g++)
            {
                ushort qh = Read<ushort>(qhBytes + g * 2);
                float dl = d * (2 * ((qh >> 12) & 7) + 1);
                float delta = (qh & 0x8000) == 0 ? Iq1Delta : -Iq1Delta;
                for (int k = 0; k < 4; k++)
                {
                    int gridIdx = qs[g * 4 + k] | (((qh >> (3 * k)) & 7) << 8);
                    float* o = dst + g * 32 + k * 8;
                    for (int t = 0; t < 8; t++) o[t] = dl * (grid[gridIdx * 8 + t] + delta);
                }
            }
        }
    }

    private static void Iq1M(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> grid = QuantGrids.Iq1SGrid;
        for (int b = 0; b < n / QK_K; b++, src += 56, dst += QK_K)
        {
            byte* qs = src;
            byte* qh = src + 32;
            byte* scaleBytes = src + 48;

            // IQ1_M is the one type whose f16 scale is scattered: the top nibble of
            // each of the four scale halfwords supplies one nibble of the exponent
            // and mantissa, most significant first.
            ushort dBits = 0;
            for (int i = 0; i < 4; i++)
            {
                ushort s = Read<ushort>(scaleBytes + i * 2);
                dBits |= (ushort)((s & 0xF000) >> (12 - 4 * i));
            }
            float d = (float)BitConverter.UInt16BitsToHalf(dBits);

            for (int m = 0; m < 32; m++)
            {
                int scaleIdx = m >> 1;                       // 16 six-bit-ish scales
                ushort s = Read<ushort>(scaleBytes + (scaleIdx >> 2) * 2);
                int scale = (s >> (3 * (scaleIdx & 3))) & 7;
                float dl = d * (2 * scale + 1);

                int qhNibble = (qh[m >> 1] >> (4 * (m & 1))) & 0x0F;
                int gridIdx = qs[m] | ((qhNibble & 7) << 8);
                float delta = (qhNibble & 8) == 0 ? Iq1Delta : -Iq1Delta;

                float* o = dst + m * 8;
                for (int t = 0; t < 8; t++) o[t] = dl * (grid[gridIdx * 8 + t] + delta);
            }
        }
    }

    private static void Iq4Nl(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> kv = QuantGrids.Iq4NlValues;
        for (int b = 0; b < n / 32; b++, src += 18, dst += 32)
        {
            float d = (float)Read<Half>(src);
            byte* qs = src + 2;
            for (int j = 0; j < 16; j++)
            {
                dst[j] = d * kv[qs[j] & 0x0F];
                dst[16 + j] = d * kv[qs[j] >> 4];
            }
        }
    }

    private static void Iq4Xs(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> kv = QuantGrids.Iq4NlValues;
        for (int b = 0; b < n / QK_K; b++, src += 136, dst += QK_K)
        {
            float d = (float)Read<Half>(src);
            ushort scalesH = Read<ushort>(src + 2);
            byte* scalesL = src + 4;
            byte* qs = src + 8;

            for (int g = 0; g < 8; g++)
            {
                int lo = (scalesL[g >> 1] >> (4 * (g & 1))) & 0x0F;
                int hi = (scalesH >> (2 * g)) & 0x03;
                int scale = (sbyte)(lo | (hi << 4)) - 32;
                float dl = d * scale;
                for (int s = 0; s < 2; s++)
                    for (int j = 0; j < 16; j++)
                        dst[g * 32 + s * 16 + j] = dl * kv[(qs[g * 16 + j] >> (4 * s)) & 0x0F];
            }
        }
    }

    // ---------------------------------------------------------------- ternary

    private static void Tq1_0(byte* src, float* dst, int n)
    {
        // Five ternary digits are base-3 packed into each byte; the reference
        // decoder recovers a digit with the fixed-point trick (x * 3) >> 8 after
        // scaling by the corresponding power of three, all in wrapping uint8.
        ReadOnlySpan<byte> pow3 = [1, 3, 9, 27, 81];
        byte* work = stackalloc byte[QK_K];
        for (int b = 0; b < n / QK_K; b++, src += 54, dst += QK_K)
        {
            byte* qs = src;
            byte* qh = src + 48;
            float d = (float)Read<Half>(src + 52);

            int o = 0;
            for (int s = 0; s < 5; s++)
                for (int j = 0; j < 32; j++) work[o++] = (byte)(qs[j] * pow3[s]);
            for (int s = 0; s < 5; s++)
                for (int j = 0; j < 16; j++) work[o++] = (byte)(qs[32 + j] * pow3[s]);
            for (int s = 0; s < 4; s++)
                for (int j = 0; j < 4; j++) work[o++] = (byte)(qh[j] * pow3[s]);

            for (int i = 0; i < QK_K; i++)
                dst[i] = d * (((work[i] * 3) >> 8) - 1);
        }
    }

    private static void Tq2_0(byte* src, float* dst, int n)
    {
        for (int b = 0; b < n / QK_K; b++, src += 66, dst += QK_K)
        {
            byte* qs = src;
            float d = (float)Read<Half>(src + 64);
            int outIdx = 0;
            for (int g = 0; g < 2; g++)
                for (int s = 0; s < 4; s++)
                    for (int j = 0; j < 32; j++, outIdx++)
                        dst[outIdx] = d * (((qs[g * 32 + j] >> (2 * s)) & 3) - 1);
        }
    }

    // ------------------------------------------------------------------ MXFP4

    /// <summary>ggml's <c>ggml_e8m0_to_fp32_half</c>: an 8-bit exponent to 2^(e-127)/2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float E8m0ToFp32Half(byte e)
    {
        uint bits = e < 2 ? 0x00200000u << e : (uint)(e - 1) << 23;
        return BitConverter.UInt32BitsToSingle(bits);
    }

    private static void Mxfp4(byte* src, float* dst, int n)
    {
        ReadOnlySpan<sbyte> kv = QuantGrids.Mxfp4Values;
        for (int b = 0; b < n / 32; b++, src += 17, dst += 32)
        {
            float d = E8m0ToFp32Half(src[0]);
            byte* qs = src + 1;
            for (int j = 0; j < 16; j++)
            {
                dst[j] = d * kv[qs[j] & 0x0F];
                dst[16 + j] = d * kv[qs[j] >> 4];
            }
        }
    }

    // ----------------------------------------------------------------- helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Read<T>(byte* p) where T : unmanaged => Unsafe.ReadUnaligned<T>(p);
}
