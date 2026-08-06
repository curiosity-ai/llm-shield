// Derived from TensorSharp (https://github.com/zhongkaifu/TensorSharp), BSD-3-Clause.
// See third-party/TensorSharp-LICENSE.
namespace LlmShield.Shieldstral.Gguf;

/// <summary>
/// GGML tensor element types, numbered exactly as <c>ggml_type</c> in ggml.h.
/// The gaps (4, 5, 31-33, 36-38) are types ggml has retired; keeping the holes
/// means a GGUF written by any llama.cpp build still resolves to the same name.
/// </summary>
public enum GgmlType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

/// <summary>
/// Block geometry for every <see cref="GgmlType"/>: how many logical elements a
/// block covers and how many bytes it occupies on disk. Everything that walks a
/// quantized row — the reader's byte-count maths, the dequantizers, the dot
/// kernels — derives its strides from here, so a single table keeps them in step.
/// </summary>
public static class GgmlTypeInfo
{
    /// <summary>Elements per quantization block (1 for the unquantized types).</summary>
    public static int BlockSize(GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16 or GgmlType.F64 or
        GgmlType.I8 or GgmlType.I16 or GgmlType.I32 or GgmlType.I64 => 1,

        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 or
        GgmlType.Q8_0 or GgmlType.Q8_1 or GgmlType.IQ4_NL or GgmlType.MXFP4 => 32,

        _ => 256,
    };

    /// <summary>Bytes one block occupies.</summary>
    public static int TypeSize(GgmlType type) => type switch
    {
        GgmlType.F32 => 4,
        GgmlType.F16 => 2,
        GgmlType.BF16 => 2,
        GgmlType.F64 => 8,
        GgmlType.I8 => 1,
        GgmlType.I16 => 2,
        GgmlType.I32 => 4,
        GgmlType.I64 => 8,

        GgmlType.Q4_0 => 2 + 16,                    // d, 32x4bit
        GgmlType.Q4_1 => 2 + 2 + 16,                // d, m, 32x4bit
        GgmlType.Q5_0 => 2 + 4 + 16,                // d, qh, 32x4bit
        GgmlType.Q5_1 => 2 + 2 + 4 + 16,            // d, m, qh, 32x4bit
        GgmlType.Q8_0 => 2 + 32,                    // d, 32x8bit
        GgmlType.Q8_1 => 2 + 2 + 32,                // d, s, 32x8bit

        GgmlType.Q2_K => 16 + 64 + 2 + 2,           // scales, qs, d, dmin
        GgmlType.Q3_K => 32 + 64 + 12 + 2,          // hmask, qs, scales, d
        GgmlType.Q4_K => 2 + 2 + 12 + 128,          // d, dmin, scales, qs
        GgmlType.Q5_K => 2 + 2 + 12 + 32 + 128,     // d, dmin, scales, qh, qs
        GgmlType.Q6_K => 128 + 64 + 16 + 2,         // ql, qh, scales, d
        GgmlType.Q8_K => 4 + 256 + 32,              // d, qs, bsums

        GgmlType.IQ2_XXS => 2 + 64,                 // 66
        GgmlType.IQ2_XS => 2 + 64 + 8,              // 74
        GgmlType.IQ3_XXS => 2 + 96,                 // 98
        GgmlType.IQ1_S => 2 + 32 + 16,              // 50
        GgmlType.IQ4_NL => 2 + 16,                  // 18
        GgmlType.IQ3_S => 2 + 104 + 4,              // 110
        GgmlType.IQ2_S => 2 + 64 + 16,              // 82
        GgmlType.IQ4_XS => 2 + 2 + 4 + 128,         // 136
        GgmlType.IQ1_M => 32 + 16 + 8,              // 56
        GgmlType.TQ1_0 => 2 + 4 + 48,               // 54
        GgmlType.TQ2_0 => 2 + 64,                   // 66
        GgmlType.MXFP4 => 1 + 16,                   // 17

        _ => throw new NotSupportedException($"Unknown GGML tensor type: {type}"),
    };

    /// <summary>True when the type stores quantized blocks rather than plain scalars.</summary>
    public static bool IsQuantized(GgmlType type) => BlockSize(type) > 1;

    /// <summary>Bytes occupied by <paramref name="elementCount"/> contiguous elements.</summary>
    public static long RowBytes(GgmlType type, long elementCount)
    {
        int block = BlockSize(type);
        if (elementCount % block != 0)
            throw new ArgumentException(
                $"{type} rows must be a multiple of {block} elements; got {elementCount}.",
                nameof(elementCount));
        return elementCount / block * TypeSize(type);
    }

    /// <summary>Canonical GGUF spelling, e.g. <c>Q4_K</c>. Used in file names and logs.</summary>
    public static string Name(GgmlType type) => type.ToString();

    /// <summary>Parses a name produced by <see cref="Name"/>, case-insensitively.</summary>
    public static bool TryParse(string text, out GgmlType type)
        => Enum.TryParse(text, ignoreCase: true, out type) && Enum.IsDefined(type);
}
