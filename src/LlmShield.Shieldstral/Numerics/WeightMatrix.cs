using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Quantization;

namespace LlmShield.Shieldstral.Numerics;

/// <summary>
/// A 2-D weight still in its on-disk GGUF form, pointing straight into the
/// memory-mapped file.
///
/// GGUF stores a linear layer as <c>[in_features, out_features]</c> with the
/// input axis fastest-varying, so one output feature is one contiguous
/// (possibly quantized) row of <see cref="Cols"/> values. That is exactly the
/// layout a row-at-a-time dot product wants, which is why nothing here ever
/// transposes or repacks.
/// </summary>
public readonly unsafe struct WeightMatrix
{
    public readonly GgmlType Type;
    public readonly byte* Data;
    /// <summary>Output features — the number of rows.</summary>
    public readonly int Rows;
    /// <summary>Input features — elements per row.</summary>
    public readonly int Cols;
    /// <summary>Bytes per row, which for a quantized type is less than <c>Cols * 4</c>.</summary>
    public readonly int RowBytes;

    public WeightMatrix(GgmlType type, byte* data, int rows, int cols)
    {
        if (!Dequantizer.Supports(type))
            throw new NotSupportedException($"Weights of type {type} cannot be decoded.");
        Type = type;
        Data = data;
        Rows = rows;
        Cols = cols;
        RowBytes = checked((int)GgmlTypeInfo.RowBytes(type, cols));
    }

    public bool IsEmpty => Data is null;

    public byte* Row(int index) => Data + (long)index * RowBytes;

    /// <summary>Reads one weight row into <paramref name="destination"/> as float32.</summary>
    public void DequantizeRow(int index, Span<float> destination)
    {
        fixed (float* d = destination)
            Dequantizer.Dequantize(Type, Row(index), d, Cols);
    }

    /// <summary>
    /// Binds a GGUF tensor as a weight matrix. Optional tensors (a tied
    /// <c>output.weight</c>, a missing bias) come back with <see cref="IsEmpty"/>.
    /// </summary>
    public static WeightMatrix From(GgufFile file, string name, bool required = true)
    {
        if (!file.TryGetTensor(name, out GgufTensorInfo? info))
        {
            if (required) throw new KeyNotFoundException($"{file.Path} is missing tensor '{name}'.");
            return default;
        }
        if (info.Shape.Length != 2)
            throw new InvalidDataException(
                $"'{name}' has {info.Shape.Length} dimensions; a weight matrix needs 2.");
        return new WeightMatrix(info.Type, file.GetTensorPointer(info),
            rows: checked((int)info.Shape[1]), cols: checked((int)info.Shape[0]));
    }
}

/// <summary>A 1-D parameter (norm weight, bias) materialised as float32 once at load.</summary>
public sealed class VectorParameter
{
    public float[] Values { get; }
    public int Length => Values.Length;

    private VectorParameter(float[] values) => Values = values;

    public static VectorParameter From(GgufFile file, string name, bool required = true)
    {
        if (!file.TryGetTensor(name, out GgufTensorInfo? info))
        {
            if (required) throw new KeyNotFoundException($"{file.Path} is missing tensor '{name}'.");
            return new VectorParameter([]);
        }
        var values = new float[info.ElementCount];
        unsafe
        {
            fixed (float* d = values)
                Dequantizer.Dequantize(info.Type, file.GetTensorPointer(info), d, values.Length);
        }
        return new VectorParameter(values);
    }

    public static implicit operator ReadOnlySpan<float>(VectorParameter p) => p.Values;
}
