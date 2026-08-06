namespace LlmShield.Shieldstral.Model;

/// <summary>
/// Per-layer key/value cache.
///
/// Layout is <c>[head][position][dim]</c> so one head's history is contiguous —
/// attention scans a single head across all positions, and that is the access
/// pattern the score loop wants. Capacity doubles on demand rather than being
/// sized for the model's 256k advertised context, which would otherwise reserve
/// tens of gigabytes for prompts that are typically a few hundred tokens.
/// </summary>
public sealed class KvCache
{
    private readonly int _layers;
    private readonly int _kvHeads;
    private readonly int _headDim;

    private float[][] _keys;    // [layer][head * capacity * headDim]
    private float[][] _values;

    public int Capacity { get; private set; }
    /// <summary>Number of positions currently populated.</summary>
    public int Length { get; private set; }

    public KvCache(int layers, int kvHeads, int headDim, int initialCapacity = 512)
    {
        _layers = layers;
        _kvHeads = kvHeads;
        _headDim = headDim;
        Capacity = Math.Max(1, initialCapacity);
        _keys = Allocate(Capacity);
        _values = Allocate(Capacity);
    }

    private float[][] Allocate(int capacity)
    {
        var arrays = new float[_layers][];
        for (int l = 0; l < _layers; l++)
            arrays[l] = new float[(long)_kvHeads * capacity * _headDim];
        return arrays;
    }

    public void Reset() => Length = 0;

    /// <summary>Drops everything past <paramref name="length"/> positions.</summary>
    public void Truncate(int length) => Length = Math.Clamp(length, 0, Length);

    public void EnsureCapacity(int required)
    {
        if (required <= Capacity) return;

        int capacity = Capacity;
        while (capacity < required) capacity *= 2;

        float[][] keys = Allocate(capacity);
        float[][] values = Allocate(capacity);
        for (int l = 0; l < _layers; l++)
            for (int h = 0; h < _kvHeads; h++)
            {
                long from = (long)h * Capacity * _headDim;
                long to = (long)h * capacity * _headDim;
                long count = (long)Length * _headDim;
                Array.Copy(_keys[l], from, keys[l], to, count);
                Array.Copy(_values[l], from, values[l], to, count);
            }

        _keys = keys;
        _values = values;
        Capacity = capacity;
    }

    /// <summary>Marks <paramref name="count"/> more positions as written.</summary>
    public void Advance(int count) => Length += count;

    private long Offset(int head, int position) => ((long)head * Capacity + position) * _headDim;

    public Span<float> Key(int layer, int head, int position)
        => _keys[layer].AsSpan(checked((int)Offset(head, position)), _headDim);

    public Span<float> Value(int layer, int head, int position)
        => _values[layer].AsSpan(checked((int)Offset(head, position)), _headDim);

    /// <summary>All cached keys for one head, laid out <c>[position][dim]</c>.</summary>
    public ReadOnlySpan<float> KeyHistory(int layer, int head, int length)
        => _keys[layer].AsSpan(checked((int)Offset(head, 0)), length * _headDim);

    public ReadOnlySpan<float> ValueHistory(int layer, int head, int length)
        => _values[layer].AsSpan(checked((int)Offset(head, 0)), length * _headDim);

    /// <summary>
    /// Copies the first <paramref name="length"/> positions of every layer and head
    /// into a flat buffer: keys for all layers, then values. Used by the prefix cache.
    /// </summary>
    public float[] Snapshot(int length)
    {
        if (length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), $"Only {Length} positions are cached.");
        long per = (long)_layers * _kvHeads * length * _headDim;
        var buffer = new float[per * 2];
        long cursor = 0;
        for (int pass = 0; pass < 2; pass++)
        {
            float[][] source = pass == 0 ? _keys : _values;
            for (int l = 0; l < _layers; l++)
                for (int h = 0; h < _kvHeads; h++)
                {
                    Array.Copy(source[l], Offset(h, 0), buffer, cursor, (long)length * _headDim);
                    cursor += (long)length * _headDim;
                }
        }
        return buffer;
    }

    /// <summary>Inverse of <see cref="Snapshot"/>; leaves the cache holding exactly that prefix.</summary>
    public void Restore(ReadOnlySpan<float> snapshot, int length)
    {
        long per = (long)_layers * _kvHeads * length * _headDim;
        if (snapshot.Length != per * 2)
            throw new ArgumentException(
                $"Snapshot holds {snapshot.Length} floats; {length} positions need {per * 2}.", nameof(snapshot));

        EnsureCapacity(length);
        int cursor = 0;
        for (int pass = 0; pass < 2; pass++)
        {
            float[][] destination = pass == 0 ? _keys : _values;
            for (int l = 0; l < _layers; l++)
                for (int h = 0; h < _kvHeads; h++)
                {
                    snapshot.Slice(cursor, length * _headDim)
                        .CopyTo(destination[l].AsSpan(checked((int)Offset(h, 0)), length * _headDim));
                    cursor += length * _headDim;
                }
        }
        Length = length;
    }

    /// <summary>Bytes one position of cache occupies across every layer (keys + values).</summary>
    public long BytesPerPosition => (long)_layers * _kvHeads * _headDim * sizeof(float) * 2;
}
