// Derived from TensorSharp's GgufReader (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Text;

namespace LlmShield.Shieldstral.Gguf;

/// <summary>GGUF metadata value types (spec section "General structure").</summary>
public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3,
    UInt32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
    String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
}

/// <summary>One entry of a GGUF tensor table.</summary>
public sealed class GgufTensorInfo
{
    public required string Name { get; init; }
    /// <summary>Dimensions in GGUF order: <c>Shape[0]</c> is the fastest-varying axis.</summary>
    public required long[] Shape { get; init; }
    public required GgmlType Type { get; init; }
    /// <summary>Byte offset from the file's tensor-data section.</summary>
    public required long Offset { get; init; }

    public long ElementCount
    {
        get { long n = 1; foreach (long d in Shape) n *= d; return n; }
    }

    /// <summary>Elements per row (the fastest-varying dimension).</summary>
    public long RowElements => Shape.Length > 0 ? Shape[0] : 0;

    /// <summary>Number of rows, i.e. the product of every dimension above the first.</summary>
    public long RowCount
    {
        get { long n = 1; for (int i = 1; i < Shape.Length; i++) n *= Shape[i]; return n; }
    }

    public long ByteCount => GgmlTypeInfo.RowBytes(Type, RowElements) * RowCount;

    public override string ToString()
        => $"{Name} [{string.Join(", ", Shape)}] {Type}";
}

/// <summary>
/// Memory-mapped GGUF reader.
///
/// Weights are never copied into the managed heap: the whole file is mapped once
/// and every tensor is handed out as a <see cref="ReadOnlySpan{Byte}"/> over that
/// mapping. A quantized 3B checkpoint therefore costs its on-disk size in shared,
/// evictable page cache rather than that plus a private dequantized copy.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF"

    private MemoryMappedFile? _map;
    private MemoryMappedViewAccessor? _view;
    private byte* _base;
    private bool _pointerAcquired;
    private readonly long _length;

    public string Path { get; }
    public uint Version { get; }
    public IReadOnlyDictionary<string, object> Metadata => _metadata;
    public IReadOnlyDictionary<string, GgufTensorInfo> Tensors => _tensors;
    /// <summary>File offset at which the tensor-data section starts.</summary>
    public long DataOffset { get; }

    private readonly Dictionary<string, object> _metadata = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgufTensorInfo> _tensors = new(StringComparer.Ordinal);

    public GgufFile(string path)
    {
        Path = path;
        var info = new FileInfo(path);
        if (!info.Exists) throw new FileNotFoundException($"GGUF not found: {path}", path);
        _length = info.Length;

        // Map read-only and share-read: several GgufFile instances over the same
        // checkpoint (model + mmproj, or a test opening it twice) must not collide.
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        _map = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read,
            HandleInheritability.None, leaveOpen: false);
        _view = _map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _pointerAcquired = true;
        _base = p + _view.PointerOffset;

        long cursor = 0;
        uint magic = ReadU32(ref cursor);
        if (magic != Magic)
            throw new InvalidDataException($"{path} is not a GGUF file (magic 0x{magic:X8}).");
        Version = ReadU32(ref cursor);
        if (Version is < 2 or > 3)
            throw new NotSupportedException($"GGUF version {Version} is not supported (expected 2 or 3).");

        long tensorCount = (long)ReadU64(ref cursor);
        long kvCount = (long)ReadU64(ref cursor);

        for (long i = 0; i < kvCount; i++)
        {
            string key = ReadString(ref cursor);
            var type = (GgufValueType)ReadU32(ref cursor);
            _metadata[key] = ReadValue(ref cursor, type);
        }

        for (long i = 0; i < tensorCount; i++)
        {
            string name = ReadString(ref cursor);
            uint dims = ReadU32(ref cursor);
            var shape = new long[dims];
            for (uint d = 0; d < dims; d++) shape[d] = (long)ReadU64(ref cursor);
            var ttype = (GgmlType)ReadU32(ref cursor);
            long offset = (long)ReadU64(ref cursor);
            _tensors[name] = new GgufTensorInfo
            {
                Name = name, Shape = shape, Type = ttype, Offset = offset,
            };
        }

        int alignment = _metadata.TryGetValue("general.alignment", out object? a)
            ? Convert.ToInt32(a) : 32;
        DataOffset = cursor + (alignment - cursor % alignment) % alignment;

        ThrowIfTruncated();
    }

    /// <summary>
    /// Rejects a short file up front. Without this a partial download fails as an
    /// access violation deep inside a dequantizer, which reads like a kernel bug
    /// rather than "re-download this file".
    /// </summary>
    private void ThrowIfTruncated()
    {
        long required = DataOffset;
        string? last = null;
        foreach (GgufTensorInfo t in _tensors.Values)
        {
            long end;
            try { end = DataOffset + t.Offset + t.ByteCount; }
            catch (NotSupportedException) { continue; }
            if (end > required) { required = end; last = t.Name; }
        }
        if (_length >= required) return;
        double missingGiB = (required - _length) / (1024.0 * 1024 * 1024);
        throw new IOException(
            $"{Path} is incomplete: {_length} bytes on disk but its {_tensors.Count} tensors " +
            $"need {required} ({missingGiB:F2} GiB missing; '{last}' is the last one). Re-download it.");
    }

    // ------------------------------------------------------------ tensor access

    public GgufTensorInfo GetTensor(string name)
        => _tensors.TryGetValue(name, out GgufTensorInfo? t)
            ? t
            : throw new KeyNotFoundException($"{Path} has no tensor '{name}'.");

    public bool TryGetTensor(string name, out GgufTensorInfo tensor)
        => _tensors.TryGetValue(name, out tensor!);

    /// <summary>Raw bytes of a tensor, as a view straight into the mapping.</summary>
    public ReadOnlySpan<byte> GetTensorBytes(GgufTensorInfo info)
    {
        long bytes = info.ByteCount;
        if (bytes > int.MaxValue)
            throw new NotSupportedException(
                $"'{info.Name}' is {bytes} bytes; use {nameof(GetTensorPointer)} for tensors above 2 GiB.");
        return new ReadOnlySpan<byte>(GetTensorPointer(info), (int)bytes);
    }

    /// <summary>Pointer to a tensor's first byte inside the mapping.</summary>
    public byte* GetTensorPointer(GgufTensorInfo info)
    {
        ObjectDisposedException.ThrowIf(_base is null, this);
        return _base + DataOffset + info.Offset;
    }

    public byte* GetTensorPointer(string name) => GetTensorPointer(GetTensor(name));

    // ---------------------------------------------------------- metadata access

    public string? GetString(string key, string? fallback = null)
        => _metadata.TryGetValue(key, out object? v) && v is string s ? s : fallback;

    public uint GetUInt32(string key, uint fallback = 0)
        => _metadata.TryGetValue(key, out object? v) ? Convert.ToUInt32(v) : fallback;

    public int GetInt32(string key, int fallback = 0)
        => _metadata.TryGetValue(key, out object? v) ? Convert.ToInt32(v) : fallback;

    public float GetFloat32(string key, float fallback = 0f)
        => _metadata.TryGetValue(key, out object? v) ? Convert.ToSingle(v) : fallback;

    public bool GetBool(string key, bool fallback = false)
        => _metadata.TryGetValue(key, out object? v) ? Convert.ToBoolean(v) : fallback;

    public bool HasKey(string key) => _metadata.ContainsKey(key);

    public string[]? GetStringArray(string key)
        => _metadata.TryGetValue(key, out object? v) ? v as string[] : null;

    public float[]? GetFloatArray(string key)
        => _metadata.TryGetValue(key, out object? v) ? v as float[] : null;

    public int[]? GetInt32Array(string key)
    {
        if (!_metadata.TryGetValue(key, out object? v)) return null;
        return v switch
        {
            int[] ia => ia,
            uint[] ua => Array.ConvertAll(ua, x => (int)x),
            long[] la => Array.ConvertAll(la, x => (int)x),
            _ => null,
        };
    }

    // ----------------------------------------------------------------- decoding

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ReadOnlySpan<byte> Window(long offset, int count)
    {
        if (offset < 0 || offset + count > _length)
            throw new InvalidDataException($"{Path}: read of {count} bytes at {offset} runs past the file.");
        return new ReadOnlySpan<byte>(_base + offset, count);
    }

    private uint ReadU32(ref long c) { uint v = BinaryPrimitives.ReadUInt32LittleEndian(Window(c, 4)); c += 4; return v; }
    private ulong ReadU64(ref long c) { ulong v = BinaryPrimitives.ReadUInt64LittleEndian(Window(c, 8)); c += 8; return v; }

    private string ReadString(ref long c)
    {
        long len = (long)ReadU64(ref c);
        if (len is < 0 or > int.MaxValue)
            throw new InvalidDataException($"{Path}: implausible string length {len}.");
        string s = Encoding.UTF8.GetString(Window(c, (int)len));
        c += len;
        return s;
    }

    private object ReadValue(ref long c, GgufValueType type)
    {
        switch (type)
        {
            case GgufValueType.UInt8: { byte v = Window(c, 1)[0]; c += 1; return v; }
            case GgufValueType.Int8: { sbyte v = (sbyte)Window(c, 1)[0]; c += 1; return v; }
            case GgufValueType.UInt16: { ushort v = BinaryPrimitives.ReadUInt16LittleEndian(Window(c, 2)); c += 2; return v; }
            case GgufValueType.Int16: { short v = BinaryPrimitives.ReadInt16LittleEndian(Window(c, 2)); c += 2; return v; }
            case GgufValueType.UInt32: return ReadU32(ref c);
            case GgufValueType.Int32: { int v = BinaryPrimitives.ReadInt32LittleEndian(Window(c, 4)); c += 4; return v; }
            case GgufValueType.Float32: { float v = BinaryPrimitives.ReadSingleLittleEndian(Window(c, 4)); c += 4; return v; }
            case GgufValueType.Bool: { bool v = Window(c, 1)[0] != 0; c += 1; return v; }
            case GgufValueType.String: return ReadString(ref c);
            case GgufValueType.UInt64: return ReadU64(ref c);
            case GgufValueType.Int64: { long v = BinaryPrimitives.ReadInt64LittleEndian(Window(c, 8)); c += 8; return v; }
            case GgufValueType.Float64: { double v = BinaryPrimitives.ReadDoubleLittleEndian(Window(c, 8)); c += 8; return v; }
            case GgufValueType.Array: return ReadArray(ref c);
            default: throw new NotSupportedException($"{Path}: unknown GGUF value type {(uint)type}.");
        }
    }

    private object ReadArray(ref long c)
    {
        var elem = (GgufValueType)ReadU32(ref c);
        long count = (long)ReadU64(ref c);
        if (count is < 0 or > int.MaxValue)
            throw new InvalidDataException($"{Path}: implausible array length {count}.");
        int n = (int)count;

        switch (elem)
        {
            case GgufValueType.String:
            {
                var arr = new string[n];
                for (int i = 0; i < n; i++) arr[i] = ReadString(ref c);
                return arr;
            }
            case GgufValueType.UInt32: return ReadPrimitiveArray<uint>(ref c, n);
            case GgufValueType.Int32: return ReadPrimitiveArray<int>(ref c, n);
            case GgufValueType.Float32: return ReadPrimitiveArray<float>(ref c, n);
            case GgufValueType.UInt64: return ReadPrimitiveArray<ulong>(ref c, n);
            case GgufValueType.Int64: return ReadPrimitiveArray<long>(ref c, n);
            case GgufValueType.Float64: return ReadPrimitiveArray<double>(ref c, n);
            case GgufValueType.UInt16: return ReadPrimitiveArray<ushort>(ref c, n);
            case GgufValueType.Int16: return ReadPrimitiveArray<short>(ref c, n);
            case GgufValueType.UInt8: return ReadPrimitiveArray<byte>(ref c, n);
            case GgufValueType.Int8: return ReadPrimitiveArray<sbyte>(ref c, n);
            case GgufValueType.Bool:
            {
                var arr = new bool[n];
                ReadOnlySpan<byte> src = Window(c, n);
                for (int i = 0; i < n; i++) arr[i] = src[i] != 0;
                c += n;
                return arr;
            }
            case GgufValueType.Array:
                throw new NotSupportedException($"{Path}: nested GGUF arrays are not supported.");
            default:
                throw new NotSupportedException($"{Path}: unknown GGUF array element type {(uint)elem}.");
        }
    }

    private T[] ReadPrimitiveArray<T>(ref long c, int n) where T : unmanaged
    {
        // GGUF is little-endian on the wire and every platform this ships on is
        // little-endian, so the mapped bytes are the array's memory verbatim.
        int bytes = n * sizeof(T);
        var arr = new T[n];
        Window(c, bytes).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
        c += bytes;
        return arr;
    }

    public void Dispose()
    {
        if (_pointerAcquired && _view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _pointerAcquired = false;
        }
        _base = null;
        _view?.Dispose(); _view = null;
        _map?.Dispose(); _map = null;
    }
}
