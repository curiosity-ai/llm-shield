using System.Runtime.InteropServices;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Runs one <see cref="QuantMatMul.ForwardAsync(WeightMatrix, ReadOnlyMemory{float}, int, Memory{float}, ParallelOptions)"/>
/// against a managed weight buffer.
///
/// A <c>fixed</c> region cannot span an <c>await</c>, so the buffer is pinned with a
/// <see cref="GCHandle"/> for the duration instead, and the pointer lives in a closure
/// built by a synchronous helper rather than as a local of the async method.
/// </summary>
internal static class PinnedMatMul
{
    internal static Task ForwardAsync(
        GgmlType type, Array weights, int rows, int cols,
        float[] x, int tokens, float[] destination, ParallelOptions? options = null)
    {
        var handle = GCHandle.Alloc(weights, GCHandleType.Pinned);
        return Run(Bind(handle, type, rows, cols, x, tokens, destination, options ?? new ParallelOptions()), handle);

        static async Task Run(Func<ValueTask> work, GCHandle handle)
        {
            try { await work().ConfigureAwait(false); }
            finally { handle.Free(); }
        }
    }

    /// <summary><see cref="QuantMatMul.ForwardRowsAsync"/> against a managed weight buffer.</summary>
    internal static Task ForwardRowsAsync(
        GgmlType type, Array weights, int rows, int cols, int[] rowIds,
        float[] x, int tokens, float[] destination, ParallelOptions? options = null)
    {
        var handle = GCHandle.Alloc(weights, GCHandleType.Pinned);
        return Run(BindRows(handle, type, rows, cols, rowIds, x, tokens, destination, options ?? new ParallelOptions()), handle);

        static async Task Run(Func<ValueTask> work, GCHandle handle)
        {
            try { await work().ConfigureAwait(false); }
            finally { handle.Free(); }
        }
    }

    private static unsafe Func<ValueTask> BindRows(
        GCHandle handle, GgmlType type, int rows, int cols, int[] rowIds,
        float[] x, int tokens, float[] destination, ParallelOptions options)
    {
        var matrix = new WeightMatrix(type, (byte*)handle.AddrOfPinnedObject(), rows, cols);
        return () => QuantMatMul.ForwardRowsAsync(matrix, rowIds, x, tokens, destination, options);
    }

    private static unsafe Func<ValueTask> Bind(
        GCHandle handle, GgmlType type, int rows, int cols,
        float[] x, int tokens, float[] destination, ParallelOptions options)
    {
        var matrix = new WeightMatrix(type, (byte*)handle.AddrOfPinnedObject(), rows, cols);
        return () => QuantMatMul.ForwardAsync(matrix, x, tokens, destination, options);
    }
}
