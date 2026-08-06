// Vectorised primitives for the transformer forward pass.
//
// Everything here is `Vector<float>`-width agnostic: the JIT picks 128/256/512-bit
// registers from the host ISA, so one body covers SSE, AVX2, AVX-512 and NEON.
// TensorPrimitives is used wherever it already has a tuned kernel; the remaining
// ones are fused shapes it has no single-call equivalent for.
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace LlmShield.Shieldstral.Numerics;

public static class Kernels
{
    /// <summary>
    /// RMS norm: <c>y = x / sqrt(mean(x^2) + eps) * weight</c>.
    /// </summary>
    /// <remarks>
    /// The sum of squares uses two independent vector accumulators and widens to
    /// double at the end. Two accumulators halve the FMA dependency chain and, more
    /// to the point, halve the length of each running float32 sum — this reduction
    /// runs over 3072 elements and its result is multiplied back into every one of
    /// them, so drift here is not local.
    /// </remarks>
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps, Span<float> y)
    {
        int n = x.Length;
        double sum = 0;
        int i = 0;
        int width = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated && n >= 2 * width)
        {
            Vector<float> even = Vector<float>.Zero, odd = Vector<float>.Zero;
            for (; i + 2 * width <= n; i += 2 * width)
            {
                var a = new Vector<float>(x.Slice(i, width));
                var b = new Vector<float>(x.Slice(i + width, width));
                even += a * a;
                odd += b * b;
            }
            sum = (double)Vector.Sum(even) + Vector.Sum(odd);
        }
        for (; i < n; i++) sum += (double)x[i] * x[i];

        float scale = 1f / MathF.Sqrt((float)(sum / n) + eps);

        // y = x * scale * weight, fused so the intermediate never round-trips.
        i = 0;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var vs = new Vector<float>(scale);
            for (; i + width <= n; i += width)
            {
                var v = new Vector<float>(x.Slice(i, width));
                var w = new Vector<float>(weight.Slice(i, width));
                (v * vs * w).CopyTo(y.Slice(i, width));
            }
        }
        for (; i < n; i++) y[i] = x[i] * scale * weight[i];
    }

    /// <summary>In-place <c>a += b</c>.</summary>
    public static void Add(Span<float> a, ReadOnlySpan<float> b) => TensorPrimitives.Add(a, b, a);

    /// <summary>In-place <c>a *= scale</c>.</summary>
    public static void Scale(Span<float> a, float scale) => TensorPrimitives.Multiply(a, scale, a);

    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => TensorPrimitives.Dot(a, b);

    /// <summary>
    /// SwiGLU activation: <c>out = silu(gate) * up</c> with
    /// <c>silu(v) = v * sigmoid(v)</c>. Fused because the FFN's intermediate is
    /// 9216 wide per token — three separate passes over it costs more in memory
    /// traffic than the arithmetic itself.
    /// </summary>
    /// <remarks>
    /// <paramref name="up"/> is used as scratch and left holding <c>gate * up</c>;
    /// <paramref name="destination"/> may alias <paramref name="gate"/>. Folding the
    /// product into <paramref name="up"/> first is what makes that aliasing safe,
    /// since the sigmoid pass then destroys a value nothing still needs.
    /// </remarks>
    public static void SwiGlu(ReadOnlySpan<float> gate, Span<float> up, Span<float> destination)
    {
        // TensorPrimitives.Sigmoid is the vectorised exp; three vector passes still
        // beat a scalar loop by a wide margin because the transcendental dominates.
        TensorPrimitives.Multiply(gate, up, up);
        TensorPrimitives.Sigmoid(gate, destination);
        TensorPrimitives.Multiply(destination, up, destination);
    }

    /// <summary>
    /// In-place softmax, max-shifted for stability.
    /// </summary>
    /// <remarks>
    /// Deliberately not <c>TensorPrimitives.SoftMax</c>: that one evaluates
    /// <c>exp(x) / Σexp(x)</c> without subtracting the maximum first. Attention
    /// scores in this model reach the 90s in the deepest layers, <c>exp</c>
    /// overflows to infinity there, and the division then yields NaN for exactly
    /// the positions that mattered most. Subtracting the max costs one extra pass
    /// and makes the range irrelevant.
    /// </remarks>
    public static void Softmax(Span<float> x)
    {
        if (x.Length == 0) return;

        float max = TensorPrimitives.Max(x);

        // An entirely masked-out row divides 0 by 0 and poisons every later layer
        // with NaN. A causal mask never produces one — each query always attends to
        // itself — but a caller-supplied mask might, and a uniform distribution is
        // the answer that keeps the failure local.
        if (!float.IsFinite(max))
        {
            x.Fill(1f / x.Length);
            return;
        }

        TensorPrimitives.Subtract(x, max, x);
        TensorPrimitives.Exp(x, x);
        TensorPrimitives.Divide(x, TensorPrimitives.Sum(x), x);
    }

    /// <summary>
    /// <c>destination += scale * source</c>, the accumulation step of attention's
    /// weighted value sum.
    /// </summary>
    public static void AddScaled(Span<float> destination, ReadOnlySpan<float> source, float scale)
    {
        int n = destination.Length;
        int i = 0;
        int width = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var vs = new Vector<float>(scale);
            for (; i + width <= n; i += width)
            {
                var d = new Vector<float>(destination.Slice(i, width));
                var s = new Vector<float>(source.Slice(i, width));
                (d + s * vs).CopyTo(destination.Slice(i, width));
            }
        }
        for (; i < n; i++) destination[i] += source[i] * scale;
    }

    /// <summary>Index of the largest element; ties resolve to the lowest index.</summary>
    public static int ArgMax(ReadOnlySpan<float> x) => TensorPrimitives.IndexOfMax(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}
