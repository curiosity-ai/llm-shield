using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Relative-error assertions.
///
/// xunit's decimal-places overload compares rounded strings, so two float32
/// values a single ULP apart can straddle a rounding boundary and "fail" at four
/// decimal places while agreeing to seven significant figures. Every comparison
/// in this suite is between two float32 pipelines that differ only in evaluation
/// order, so relative error is the right measure.
/// </summary>
internal static class Numeric
{
    public static void Close(double expected, double actual, double relativeTolerance, string? because = null)
    {
        double scale = Math.Max(1.0, Math.Abs(expected));
        double error = Math.Abs(actual - expected) / scale;
        Assert.True(error <= relativeTolerance,
            $"{because ?? "value"}: expected {expected:R}, got {actual:R} " +
            $"(relative error {error:E3}, tolerance {relativeTolerance:E0})");
    }

    public static void Close(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        double relativeTolerance, string? because = null)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worstIndex = -1;
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double scale = Math.Max(1.0, Math.Abs(expected[i]));
            double error = Math.Abs(actual[i] - (double)expected[i]) / scale;
            if (error > worst) { worst = error; worstIndex = i; }
        }
        Assert.True(worst <= relativeTolerance,
            $"{because ?? "vector"}: element {worstIndex} is {actual[Math.Max(worstIndex, 0)]:R} " +
            $"but expected {expected[Math.Max(worstIndex, 0)]:R} " +
            $"(worst relative error {worst:E3}, tolerance {relativeTolerance:E0})");
    }

    /// <summary>Worst relative error across a vector, for reporting rather than asserting.</summary>
    public static double WorstRelativeError(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double worst = 0;
        for (int i = 0; i < expected.Length && i < actual.Length; i++)
        {
            double scale = Math.Max(1.0, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs(actual[i] - (double)expected[i]) / scale);
        }
        return worst;
    }
}
