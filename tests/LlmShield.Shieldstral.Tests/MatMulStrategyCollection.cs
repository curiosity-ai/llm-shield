using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Serialises the test classes that set <c>QuantMatMul.Strategy</c>.
///
/// The strategy is process-wide, and xunit runs test classes in parallel by
/// default — so one class pinning it to Float while another pins it to Integer
/// makes both measure whatever the scheduler happened to leave behind. Restoring
/// it afterwards is not enough; the classes must not overlap at all.
///
/// Any new test that touches the strategy belongs in this collection.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class MatMulStrategyCollection
{
    public const string Name = "matmul-strategy";
}
