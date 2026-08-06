using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Pins the prompt framing against mistralai/Shieldstral-1.0-3B's published
/// <c>chat_template.jinja</c>. The model was trained on these exact bytes — an
/// extra space or a missing marker is not a formatting nit, it moves the verdict.
/// </summary>
public class ChatTemplateTests
{
    private const string System = ShieldstralModerator.SystemPrompt;

    [Fact]
    public void SystemAndUserRenderWithNoSeparator()
    {
        string rendered = ChatTemplate.Render(
        [
            ChatMessage.System(System),
            ChatMessage.User("hello"),
        ]);
        Assert.Equal($"[SYSTEM_PROMPT]{System}[/SYSTEM_PROMPT][INST]hello[/INST]", rendered);
    }

    [Fact]
    public void NoTrailingGenerationMarker()
    {
        // Mistral's template ends at [/INST]; the model answers straight after it.
        string rendered = ChatTemplate.Render([ChatMessage.User("x")]);
        Assert.EndsWith("[/INST]", rendered);
        Assert.Equal("[INST]x[/INST]", rendered);
    }

    [Fact]
    public void ImagesPrecedeTheTextWithinATurn()
    {
        string rendered = ChatTemplate.Render(
        [
            ChatMessage.System(System),
            ChatMessage.User("describe") with { ImagePaths = ["a.jpg"] },
        ]);
        Assert.StartsWith($"[SYSTEM_PROMPT]{System}[/SYSTEM_PROMPT][INST][IMG]describe", rendered);
    }

    [Fact]
    public void OneMarkerPerImage()
    {
        string rendered = ChatTemplate.Render(
            [ChatMessage.User("compare these") with { ImagePaths = ["a.jpg", "b.jpg", "c.jpg"] }]);
        Assert.Equal("[INST][IMG][IMG][IMG]compare these[/INST]", rendered);
    }

    [Fact]
    public void AssistantTurnsAreFollowedByEos()
    {
        string rendered = ChatTemplate.Render(
        [
            ChatMessage.User("q"),
            ChatMessage.Assistant("a"),
            ChatMessage.User("q2"),
        ]);
        Assert.Equal("[INST]q[/INST]a</s>[INST]q2[/INST]", rendered);
    }

    [Fact]
    public void UnsupportedRoleIsRejected()
    {
        Assert.Throws<ArgumentException>(() => ChatTemplate.Render([new ChatMessage("tool", "x")]));
        Assert.Throws<ArgumentException>(() => ChatTemplate.Render([]));
        Assert.Throws<ArgumentException>(() => ChatTemplate.Render([ChatMessage.Assistant("")]));
    }

    [Fact]
    public void ModeratorUsesTheModelCardsFraming()
    {
        var request = new ModerationRequest("be strict", "is it unsafe?", "some text");
        string user = ShieldstralModerator.FormatUserMessage(request);
        Assert.Equal("<Instruct>: be strict\n\n<Query>: is it unsafe?\n\n<Document>: some text", user);
    }

    [Fact]
    public void SystemPromptMatchesTheModelCard()
    {
        Assert.Equal(
            "Judge whether the Document meets the requirements based on the Query and the Instruction " +
            "provided. Note that the answer can only be \"yes\" or \"no\".",
            ShieldstralModerator.SystemPrompt);
    }
}
