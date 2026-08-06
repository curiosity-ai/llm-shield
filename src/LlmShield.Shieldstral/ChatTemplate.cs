using System.Text;

namespace LlmShield.Shieldstral;

/// <summary>One turn of a Shieldstral conversation.</summary>
public sealed record ChatMessage(string Role, string Content)
{
    /// <summary>Images to place ahead of the text as <c>[IMG]</c> markers.</summary>
    public IReadOnlyList<string> ImagePaths { get; init; } = [];

    public static ChatMessage System(string content) => new("system", content);
    public static ChatMessage User(string content) => new("user", content);
    public static ChatMessage Assistant(string content) => new("assistant", content);
}

/// <summary>
/// Renders Shieldstral's prompt format.
///
/// The published <c>chat_template.jinja</c> reduces, for the roles this model
/// accepts, to a short concatenation: BOS (added by the tokenizer), then a system
/// message wrapped in <c>[SYSTEM_PROMPT]…[/SYSTEM_PROMPT]</c>, then each user turn
/// in <c>[INST]…[/INST]</c>, with assistant turns followed by EOS. There is no
/// separator between blocks and no trailing generation marker — the model answers
/// straight after <c>[/INST]</c>.
///
/// Implementing it directly rather than running a Jinja interpreter keeps the
/// rendering exact and auditable; <c>ChatTemplateTests</c> pins it against the
/// published template's own output.
/// </summary>
public static class ChatTemplate
{
    public const string SystemOpen = "[SYSTEM_PROMPT]";
    public const string SystemClose = "[/SYSTEM_PROMPT]";
    public const string InstructionOpen = "[INST]";
    public const string InstructionClose = "[/INST]";
    /// <summary>Placeholder the vision path later expands into a grid of patch tokens.</summary>
    public const string ImageMarker = "[IMG]";
    public const string EndOfSequence = "</s>";

    public static string Render(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder(256);
        string? previousRole = null;

        foreach (ChatMessage message in messages)
        {
            switch (message.Role)
            {
                case "system":
                    sb.Append(SystemOpen).Append(message.Content).Append(SystemClose);
                    break;

                case "user":
                    sb.Append(InstructionOpen);
                    // The template puts images ahead of the text within a turn.
                    for (int i = 0; i < message.ImagePaths.Count; i++) sb.Append(ImageMarker);
                    sb.Append(message.Content).Append(InstructionClose);
                    break;

                case "assistant":
                    if (string.IsNullOrEmpty(message.Content))
                        throw new ArgumentException("An assistant message must have content.", nameof(messages));
                    sb.Append(message.Content).Append(EndOfSequence);
                    break;

                default:
                    throw new ArgumentException(
                        $"Shieldstral accepts system, user and assistant roles; got '{message.Role}'.",
                        nameof(messages));
            }
            previousRole = message.Role;
        }

        if (previousRole is null)
            throw new ArgumentException("No messages to render.", nameof(messages));
        return sb.ToString();
    }
}
