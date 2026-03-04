"""
ChatML Formatting with Loss Masking.

Formats multi-turn conversations using the ChatML template and creates
label masks so that loss is only computed on assistant responses.

ChatML format:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    Hi there!<|im_end|>
"""

from dataclasses import dataclass

from yagpt.tokenizer import Tokenizer

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Label value that cross_entropy ignores
IGNORE_INDEX = -100


@dataclass
class FormattedSample:
    """A formatted chat sample with loss masking."""

    input_ids: list[int]
    labels: list[int]  # Same length as input_ids, IGNORE_INDEX for masked positions


def setup_chat_tokenizer(tokenizer: Tokenizer) -> tuple[int, int]:
    """
    Add ChatML special tokens to a tokenizer.

    Returns:
        Tuple of (im_start_id, im_end_id)
    """
    # Use IDs beyond the base vocab
    base_vocab = tokenizer.vocab_size
    im_start_id = base_vocab
    im_end_id = base_vocab + 1

    tokenizer.add_special_tokens({
        IM_START: im_start_id,
        IM_END: im_end_id,
    })

    return im_start_id, im_end_id


def format_chat(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
    im_start_id: int | None = None,
    im_end_id: int | None = None,
) -> FormattedSample | None:
    """
    Format a conversation into ChatML with loss masking.

    User/system turns are masked (label=IGNORE_INDEX).
    Assistant turns have actual token IDs as labels.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
        tokenizer: Tokenizer with ChatML special tokens added
        max_seq_len: Maximum sequence length (truncate if exceeded)
        im_start_id: Token ID for <|im_start|>, auto-detected if None
        im_end_id: Token ID for <|im_end|>, auto-detected if None

    Returns:
        FormattedSample or None if the conversation is empty/too short
    """
    if not messages:
        return None

    # Auto-detect special token IDs if not provided
    if im_start_id is None:
        im_start_id = tokenizer.encode_special(IM_START)[0]
    if im_end_id is None:
        im_end_id = tokenizer.encode_special(IM_END)[0]

    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Encode: <|im_start|>role\ncontent<|im_end|>\n
        role_tokens = tokenizer.encode(role + "\n")
        content_tokens = tokenizer.encode(content)

        # Build turn tokens
        turn_tokens = [im_start_id] + role_tokens + content_tokens + [im_end_id]

        if role == "assistant":
            # For assistant: mask the header, train on content + im_end
            header_len = 1 + len(role_tokens)  # im_start + "assistant\n"
            turn_labels = (
                [IGNORE_INDEX] * header_len
                + content_tokens
                + [im_end_id]
            )
        else:
            # For system/user: mask everything
            turn_labels = [IGNORE_INDEX] * len(turn_tokens)

        input_ids.extend(turn_tokens)
        labels.extend(turn_labels)

    # Truncate to max_seq_len
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    if len(input_ids) < 2:
        return None

    return FormattedSample(input_ids=input_ids, labels=labels)
