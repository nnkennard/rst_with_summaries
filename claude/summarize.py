"""
Summarize text with the Claude API, keeping the summary within a token budget.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"

Usage:
    python summarize.py input.txt --max-tokens 200
    cat input.txt | python summarize.py --max-tokens 150
"""

import argparse
import sys

import anthropic

MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = (
    "You are a precise summarizer. You write faithful, self-contained summaries "
    "that keep the most important facts, conclusions, and figures, and never add "
    "information that isn't in the source."
)


def build_prompt(text: str, token_limit: int) -> str:
    # Aim below the hard limit so the model finishes its last sentence
    # instead of being cut off by max_tokens.
    target = int(token_limit * 0.8)
    approx_words = int(target * 0.75)  # ~0.75 English words per token
    return (
        f"Summarize the text inside <text> tags.\n\n"
        f"Length requirement: the summary must fit within {token_limit} tokens. "
        f"Aim for about {target} tokens (roughly {approx_words} words) and end on a "
        f"complete sentence.\n\n"
        f"Output only the summary, with no preamble.\n\n"
        f"<text>\n{text}\n</text>"
    )


def summarize(text: str, token_limit: int, max_attempts: int = 3) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    limit = token_limit

    for attempt in range(1, max_attempts + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=token_limit,  # hard cap on output length
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(text, limit)}],
        )
        summary = "".join(b.text for b in response.content if b.type == "text").strip()

        # stop_reason == "max_tokens" means the summary was cut off mid-thought.
        if response.stop_reason != "max_tokens":
            print(
                f"[attempt {attempt}] {response.usage.output_tokens} output tokens "
                f"(limit {token_limit})",
                file=sys.stderr,
            )
            return summary

        # Ask for something shorter next time.
        limit = int(limit * 0.7)
        print(f"[attempt {attempt}] truncated; retrying with target {limit}", file=sys.stderr)

    return summary  # best effort: last (truncated) result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize text with Claude.")
    parser.add_argument("file", nargs="?", help="Text file to summarize (default: stdin)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Token limit for the summary")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        sys.exit("No input text provided.")

    print(summarize(text, args.max_tokens))


if __name__ == "__main__":
    main()