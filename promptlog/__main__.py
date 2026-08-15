"""Command-line interface for the prompt journal."""

from __future__ import annotations

import argparse

from promptlog.journal import add_prompt, export_markdown, list_prompts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m promptlog",
        description="Record every prompt you send, in JSONL + Markdown.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Record a new prompt")
    add.add_argument("prompt", help="The prompt text")
    add.add_argument("--tag", action="append", default=[], help="Repeatable tag, e.g. --tag finance")
    add.add_argument("--note", default=None, help="Optional note")

    ls = sub.add_parser("list", help="List recent prompts")
    ls.add_argument("--limit", type=int, default=20, help="Number of recent entries to show")

    sub.add_parser("export", help="Rebuild prompts.md from prompts.jsonl")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "add":
        entry = add_prompt(args.prompt, tags=args.tag, note=args.note)
        print(f"Recorded {entry['id']}  ({entry['timestamp']})")
    elif args.command == "list":
        entries = list_prompts(args.limit)
        if not entries:
            print("No prompts recorded yet.")
            return
        for entry in entries:
            print(f"[{entry['id']}] {entry['timestamp']}")
            print(f"  {entry['prompt']}")
            if entry.get("tags"):
                print(f"  tags: {', '.join(entry['tags'])}")
            print()
    elif args.command == "export":
        count = export_markdown()
        print(f"Rebuilt prompts.md from {count} entr{'y' if count == 1 else 'ies'}.")


if __name__ == "__main__":
    main()

