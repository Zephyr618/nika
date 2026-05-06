#!/usr/bin/env python3
"""Render a NIKA agent NDJSON log into a self-contained chat-style HTML.

Usage:
    python scripts/viz_agent_log.py path/to/conversation_xxx_agent.log
    python scripts/viz_agent_log.py path/to/conversation_xxx_agent.log -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ts


def fmt_full_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts


def extract_first_content(messages_str: str) -> str:
    """Pull the first content="..." out of a langchain message repr.

    langchain serializes messages as e.g. content="..." additional_kwargs={} ...
    The content can contain newlines and escaped quotes. We grab from the first
    content=" up to the matching unescaped " by walking the string.
    """
    idx = messages_str.find('content="')
    if idx < 0:
        return messages_str
    i = idx + len('content="')
    out = []
    while i < len(messages_str):
        c = messages_str[i]
        if c == "\\" and i + 1 < len(messages_str):
            nxt = messages_str[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append(nxt)
            i += 2
            continue
        if c == '"':
            break
        out.append(c)
        i += 1
    return "".join(out)


_TOOL_NAME_RE = re.compile(r"name='([^']+)'")


def extract_tool_name_from_output(output_str: str) -> str | None:
    m = _TOOL_NAME_RE.search(output_str)
    return m.group(1) if m else None


def extract_text_from_tool_output(output_str: str) -> str:
    """Best-effort: pull the inner 'text' field out of a langchain ToolMessage repr.

    Format looks like: content=[{'type': 'text', 'text': '<JSON or text>', 'id': ...}]
    We try to locate "'text': '" and read until the matching unescaped "'".
    Falls back to the raw string if parsing fails.
    """
    key = "'text': '"
    idx = output_str.find(key)
    if idx < 0:
        return output_str
    i = idx + len(key)
    out = []
    while i < len(output_str):
        c = output_str[i]
        if c == "\\" and i + 1 < len(output_str):
            nxt = output_str[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "'":
                out.append("'")
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append(nxt)
            i += 2
            continue
        if c == "'":
            break
        out.append(c)
        i += 1
    return "".join(out)


def pretty_json(s: str) -> str:
    """Try to format s as JSON; return original on failure."""
    s = s.strip()
    if not s:
        return s
    try:
        return json.dumps(json.loads(s), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return s


def render_html(events: list[dict], log_path: Path) -> str:
    n_llm = sum(1 for e in events if e.get("event") == "llm_end")
    n_tools = sum(1 for e in events if e.get("event") == "tool_start")

    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Agent Conversation — {html.escape(log_path.name)}</title>
<style>
  :root {{
    --bg: #ededed; --card: #ffffff; --agent: #95ec69; --tool-call: #fdf6b2;
    --tool-result: #ffffff; --system: #fff8dc; --muted: #999;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
                 "Microsoft YaHei", "Segoe UI", sans-serif;
    max-width: 980px; margin: 0 auto; padding: 20px;
    color: #222;
  }}
  .header {{
    background: var(--card); border-radius: 8px; padding: 16px 20px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .header h2 {{ margin: 0 0 6px 0; font-size: 18px; }}
  .header .meta {{ font-size: 13px; color: #666; }}
  .header .stats {{ font-size: 12px; color: #888; margin-top: 4px; }}

  .turn {{ margin: 12px 0; display: flex; }}
  .turn.right {{ justify-content: flex-end; }}
  .turn.left {{ justify-content: flex-start; }}
  .turn.center {{ justify-content: center; }}

  .bubble {{
    max-width: 75%; padding: 10px 14px; border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08); position: relative;
  }}
  .bubble .ts {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .bubble .body {{
    white-space: pre-wrap; word-break: break-word; line-height: 1.55;
    font-size: 14px;
  }}

  .agent {{ background: var(--agent); }}
  .agent .ts {{ color: #4d6b3a; }}
  .tool-call {{ background: var(--tool-call); }}
  .tool-call .tool-name {{
    font-weight: 600; color: #b45309; margin-bottom: 4px; font-size: 13px;
  }}
  .tool-result {{ background: var(--tool-result); border: 1px solid #e5e7eb; }}
  .tool-result .tool-name {{
    font-weight: 600; color: #047857; margin-bottom: 4px; font-size: 13px;
  }}

  .system {{
    background: var(--system); border-left: 4px solid #d4a017;
    padding: 14px 18px; border-radius: 6px; margin: 20px 0;
    font-size: 13px; color: #555;
  }}
  .system .label {{
    font-weight: 600; color: #92400e; margin-bottom: 6px; font-size: 13px;
  }}
  .system .body {{ white-space: pre-wrap; line-height: 1.5; }}

  .badges {{ font-size: 11px; color: #666; margin-top: 8px; }}
  .badge {{
    display: inline-block; background: #f3f4f6; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1px 8px; margin-right: 4px;
  }}

  details {{ margin-top: 8px; }}
  summary {{
    cursor: pointer; color: #2563eb; font-size: 12px; user-select: none;
    outline: none;
  }}
  summary:hover {{ text-decoration: underline; }}
  pre.payload {{
    background: #f9fafb; padding: 10px; border-radius: 4px;
    max-height: 420px; overflow: auto; font-size: 12px; line-height: 1.45;
    margin: 6px 0 0 0; border: 1px solid #e5e7eb;
    white-space: pre; word-break: normal;
  }}

  .divider {{
    text-align: center; color: #aaa; font-size: 11px; margin: 16px 0;
  }}
</style>
</head>
<body>
  <div class="header">
    <h2>Agent Conversation</h2>
    <div class="meta">{html.escape(str(log_path))}</div>
    <div class="stats">{len(events)} events · {n_llm} LLM turns · {n_tools} tool calls</div>
  </div>
"""
    )

    seen_first_llm_start = False

    for ev in events:
        kind = ev.get("event")
        ts_short = fmt_ts(ev.get("timestamp", ""))
        ts_full = fmt_full_ts(ev.get("timestamp", ""))

        if kind == "llm_start":
            if seen_first_llm_start:
                continue
            seen_first_llm_start = True
            content = extract_first_content(ev.get("messages", ""))
            parts.append(
                f"""  <div class="system">
    <div class="label">📋 任务初始化 · {html.escape(ts_full)}</div>
    <div class="body">{html.escape(content)}</div>
  </div>
"""
            )

        elif kind == "llm_end":
            text = ev.get("text", "") or "(empty)"
            usage = ev.get("usage_metadata") or {}
            in_t = usage.get("input_tokens")
            out_t = usage.get("output_tokens")
            cache = (usage.get("input_token_details") or {}).get("cache_read")
            badges = []
            if in_t is not None:
                badges.append(f"in: {in_t}")
            if cache:
                badges.append(f"cache: {cache}")
            if out_t is not None:
                badges.append(f"out: {out_t}")
            badge_html = (
                "<div class=\"badges\">"
                + "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
                + "</div>"
                if badges
                else ""
            )
            parts.append(
                f"""  <div class="turn right">
    <div class="bubble agent">
      <div class="ts">🤖 agent · {html.escape(ts_short)}</div>
      <div class="body">{html.escape(text)}</div>
      {badge_html}
    </div>
  </div>
"""
            )

        elif kind == "tool_start":
            tool = ev.get("tool") or {}
            name = tool.get("name", "(unknown)")
            desc = tool.get("description", "").strip().splitlines()[0] if tool.get("description") else ""
            inp = ev.get("input", "")
            inp_pretty = pretty_json(inp) if inp.strip().startswith(("{", "[")) else inp
            parts.append(
                f"""  <div class="turn right">
    <div class="bubble tool-call">
      <div class="ts">🔧 agent calls tool · {html.escape(ts_short)}</div>
      <div class="tool-name">→ {html.escape(name)}</div>
      <div class="body">{html.escape(desc)}</div>
      <details>
        <summary>input</summary>
        <pre class="payload">{html.escape(inp_pretty)}</pre>
      </details>
    </div>
  </div>
"""
            )

        elif kind == "tool_end":
            output = ev.get("output", "")
            name = extract_tool_name_from_output(output) or "tool"
            text_payload = extract_text_from_tool_output(output)
            text_pretty = pretty_json(text_payload) if text_payload.strip().startswith(("{", "[")) else text_payload
            preview = text_pretty if len(text_pretty) <= 200 else text_pretty[:200] + " …"
            parts.append(
                f"""  <div class="turn left">
    <div class="bubble tool-result">
      <div class="ts">✅ tool result · {html.escape(ts_short)}</div>
      <div class="tool-name">← {html.escape(name)}</div>
      <div class="body">{html.escape(preview)}</div>
      <details>
        <summary>full output</summary>
        <pre class="payload">{html.escape(text_pretty)}</pre>
      </details>
    </div>
  </div>
"""
            )

        else:
            parts.append(
                f'  <div class="divider">unknown event: {html.escape(str(kind))} @ {html.escape(ts_short)}</div>\n'
            )

    parts.append("</body>\n</html>\n")
    return "".join(parts)


def parse_log(log_path: Path) -> list[dict]:
    """Parse a NDJSON log file, returning the list of events. Empty if not NDJSON."""
    events: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                # First-line failure → not NDJSON; bail out silently.
                if lineno == 1:
                    return []
                print(f"warn: {log_path}:{lineno} is not valid JSON: {e}", file=sys.stderr)
    return events


def convert_one(log_path: Path, out_path: Path | None = None) -> tuple[Path, int] | None:
    """Convert a single log → HTML. Returns (out_path, n_events) or None on skip."""
    events = parse_log(log_path)
    if not events:
        return None
    out = out_path or log_path.with_suffix(".html")
    out.write_text(render_html(events, log_path), encoding="utf-8")
    return out, len(events)


def render_index(out_dir: Path, items: list[tuple[Path, Path, int]]) -> Path:
    """Generate an index.html linking to each converted HTML.

    items: list of (log_path, html_path, n_events)
    """
    rows: list[str] = []
    grouped: dict[str, list[tuple[Path, Path, int]]] = {}
    for log_p, html_p, n in items:
        try:
            rel_parts = log_p.relative_to(out_dir).parts
            group = rel_parts[0] if len(rel_parts) > 1 else "(root)"
        except ValueError:
            group = str(log_p.parent)
        grouped.setdefault(group, []).append((log_p, html_p, n))

    for group, entries in sorted(grouped.items()):
        rows.append(f'  <h3>{html.escape(group)}</h3>\n  <ul>\n')
        for log_p, html_p, n in sorted(entries, key=lambda x: x[0]):
            try:
                href = html_p.relative_to(out_dir).as_posix()
            except ValueError:
                href = html_p.as_posix()
            label = log_p.name
            try:
                rel_for_label = log_p.relative_to(out_dir).as_posix()
                label = rel_for_label
            except ValueError:
                pass
            rows.append(
                f'    <li><a href="{html.escape(href)}">{html.escape(label)}</a> '
                f'<span style="color:#888;font-size:12px">({n} events)</span></li>\n'
            )
        rows.append("  </ul>\n")

    index_path = out_dir / "agent_logs_index.html"
    index_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Agent log index — {html.escape(str(out_dir))}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif;
          max-width: 960px; margin: 0 auto; padding: 24px; color: #222; }}
  h2 {{ margin-top: 0; }}
  h3 {{ margin: 24px 0 8px 0; color: #444; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  ul {{ list-style: none; padding-left: 0; }}
  li {{ padding: 4px 0; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
  <h2>Agent log index</h2>
  <div style="color:#666">{html.escape(str(out_dir))} · {len(items)} logs</div>
{''.join(rows)}</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "path", type=Path,
        help="Path to a single NDJSON log, or a directory to scan recursively",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML path (only valid for single-file mode; default: alongside input)",
    )
    p.add_argument(
        "--glob", default="conversation_*.log",
        help="Filename pattern to match in directory mode (default: conversation_*.log)",
    )
    p.add_argument(
        "--index", action="store_true",
        help="In directory mode, also generate an index.html linking to each converted HTML",
    )
    args = p.parse_args()

    if args.path.is_file():
        result = convert_one(args.path, args.output)
        if result is None:
            print(f"error: {args.path} is not a parseable NDJSON log", file=sys.stderr)
            return 1
        out, n = result
        print(f"wrote {out}  ({n} events)")
        return 0

    if args.path.is_dir():
        if args.output is not None:
            print("error: -o/--output is not valid in directory mode", file=sys.stderr)
            return 2
        logs = sorted(args.path.rglob(args.glob))
        if not logs:
            print(f"error: no files matching '{args.glob}' under {args.path}", file=sys.stderr)
            return 1
        converted: list[tuple[Path, Path, int]] = []
        skipped: list[Path] = []
        for log_p in logs:
            res = convert_one(log_p)
            if res is None:
                skipped.append(log_p)
                continue
            out, n = res
            converted.append((log_p, out, n))
            print(f"  ✓ {log_p}  →  {out.name}  ({n} events)")
        for log_p in skipped:
            print(f"  ⊘ skipped (not NDJSON): {log_p}")
        print(f"\nconverted {len(converted)} log(s); skipped {len(skipped)}")
        if args.index and converted:
            idx = render_index(args.path, converted)
            print(f"index: {idx}")
        return 0

    print(f"error: {args.path} is neither a file nor a directory", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
