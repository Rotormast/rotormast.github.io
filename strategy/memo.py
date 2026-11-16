"""Advisor memo. This is the page to copy to Merrill Lynch. It is not an order."""

from __future__ import annotations

import html


def plain_text(state: dict) -> str:
    action = state.get("action", {})
    lines = [
        "Research Deck index desk",
        f"As of {state.get('asOf')} (last daily close)",
        f"Instruction: {action.get('label')}",
        "",
        "Send these weights:",
    ]
    for row in state.get("instruction", []):
        if row.get("weight", 0) < 0.005 and row.get("group") != "cash":
            continue
        lines.append(f"  {row['name']} ({row['ticker']})  {row['weight'] * 100:.1f}%")
    changes = state.get("changes") or []
    if changes:
        lines.append("")
        lines.append("Changes from the book carried into this close:")
        for row in changes:
            lines.append(f"  {row['name']} ({row['ticker']})  {row['from'] * 100:.1f}% -> {row['to'] * 100:.1f}%")
    reasons = action.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Why this is a real-time adjustment:")
        lines.extend(f"  {reason}" for reason in reasons)
    objective = state.get("objective", {})
    best = objective.get("best") or {}
    lines.extend(
        [
            "",
            "Measured track record after the factor choice, assuming every past instruction was followed:",
            f"  Desk CAGR {objective.get('strategyCagr', 0) * 100:.1f}%",
            f"  Best index {best.get('name', 'n/a')} ({best.get('ticker', '')}) CAGR {best.get('cagr', 0) * 100:.1f}%",
            f"  Gap {objective.get('gap', 0) * 100:.1f} percentage points",
            f"  Beat every index: {'yes' if objective.get('beatEveryIndex') else 'no'}",
            "",
            state.get("disclaimer", ""),
            "",
            "Merrill Lynch places any trades. This note is not an order.",
        ]
    )
    return "\n".join(lines) + "\n"


def html_memo(state: dict, narrative: str | None = None) -> str:
    text = plain_text(state)
    note = ""
    if narrative:
        note = (
            "<h2>Note from the local language model</h2>"
            "<p>This note did not choose the weights.</p>"
            f"<p>{html.escape(narrative)}</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Advisor memo · Research Deck</title>
<style>
  body {{ font-family: Georgia, "Iowan Old Style", serif; max-width: 680px; margin: 48px auto; padding: 0 20px; color: #0B1F3A; background: #fcfcfb; line-height: 1.5; }}
  pre {{ font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; white-space: pre-wrap; font-size: 14px; }}
  a {{ color: #2A78D6; }}
</style>
</head>
<body>
<p><a href="/">Back to the desk</a></p>
<h1>Advisor memo</h1>
<p>Give this to Merrill Lynch. They place the trades.</p>
<pre>{html.escape(text)}</pre>
{note}
</body>
</html>
"""
