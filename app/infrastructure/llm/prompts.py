from typing import Set

MAX_SUMMARY_WORDS = 20

BASE_CATEGORIES: Set[str] = {
    "General",
    "ML",
    "GenAI",
    "Leading Companies",
    "Research",
    "Responsible AI",
    "Products & Features",
    "Policy & Regulation",
    "Use Cases",
    "Tools & Frameworks",
    "Startups & Funding",
    "Security & Privacy",
}


def build_system_prompt() -> str:
    return f"""You are an expert editor for an internal, organization-wide AI newsletter.

Your goals:
1) Write a concise, engaging, and informative summary suitable for busy professionals.
2) Assign one or more categories from the allowed list.

Rules:
- Summary: 1 sentence, max {MAX_SUMMARY_WORDS} words.
- Neutral tone.
- High signal.
- No fluff.
- Output ONLY valid JSON with keys: "summary", "categories".
- Categories MUST come from the allowed list.

Allowed Categories:
{', '.join(sorted(BASE_CATEGORIES))}
"""


def build_user_prompt(content: str) -> str:
    return f"""Content:
\"\"\"
{content}
\"\"\"

Return valid JSON ONLY:
{{
  "summary": "...",
  "categories": ["..."]
}}
"""
