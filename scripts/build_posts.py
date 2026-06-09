#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent
POSTS_SRC = ROOT / "blog" / "source" / "_posts"
POSTS_OUT = ROOT / "posts"
SITE_INDEX = ROOT / "index.html"
TAGS_INDEX = ROOT / "tags" / "index.html"
CATEGORIES_INDEX = ROOT / "categories" / "index.html"


ARTICLE_STYLE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | Mengqing Cao</title>
  <meta name="description" content="{description}">
  <link rel="shortcut icon" href="/img/favicon.ico">
  <link rel="stylesheet" href="/css/index.css">
  <style>
    :root {{
      --page-bg: #f7f8fb;
      --panel-bg: rgba(255, 255, 255, 0.92);
      --panel-border: rgba(21, 35, 56, 0.08);
      --text-main: #1f2933;
      --text-muted: #5b6472;
      --accent: #0f6cbd;
      --accent-soft: #e8f1fb;
      --shadow: 0 18px 48px rgba(16, 34, 56, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 108, 189, 0.1), transparent 30%),
        linear-gradient(180deg, #f8fbff 0%, var(--page-bg) 55%, #f3f4f7 100%);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.75;
    }}

    .shell {{
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 64px;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 28px;
    }}

    .brand {{
      color: var(--text-main);
      text-decoration: none;
      font-size: 15px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}

    .topnav {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .topnav a,
    .back-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 16px;
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.8);
      color: var(--text-main);
      text-decoration: none;
    }}

    .topnav a:hover,
    .back-link:hover {{
      border-color: rgba(15, 108, 189, 0.3);
      color: var(--accent);
    }}

    .article {{
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .hero {{
      padding: 48px 48px 28px;
      background:
        linear-gradient(135deg, rgba(15, 108, 189, 0.12), rgba(255, 255, 255, 0.35)),
        linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.88));
      border-bottom: 1px solid var(--panel-border);
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 30px;
      padding: 0 12px;
      margin-bottom: 18px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 600;
    }}

    h1 {{
      margin: 0 0 16px;
      font-size: clamp(32px, 4vw, 52px);
      line-height: 1.08;
    }}

    .summary {{
      max-width: 760px;
      margin: 0;
      color: var(--text-muted);
      font-size: 18px;
    }}

    .meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 22px;
    }}

    .meta span {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(21, 35, 56, 0.08);
      color: var(--text-muted);
      font-size: 14px;
    }}

    .content {{
      padding: 40px 48px 48px;
      font-size: 17px;
    }}

    .toc {{
      margin: 0 0 30px;
      padding: 20px 22px;
      border: 1px solid rgba(21, 35, 56, 0.08);
      border-radius: 18px;
      background: rgba(15, 108, 189, 0.04);
    }}

    .toc-title {{
      margin: 0 0 12px;
      color: var(--text-main);
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .toc-links {{
      display: grid;
      gap: 10px;
    }}

    .toc-item,
    .toc-subitem {{
      color: var(--text-main);
      text-decoration: none;
      line-height: 1.5;
    }}

    .toc-item {{
      font-weight: 600;
    }}

    .toc-subitem {{
      padding-left: 14px;
      color: var(--text-muted);
      font-size: 15px;
    }}

    .toc-item:hover,
    .toc-subitem:hover {{
      color: var(--accent);
    }}

    .content h2,
    .content h3 {{
      margin: 40px 0 14px;
      line-height: 1.25;
    }}

    .content h2 {{
      font-size: 28px;
    }}

    .content h3 {{
      font-size: 22px;
    }}

    .content p,
    .content ul,
    .content ol,
    .content blockquote,
    .content pre {{
      margin: 0 0 18px;
    }}

    .content ul,
    .content ol {{
      padding-left: 22px;
    }}

    .content li {{
      margin-bottom: 8px;
    }}

    .content code {{
      padding: 0.12em 0.4em;
      border-radius: 6px;
      background: rgba(15, 108, 189, 0.08);
      color: #0a4e88;
      font-size: 0.95em;
    }}

    .content pre {{
      overflow-x: auto;
      padding: 18px 20px;
      border-radius: 16px;
      background: #152238;
      color: #f3f7fb;
    }}

    .content pre code {{
      padding: 0;
      background: transparent;
      color: inherit;
    }}

    .content blockquote {{
      padding: 12px 18px;
      border-left: 4px solid var(--accent);
      background: rgba(15, 108, 189, 0.06);
      color: var(--text-muted);
      border-radius: 0 12px 12px 0;
    }}

    .content img {{
      display: block;
      width: 100%;
      max-width: 860px;
      margin: 22px auto;
      border-radius: 18px;
      border: 1px solid rgba(21, 35, 56, 0.08);
      background: #fff;
      box-shadow: 0 10px 30px rgba(18, 38, 63, 0.08);
    }}

    .footer-nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-top: 28px;
      padding-top: 28px;
      border-top: 1px solid rgba(21, 35, 56, 0.08);
    }}

    .footer-note {{
      color: var(--text-muted);
      font-size: 14px;
    }}

    a {{
      color: var(--accent);
    }}

    @media (max-width: 720px) {{
      .shell {{
        width: min(100%, calc(100% - 20px));
        padding-top: 20px;
      }}

      .hero,
      .content {{
        padding-left: 20px;
        padding-right: 20px;
      }}

      .topbar,
      .footer-nav {{
        flex-direction: column;
        align-items: flex-start;
      }}

      .summary {{
        font-size: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <a class="brand" href="/">Mengqing Cao</a>
      <nav class="topnav" aria-label="Primary">
        <a href="/">Home</a>
        <a href="/tags/">Tags</a>
        <a href="/categories/">Categories</a>
      </nav>
    </div>

    <article class="article">
      <header class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p class="summary">{summary}</p>
        <div class="meta">
          <span>{date_label}</span>
          {tag_badges}
        </div>
      </header>

      <div class="content">
        {toc}
        {content}
        <div class="footer-nav">
          <a class="back-link" href="/">Back Home</a>
          <div class="footer-note">Generated from markdown source.</div>
        </div>
      </div>
    </article>
  </div>
</body>
</html>
"""


INDEX_STYLE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mengqing Cao</title>
  <meta name="description" content="Mengqing Cao's blog on systems, inference, and large model serving.">
  <link rel="shortcut icon" href="/img/favicon.ico">
  <link rel="stylesheet" href="/css/index.css">
  <style>
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 108, 189, 0.1), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #f7f8fb 58%, #f2f4f7 100%);
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0 72px;
    }}

    .hero {{
      padding: 28px 0 24px;
    }}

    h1 {{
      margin: 0 0 16px;
      font-size: clamp(38px, 5vw, 64px);
      line-height: 1.02;
    }}

    .hero p {{
      margin: 0;
      max-width: 620px;
      color: #5b6472;
      font-size: 18px;
      line-height: 1.7;
    }}

    nav {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 24px;
    }}

    nav a,
    .post-card a.read-more {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 112px;
      min-height: 42px;
      padding: 0 18px;
      border: 1px solid #d6dbe3;
      border-radius: 8px;
      color: #1f2933;
      text-decoration: none;
      background: #fff;
    }}

    nav a:hover,
    .post-card a.read-more:hover {{
      border-color: #8ea0b8;
    }}

    .posts-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 18px;
    }}

    .posts-header p {{
      margin: 0;
      max-width: 560px;
      color: #5b6472;
      font-size: 15px;
      line-height: 1.7;
    }}

    .posts {{
      display: grid;
      gap: 18px;
    }}

    .section-title {{
      margin: 0 0 16px;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6b7280;
    }}

    .post-card {{
      padding: 24px;
      border: 1px solid #d6dbe3;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 12px 30px rgba(18, 38, 63, 0.06);
    }}

    .post-card .meta {{
      margin-bottom: 10px;
      color: #6b7280;
      font-size: 14px;
    }}

    .post-card h2 {{
      margin: 0 0 12px;
      font-size: 28px;
      line-height: 1.2;
    }}

    .post-card h2 a {{
      color: inherit;
      text-decoration: none;
    }}

    .post-card p {{
      margin: 0 0 18px;
      color: #5b6472;
      font-size: 16px;
      line-height: 1.75;
    }}

    .post-card .tags {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}

    .post-card .tags span {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      background: #e8f1fb;
      color: #0f6cbd;
      font-size: 13px;
      font-weight: 600;
    }}

    .post-card .row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .archive-note {{
      margin-top: 22px;
      color: #6b7280;
      font-size: 14px;
    }}

    @media (max-width: 720px) {{
      main {{
        width: min(100%, calc(100% - 20px));
      }}

      .post-card {{
        padding: 18px;
      }}

      .post-card h2 {{
        font-size: 24px;
      }}

      h1 {{
        font-size: 40px;
      }}

      .posts-header {{
        align-items: flex-start;
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Mengqing Cao</h1>
      <p>Notes on systems, inference, and the mechanics underneath large model serving.</p>
      <nav aria-label="Site navigation">
        <a href="/tags/">Tags</a>
        <a href="/categories/">Categories</a>
      </nav>
    </section>

    <section aria-labelledby="all-posts">
      <div class="posts-header">
        <div>
          <h2 class="section-title" id="all-posts">All Posts</h2>
          <p>A running list of essays and technical notes. New writing will land here first.</p>
        </div>
      </div>

      <div class="posts">
        {cards}
      </div>

      <p class="archive-note">Archive building in public.</p>
    </section>
  </main>
</body>
</html>
"""


LIST_STYLE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | Mengqing Cao</title>
  <meta name="description" content="{description}">
  <link rel="shortcut icon" href="/img/favicon.ico">
  <link rel="stylesheet" href="/css/index.css">
  <style>
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 108, 189, 0.1), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #f7f8fb 58%, #f2f4f7 100%);
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0 72px;
    }}

    .hero {{
      padding: 28px 0 24px;
    }}

    h1 {{
      margin: 0 0 12px;
      font-size: clamp(32px, 4vw, 52px);
      line-height: 1.05;
    }}

    .hero p {{
      margin: 0;
      max-width: 720px;
      color: #5b6472;
      font-size: 17px;
      line-height: 1.7;
    }}

    nav {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 24px;
    }}

    nav a,
    .post-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 16px;
      border: 1px solid #d6dbe3;
      border-radius: 8px;
      color: #1f2933;
      text-decoration: none;
      background: #fff;
    }}

    nav a:hover,
    .post-link:hover {{
      border-color: #8ea0b8;
    }}

    .groups {{
      display: grid;
      gap: 18px;
    }}

    .group {{
      padding: 22px 24px;
      border: 1px solid #d6dbe3;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 12px 30px rgba(18, 38, 63, 0.06);
    }}

    .group-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}

    .group-head h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }}

    .group-count {{
      color: #6b7280;
      font-size: 14px;
    }}

    .post-list {{
      display: grid;
      gap: 12px;
    }}

    .post-item {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      padding-top: 12px;
      border-top: 1px solid #e8edf3;
    }}

    .post-item:first-child {{
      padding-top: 0;
      border-top: 0;
    }}

    .post-meta {{
      flex: 1 1 360px;
      min-width: 0;
    }}

    .post-meta a {{
      color: #1f2933;
      text-decoration: none;
      font-size: 18px;
      font-weight: 600;
      line-height: 1.4;
    }}

    .post-meta p {{
      margin: 6px 0 0;
      color: #5b6472;
      font-size: 15px;
      line-height: 1.7;
    }}

    .post-date {{
      color: #6b7280;
      font-size: 14px;
      white-space: nowrap;
    }}

    @media (max-width: 720px) {{
      main {{
        width: min(100%, calc(100% - 20px));
      }}

      .group {{
        padding: 18px;
      }}

      .post-item {{
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{title}</h1>
      <p>{description}</p>
      <nav aria-label="Site navigation">
        <a href="/">Home</a>
        <a href="/tags/">Tags</a>
        <a href="/categories/">Categories</a>
      </nav>
    </section>

    <section class="groups">
      {groups}
    </section>
  </main>
</body>
</html>
"""


@dataclass
class Post:
    source_path: Path
    title: str
    date_raw: str
    slug: str
    summary: str
    tags: List[str]
    categories: List[str]
    body_markdown: str
    published: bool

    @property
    def output_dir(self) -> Path:
        return POSTS_OUT / self.slug

    @property
    def url(self) -> str:
        return f"/posts/{self.slug}/"

    @property
    def date_label(self) -> str:
        return self.date_raw[:10] if self.date_raw else ""

    @property
    def sort_key(self) -> datetime:
        text = self.date_raw.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:19], fmt)
            except ValueError:
                continue
        return datetime.min


def parse_front_matter(text: str) -> Tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    raw_meta = text[4:end].splitlines()
    body = text[end + 5 :]
    meta = {}
    for line in raw_meta:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def split_tags(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[;,]", value)
    return [part.strip() for part in parts if part.strip()]


def split_categories(meta: dict) -> List[str]:
    value = meta.get("categories") or meta.get("category") or ""
    categories = split_tags(value)
    return categories or ["Notes"]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def slugify_heading(text: str) -> str:
    normalized = re.sub(r"[`*_~\[\]()]", "", text).strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", normalized, flags=re.UNICODE)
    normalized = normalized.strip("-")
    return normalized or "section"


def truncate_text(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_summary(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body):
        line = paragraph.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("![") or line.startswith(">") or line.startswith("```"):
            continue
        plain = re.sub(r"`([^`]+)`", r"\1", line)
        plain = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", plain)
        plain = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", plain)
        plain = plain.replace("**", "")
        return truncate_text(plain)
    return ""


def parse_inline(text: str) -> str:
    escaped = html.escape(text, quote=True)
    escaped = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escaped,
    )
    return escaped


def strip_inline_markdown(text: str) -> str:
    plain = re.sub(r"`([^`]+)`", r"\1", text)
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
    plain = re.sub(r"\*(.+?)\*", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", plain)
    return plain.strip()


def count_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_list_line(line: str, indent: int | None = None) -> bool:
    if indent is None:
        indent = count_indent(line)
    return bool(re.match(rf"^\s{{{indent}}}(?:[-*+]|\d+\.)\s+", line))


def parse_list(lines: List[str], start: int, indent: int) -> Tuple[str, int]:
    ordered = bool(re.match(rf"^\s{{{indent}}}\d+\.\s+", lines[start]))
    tag = "ol" if ordered else "ul"
    items = []
    i = start

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if count_indent(line) < indent or not is_list_line(line, indent):
            break

        content = re.sub(rf"^\s{{{indent}}}(?:[-*+]|\d+\.)\s+", "", line).strip()
        item_lines = [content] if content else []
        nested_chunks = []
        i += 1

        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                i += 1
                continue

            next_indent = count_indent(next_line)
            if next_indent < indent:
                break
            if next_indent == indent and is_list_line(next_line, indent):
                break
            if is_list_line(next_line, next_indent) and next_indent > indent:
                nested_html, i = parse_list(lines, i, next_indent)
                nested_chunks.append(nested_html)
                continue

            if next_indent > indent:
                item_lines.append(next_line.strip())
                i += 1
                continue

            break

        parts = []
        if item_lines:
            parts.append(parse_inline(" ".join(item_lines)))
        parts.extend(nested_chunks)
        items.append(f"<li>{''.join(parts)}</li>")

    return f"<{tag}>" + "".join(items) + f"</{tag}>", i


def build_toc(items: List[Tuple[int, str, str]]) -> str:
    if not items:
        return ""

    lines = []
    for level, title, anchor in items:
        class_name = "toc-subitem" if level >= 3 else "toc-item"
        lines.append(
            f'<a class="{class_name}" href="#{html.escape(anchor, quote=True)}">{html.escape(title)}</a>'
        )
    return "\n          ".join(lines)


def render_markdown(body: str) -> Tuple[str, str]:
    lines = body.splitlines()
    chunks = []
    toc_items: List[Tuple[int, str, str]] = []
    heading_counts: dict[str, int] = {}
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            chunks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            raw_title = heading.group(2).strip()
            title = parse_inline(raw_title)
            plain_title = strip_inline_markdown(raw_title)
            anchor_base = slugify_heading(plain_title)
            heading_counts[anchor_base] = heading_counts.get(anchor_base, 0) + 1
            anchor = anchor_base
            if heading_counts[anchor_base] > 1:
                anchor = f"{anchor_base}-{heading_counts[anchor_base]}"
            if level in (2, 3):
                toc_items.append((level, plain_title, anchor))
            chunks.append(f'<h{level} id="{html.escape(anchor, quote=True)}">{title}</h{level}>')
            i += 1
            continue

        image = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image:
            alt = html.escape(image.group(1), quote=True)
            src = html.escape(image.group(2), quote=True)
            chunks.append(f'<img src="{src}" alt="{alt}">')
            i += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            quote_html = "<br>".join(parse_inline(line) for line in quote_lines if line)
            chunks.append(f"<blockquote><p>{quote_html}</p></blockquote>")
            continue

        indent = count_indent(line)
        if is_list_line(line, indent):
            list_html, i = parse_list(lines, i, indent)
            chunks.append(list_html)
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line:
                break
            if next_line.startswith(("```", ">", "#", "![", "- ", "* ", "+ ")):
                break
            if re.match(r"^\d+\.\s+", next_line):
                break
            paragraph_lines.append(next_line)
            i += 1
        chunks.append(f"<p>{parse_inline(' '.join(paragraph_lines))}</p>")

    return "\n        ".join(chunks), build_toc(toc_items)


def build_post(path: Path) -> Post:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    title = meta.get("title") or path.stem
    slug = meta.get("slug") or slugify(path.stem)
    if not slug:
        raise ValueError(f"{path.name} needs a slug because its filename cannot be slugified cleanly.")
    summary = meta.get("summary") or extract_summary(body)
    tags = split_tags(meta.get("tags", ""))
    return Post(
        source_path=path,
        title=title,
        date_raw=meta.get("date", ""),
        slug=slug,
        summary=summary,
        tags=tags,
        categories=split_categories(meta),
        body_markdown=body.strip() + "\n",
        published=bool(meta.get("slug")),
    )


def write_post_html(post: Post) -> None:
    content_html, toc_links = render_markdown(post.body_markdown)
    tag_badges = "\n          ".join(f"<span>{html.escape(tag)}</span>" for tag in post.tags)
    if not tag_badges:
        tag_badges = "<span>Post</span>"
    eyebrow = " / ".join(post.tags[:2]) if post.tags else "Writing"
    description = html.escape(post.summary or post.title, quote=True)
    toc_html = ""
    if toc_links:
        toc_html = f"""<section class="toc" aria-labelledby="toc-title">
          <div class="toc-title" id="toc-title">On This Page</div>
          <nav class="toc-links" aria-label="Table of contents">
          {toc_links}
          </nav>
        </section>"""
    output = ARTICLE_STYLE.format(
        title=html.escape(post.title),
        description=description,
        eyebrow=html.escape(eyebrow),
        summary=html.escape(post.summary or post.title),
        date_label=html.escape(post.date_label),
        tag_badges=tag_badges,
        toc=toc_html,
        content=content_html,
    )
    post.output_dir.mkdir(parents=True, exist_ok=True)
    (post.output_dir / "index.html").write_text(output, encoding="utf-8")


def write_homepage(posts: List[Post]) -> None:
    cards = []
    for post in posts:
        tags = "".join(f"<span>{html.escape(tag)}</span>" for tag in post.tags)
        cards.append(
            f"""<article class="post-card">
        <div class="meta">{html.escape(post.date_label)}</div>
        <h2><a href="{post.url}">{html.escape(post.title)}</a></h2>
        <p>{html.escape(post.summary)}</p>
        <div class="tags">{tags}</div>
        <div class="row">
          <a class="read-more" href="{post.url}">Read Article</a>
        </div>
      </article>"""
        )
    SITE_INDEX.write_text(INDEX_STYLE.format(cards="\n        ".join(cards)), encoding="utf-8")


def build_grouped_posts(posts: List[Post], attr: str) -> dict[str, List[Post]]:
    grouped: dict[str, List[Post]] = {}
    for post in posts:
        for key in getattr(post, attr):
            grouped.setdefault(key, []).append(post)
    return dict(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower())))


def render_group_section(name: str, posts: List[Post]) -> str:
    items = []
    for post in posts:
        items.append(
            f"""<div class="post-item">
        <div class="post-meta">
          <a href="{post.url}">{html.escape(post.title)}</a>
          <p>{html.escape(post.summary)}</p>
        </div>
        <div class="post-date">{html.escape(post.date_label)}</div>
      </div>"""
        )
    return f"""<article class="group">
        <div class="group-head">
          <h2>{html.escape(name)}</h2>
          <div class="group-count">{len(posts)} post(s)</div>
        </div>
        <div class="post-list">
          {' '.join(items)}
        </div>
      </article>"""


def write_group_page(path: Path, title: str, description: str, groups: dict[str, List[Post]]) -> None:
    rendered_groups = "\n      ".join(render_group_section(name, posts) for name, posts in groups.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        LIST_STYLE.format(
            title=html.escape(title),
            description=html.escape(description),
            groups=rendered_groups,
        ),
        encoding="utf-8",
    )


def write_taxonomy_pages(posts: List[Post]) -> None:
    tag_groups = build_grouped_posts(posts, "tags")
    category_groups = build_grouped_posts(posts, "categories")
    write_group_page(
        TAGS_INDEX,
        "Tags",
        "Browse posts grouped by topic tags.",
        tag_groups,
    )
    write_group_page(
        CATEGORIES_INDEX,
        "Categories",
        "Browse posts grouped by categories.",
        category_groups,
    )


def copy_images_if_needed(posts: List[Post]) -> None:
    for post in posts:
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", post.body_markdown):
            src = match.group(1).strip()
            if not src.startswith("file://"):
                continue
            local_path = Path(src.replace("file://", ""))
            if not local_path.exists():
                continue
            rel = local_path.relative_to(ROOT)
            target = ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(local_path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown posts into site HTML pages.")
    parser.add_argument("markdown", nargs="*", help="Optional markdown files to build. Default: all posts in blog/source/_posts.")
    args = parser.parse_args()

    explicit = bool(args.markdown)
    if explicit:
        paths = [Path(path).resolve() for path in args.markdown]
        explicit_posts = [build_post(path) for path in paths]
        unpublished = [post.source_path.name for post in explicit_posts if not post.published]
        if unpublished:
            names = ", ".join(unpublished)
            raise SystemExit(f"These files need a front matter 'slug' before publishing: {names}")

    posts = [build_post(path) for path in sorted(POSTS_SRC.glob("*.md"))]
    if explicit:
        posts = [post for post in posts if post.published]
    else:
        posts = [post for post in posts if post.published]

    posts.sort(key=lambda post: post.sort_key, reverse=True)

    copy_images_if_needed(posts)
    for post in posts:
        write_post_html(post)
        print(f"Built {post.url} from {post.source_path.name}")
    write_homepage(posts)
    write_taxonomy_pages(posts)
    print(f"Updated {SITE_INDEX}")


if __name__ == "__main__":
    main()
