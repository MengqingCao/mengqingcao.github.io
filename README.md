# mengqingcao.github.io

This repository hosts a lightweight static blog site for GitHub Pages.

## Write a Post

Markdown source files live in:

`/Users/mengqing/code/mengqingcao.github.io/blog/source/_posts/`

Each post should include front matter like this:

```md
---
title: 文章标题
date: 2026-06-09 14:34:51
slug: your-post-slug
summary: 一句话摘要
tags: tag1; tag2
---
```

Notes:

- `slug` is required for publishing
- `summary` is used on the homepage list
- `tags` should be separated by `;` or `,`
- Image paths should use site-relative paths such as `/img/example.png`

## Build Posts

The post builder script is:

`/Users/mengqing/code/mengqingcao.github.io/scripts/build_posts.py`

Run it from the repo root:

```bash
cd /Users/mengqing/code/mengqingcao.github.io
./scripts/build_posts.py
```

This will:

1. Read markdown posts from `blog/source/_posts/`
2. Generate article pages in `posts/<slug>/index.html`
3. Rebuild the homepage article list in `index.html`

To build only one post:

```bash
./scripts/build_posts.py blog/source/_posts/你的文章.md
```

## Publish

After checking the generated files, commit and push:

```bash
git add README.md scripts/build_posts.py index.html posts/ blog/source/_posts/
git commit -m "Update blog posts"
git push origin master
```

Only markdown files with a `slug` in front matter are treated as publishable posts by default.
