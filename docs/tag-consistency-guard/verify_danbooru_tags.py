# -*- coding: utf-8 -*-
"""Verify tag_data.py tags against Danbooru and fetch wiki tag groups.

Usage (from ComfyUI root):
    .venv/bin/python custom_nodes/comfyui-alchemine-pack/docs/tag-consistency-guard/verify_danbooru_tags.py

1. Collects every tag in nodes/lib/tag_data.py and checks it exists on
   Danbooru via /tags.json (batched). Reports missing / deprecated /
   low post_count tags to verify_report.txt.
2. Fetches Danbooru wiki tag group pages and saves the extracted tag
   lists to danbooru_tag_groups.json (for refreshing tag_data.py).
"""
import asyncio
import json
import re
import sys
from os import environ
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
PACK = HERE.parent.parent
sys.path.insert(0, str(PACK / "nodes" / "lib"))
import tag_data  # noqa: E402

TAG_GROUPS = [
    "tag_group:attire",
    "tag_group:posture",
    "tag_group:face_tags",
    "tag_group:hair",
    "tag_group:hair_styles",
    "tag_group:hair_color",
    "tag_group:eyes_tags",
    "tag_group:locations",
    "tag_group:backgrounds",
]

BATCH = 100


def get_proxy_config():
    username = environ.get("WEBSHARE_PROXY_USERNAME")
    password = environ.get("WEBSHARE_PROXY_PASSWORD")
    if not username or not password:
        return None
    return {
        "server": environ.get("WEBSHARE_PROXY_SERVER", "http://p.webshare.io:80"),
        "username": username,
        "password": password,
    }


def to_danbooru(tag):
    return tag.strip().lower().replace(" ", "_")


def collect_tags():
    """All tags in tag_data.py -> {danbooru_name: source}."""
    tags = {}

    def add(lst, source):
        for t in lst:
            tags.setdefault(to_danbooru(t), source)

    for sub, lst in tag_data.CLOTHES.items():
        add(lst, "clothes/%s" % sub)
    for cat, lst in tag_data.CATEGORIES.items():
        add(lst, cat)
    add(tag_data.ACCESSORIES, "accessories")
    return tags


async def fetch_json(ctx, url):
    resp = await ctx.get(url)
    if not resp.ok:
        raise Exception("Request to %s failed with status %s" % (url, resp.status))
    return await resp.json()


async def verify(ctx, tags):
    names = sorted(tags)
    found = {}
    for i in range(0, len(names), BATCH):
        batch = names[i : i + BATCH]
        url = (
            "https://danbooru.donmai.us/tags.json"
            "?limit=1000&only=name,post_count,is_deprecated"
            "&search[name_comma_separated]=" + ",".join(batch)
        )
        for row in await fetch_json(ctx, url):
            found[row["name"]] = row
        print("verified %d/%d" % (min(i + BATCH, len(names)), len(names)))
        await asyncio.sleep(0.5)
    return found


async def fetch_tag_groups(ctx):
    groups = {}
    for group in TAG_GROUPS:
        url = "https://danbooru.donmai.us/wiki_pages/%s.json" % group
        try:
            data = await fetch_json(ctx, url)
        except Exception as e:
            print("skip %s: %s" % (group, e))
            continue
        # extract [[tag]] / [[tag|label]] wiki links
        links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", data.get("body", ""))
        tags = sorted(
            {
                to_danbooru(t)
                for t in links
                if not t.lower().startswith(("tag group", "tag_group", "howto", "help:", "list of"))
            }
        )
        groups[group] = tags
        print("%s: %d tags" % (group, len(tags)))
        await asyncio.sleep(0.5)
    return groups


async def main():
    tags = collect_tags()
    print("collecting: %d unique tags in tag_data.py" % len(tags))

    async with async_playwright() as p:
        ctx = await p.request.new_context(proxy=get_proxy_config())

        found = await verify(ctx, tags)
        groups = await fetch_tag_groups(ctx)

    missing = sorted(t for t in tags if t not in found)
    deprecated = sorted(t for t in tags if found.get(t, {}).get("is_deprecated"))
    empty = sorted(
        t for t in tags
        if t in found and not found[t]["is_deprecated"] and found[t]["post_count"] == 0
    )

    lines = []
    lines.append("total tags in tag_data.py: %d" % len(tags))
    lines.append("found on danbooru: %d" % len(found))
    lines.append("")
    lines.append("== MISSING (%d): not a danbooru tag ==" % len(missing))
    lines.extend("  %s  [%s]" % (t, tags[t]) for t in missing)
    lines.append("")
    lines.append("== DEPRECATED (%d) ==" % len(deprecated))
    lines.extend("  %s  [%s]" % (t, tags[t]) for t in deprecated)
    lines.append("")
    lines.append("== ZERO POSTS (%d) ==" % len(empty))
    lines.extend("  %s  [%s]" % (t, tags[t]) for t in empty)
    report = "\n".join(lines)
    (HERE / "verify_report.txt").write_text(report)
    (HERE / "danbooru_tag_groups.json").write_text(
        json.dumps(groups, indent=2, ensure_ascii=False)
    )
    print()
    print(report[:2000])
    print("...")
    print("report -> verify_report.txt, tag groups -> danbooru_tag_groups.json")


if __name__ == "__main__":
    asyncio.run(main())
