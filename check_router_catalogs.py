"""
Check which library models each router (OpenRouter, Requesty, Opper) carries.

    python check_router_catalogs.py          # report differences, exit 1 if any
    python check_router_catalogs.py --write  # rewrite ROUTER_AVAILABILITY in routers.py

A model counts as available on a router if any catalog ID (any host or region)
matches it. Matching is exact on a normalized name, plus the explicit ALIASES
below for routers that use dated snapshot names. Exits with status 2 if a
catalog cannot be fetched or looks truncated, so a failed fetch never wipes
availability.
"""
import json
import pathlib
import re
import sys
import urllib.request

from models import MODEL_LIBRARY
from routers import ROUTERS

CATALOGS = {
    "OpenRouter": "https://openrouter.ai/api/v1/models",
    "Requesty":   "https://router.requesty.ai/v1/models",
    "Opper":      "https://opper.ai/models",           # HTML; the JSON API needs a key
}
MIN_IDS = 200   # a smaller catalog means the fetch or the page format broke

# Router IDs whose name differs from the library name (dated snapshots, hosted variants)
ALIASES = {
    "Mistral Large 3":    ["mistral-large-2512", "mistral-large-latest"],
    "Mistral Medium 3.5": ["mistral-medium-3.5-128b"],
    "Mistral Small 4":    ["mistral-small-2603"],
    "Codestral":          ["codestral-2508", "codestral-latest"],
    "Ministral 3 3B":     ["ministral-3b-2512", "ministral-3b"],
    "Voxtral Small":      ["voxtral-small-24b-2507", "voxtral-small-2507", "voxtral-small-latest"],
    "Leanstral 1.5":      ["labs-leanstral-1-5"],
    "Gemini 3 Flash":     ["gemini-3-flash-preview"],
    "Gemini 3.1 Pro":     ["gemini-3.1-pro-preview"],
    "Grok 4.20":          ["grok-4.20-reasoning", "grok-4.20-non-reasoning", "grok-4.20-0309-non-reasoning"],
    "DeepSeek V4 Pro":    ["deepseek-v4-pro-0813"],
    "DeepSeek V4 Flash":  ["deepseek-v4-flash-0731"],
    "Qwen3.8 Max":        ["qwen3.8-max-0902"],
    "Qwen3.5 Plus":       ["qwen3.5-plus-20260420"],
    "Qwen3.5 Flash":      ["qwen3.5-flash-02-23"],
    "Qwen3 235B A22B":    ["qwen3-235b-a22b-instruct-2507"],
    "Llama 4 Maverick":   ["llama-4-maverick-17b-128e-instruct-fp8"],
    "Llama 4 Scout":      ["llama-4-scout-17b-16e-instruct"],
}

ROUTERS_PY = pathlib.Path(__file__).parent / "routers.py"
BEGIN, END = "# BEGIN GENERATED", "# END GENERATED"


def norm(s):
    s = s.lower()
    s = re.sub(r"(\d)p(\d)", r"\1\2", s)   # Fireworks writes 3.7 as 3p7
    return re.sub(r"[^a-z0-9]", "", s)


def model_key(catalog_id):
    """'vertex/claude-opus-4-8@eu' -> 'claudeopus48'"""
    return norm(catalog_id.split("/")[-1].split("@")[0].split(":")[0])


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tco-simulator/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def catalog_ids(router, body):
    if router == "Opper":
        ids = set(re.findall(r'href="/([a-z0-9-]+/[A-Za-z0-9._-]+)"', body))
        ids |= set(re.findall(r'\\"(?:model_)?id\\":\\"([a-z0-9-]+/[A-Za-z0-9._-]+)\\"', body))
        return ids
    return {m["id"] for m in json.loads(body)["data"]}


def availability(catalogs):
    keys = {r: {model_key(i) for i in ids} for r, ids in catalogs.items()}
    out = {}
    for name in MODEL_LIBRARY:
        wanted = {norm(name)} | {norm(a) for a in ALIASES.get(name, [])}
        out[name] = tuple(r for r in ROUTERS if wanted & keys[r])
    return out


def render(avail):
    lines = [BEGIN + " by check_router_catalogs.py --write; do not edit by hand",
             "ROUTER_AVAILABILITY = {"]
    for name, routers in avail.items():
        lines.append(f"    {name!r}: {routers!r},")
    lines += ["}", END]
    return "\n".join(lines)


def main():
    catalogs = {}
    for router in ROUTERS:
        try:
            ids = catalog_ids(router, fetch(CATALOGS[router]))
        except Exception as e:   # network, HTTP or JSON errors
            print(f"ERROR: could not fetch {router} catalog: {e}", file=sys.stderr)
            sys.exit(2)
        if len(ids) < MIN_IDS:
            print(f"ERROR: {router} catalog has only {len(ids)} IDs", file=sys.stderr)
            sys.exit(2)
        catalogs[router] = ids
        print(f"{router}: {len(ids)} catalog IDs")

    avail = availability(catalogs)
    src = ROUTERS_PY.read_text()
    new_block = render(avail)
    start, end = src.index(BEGIN), src.index(END) + len(END)

    if "--write" in sys.argv:
        ROUTERS_PY.write_text(src[:start] + new_block + src[end:])
        print(f"Wrote availability for {len(avail)} models to routers.py")
        return

    from routers import ROUTER_AVAILABILITY as current
    diffs = [(n, current.get(n), a) for n, a in avail.items() if current.get(n) != a]
    for n, old, new in diffs:
        print(f"  {n}: {old} -> {new}")
    print(f"{len(diffs)} model(s) differ" if diffs else "Availability up to date")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
