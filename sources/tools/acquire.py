#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
USER_AGENT = "TG-DCM-PINN-literature-review/0.1 (research source retrieval)"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, destination: pathlib.Path) -> dict:
    if destination.exists() and destination.stat().st_size > 0:
        return {"status": "cached", "path": str(destination.relative_to(ROOT)), "sha256": sha256(destination), "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=35) as response, temporary.open("wb") as stream:
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                stream.write(data)
            content_type = response.headers.get("Content-Type", "")
        if temporary.stat().st_size < 100:
            raise ValueError("source too small, likely failed retrieval")
        temporary.replace(destination)
        return {"status": "downloaded", "source": url, "path": str(destination.relative_to(ROOT)), "content_type": content_type, "sha256": sha256(destination), "bytes": destination.stat().st_size}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        return {"status": "unavailable", "source": url, "error": str(exc)}


def run_git(repo: dict) -> dict:
    url = "https://github.com/" + repo["repo"] + ".git"
    target = CACHE / "repos" / repo["repo"].replace("/", "--")
    revision = repo["revision"]
    try:
        if not (target / ".git").is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)], check=True, timeout=240, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", "origin", revision], check=True, timeout=240, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", "--force", revision], check=True, timeout=240, capture_output=True, text=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"revision mismatch: expected {revision}, got {actual}")
        return {"status": "checked_out", "repo": repo["repo"], "revision": actual, "path": str(target.relative_to(ROOT))}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "repo": repo["repo"], "revision": revision, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve reviewed paper sources and pinned Git repositories; never bypass paywalls.")
    parser.add_argument("--papers", action="store_true")
    parser.add_argument("--repos", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    papers_flag = args.all or args.papers or not (args.papers or args.repos)
    repos_flag = args.all or args.repos
    paper_index = json.loads((ROOT / "papers" / "index.json").read_text(encoding="utf-8"))
    repo_index = json.loads((ROOT / "repositories" / "index.json").read_text(encoding="utf-8"))
    actions = []
    if papers_flag:
        for paper in paper_index["papers"]:
            ident = paper.get("arxiv")
            candidates = []
            if ident:
                candidates = [
                    ("html", f"https://arxiv.org/html/{ident}", ".html"),
                    ("pdf", f"https://arxiv.org/pdf/{ident}", ".pdf"),
                    ("tex", f"https://arxiv.org/src/{ident}", ".src"),
                ]
            elif "doi.org/" not in paper["url"]:
                candidates = [("landing", paper["url"], ".html")]
            for kind, url, ext in candidates:
                if args.dry_run:
                    actions.append({"paper": paper["id"], "format": kind, "status": "planned", "url": url})
                else:
                    actions.append({"paper": paper["id"], "format": kind, **fetch(url, CACHE / "papers" / (paper["id"] + ext))})
    if repos_flag:
        for repo in repo_index["repositories"]:
            actions.append({"repo": repo["repo"], "revision": repo["revision"], "status": "planned" if args.dry_run else None} if args.dry_run else run_git(repo))
    if not args.dry_run:
        CACHE.mkdir(exist_ok=True)
        path = CACHE / "acquisition.json"
        path.write_text(json.dumps({"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "records": actions}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"actions": len(actions), "ok": sum(x["status"] in ("downloaded", "cached", "checked_out", "planned") for x in actions), "unavailable": sum(x["status"] == "unavailable" for x in actions), "records": actions}, indent=2))
    return 1 if any(x["status"] == "unavailable" for x in actions) else 0


if __name__ == "__main__":
    sys.exit(main())
