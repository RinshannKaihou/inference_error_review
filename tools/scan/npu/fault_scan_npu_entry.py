#!/usr/bin/env python3
"""两条故障扫描共用的 NPU 采集前置入口；不写编目、不处理存量 Git。"""
import argparse
from pathlib import Path
import subprocess

PROFILE = Path("/root/.hermes/profiles/third")
SHARED = PROFILE / "scripts" / "fault_scan_sources"
CACHE = PROFILE / "cache" / "fault_scan_sources"


def preflight(repo):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, timeout=90).strip()

    if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise RuntimeError(f"BLOCKED: {repo}: not on main")
    if git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") != "origin/main":
        raise RuntimeError(f"BLOCKED: {repo}: upstream is not origin/main")
    if git("status", "--porcelain", "-uall"):
        raise RuntimeError(f"BLOCKED: {repo}: dirty worktree; existing files untouched")
    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=repo, check=True, timeout=90)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise RuntimeError(f"BLOCKED: {repo}: HEAD differs from origin/main")


def collector_command(scope):
    return ["/usr/bin/python3", str(SHARED / "collect.py"), "--scope", scope,
            "--registry", str(SHARED / "sources.json"), "--output-dir", str(CACHE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=["inference", "training"])
    args = parser.parse_args()
    preflight(Path("/workspace") / f"{args.scope}_error_review")
    subprocess.run(collector_command(args.scope), check=True)


if __name__ == "__main__":
    main()
