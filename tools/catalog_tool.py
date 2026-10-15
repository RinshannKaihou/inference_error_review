#!/usr/bin/env python3
"""Structured fault catalog: the only way the daily scan writes entries (identical file in both catalog repos).

  python3 tools/catalog_tool.py new [--file entry.json] [--legacy-id 1.295] [--allow-duplicate-source] [--dry-run]
  python3 tools/catalog_tool.py update INF-0044 [--add-evidence URL ...] [--note TEXT] [--set path=value ...]
                                [--relation var:4.28 ...] [--drop-relation var:INF-0328 ...]
  python3 tools/catalog_tool.py find <ID | legacy id | slug | URL | owner/repo#N | text>
  python3 tools/catalog_tool.py build           # regenerate views/ from catalog/entries
  python3 tools/catalog_tool.py validate        # all entries + registry + views up to date; exit 1 on error
  python3 tools/catalog_tool.py stats
  python3 tools/catalog_tool.py cross-sync [--dry-run]   # add reverse links for the other catalog's cross links
  python3 tools/catalog_tool.py pattern-sync [--dry-run] # derive catalog/patterns/ + entry `pattern` from the var forest
  python3 tools/catalog_tool.py pattern-set PAT-INF-0042 --title ... --shared-mechanism ...

`new` reads ONE JSON object (stdin or --file): the structured fields plus the scan's narrative under
"original". On a problem it prints {"ok": false, "errors": [...]} and exits 2 — fix the JSON and resubmit.
On success it assigns the next ID, writes catalog/entries/<ID>.yaml with status pending_audit and
prints {"ok": true, "id": ...}. Layout and fields: catalog/SCHEMA.md; vocabulary: catalog/vocab.yaml;
per-repo settings: catalog/config.yaml.

Relation targets (`relations` in the `new` JSON, `update --relation`) may be current ids or old Markdown
numbers; old numbers are converted, and the result echoes every target with its title — check that it
is the entry you meant before going on. `cross` targets in the other catalog (config `sibling`) are
looked up there; `cross-sync` (run by the daily audit) adds the reverse link on this side.
"""

import os
import sys

# Under the hermes cron gateway `python3` is hermes' bundled interpreter, which lacks PyYAML.
# Re-exec with the system interpreter instead of failing, so `python3 tools/...` works from any caller.
try:
    import yaml  # noqa: F401
except ImportError:
    for _alt in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if os.path.exists(_alt) and os.path.realpath(_alt) != os.path.realpath(sys.executable):
            os.execv(_alt, [_alt] + sys.argv)
    sys.exit("needs PyYAML: python3 -m pip install pyyaml")

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "catalog")
ENTRIES = os.path.join(CAT, "entries")
REGISTRY = os.path.join(CAT, "id_registry.tsv")
VIEWS = os.path.join(ROOT, "views")
TBD, NA = "待定", "N/A"

ID_RE = re.compile(r"^(INF|TRN)-\d{4,}$")
DATE_RE = re.compile(r"^20\d\d-\d\d-\d\d$")
URL_RE = re.compile(r"https?://[^\s）)，。；、`>\]\"']+")
ARXIV_ID_RE = re.compile(r"arXiv[:：]\s*(\d{4}\.\d{4,5})", re.I)
TOP_KEYS = ("id", "legacy_ids", "slug", "title", "kind", "status", "cause", "effect",
            "mitigation", "labels", "remarks", "evidence", "relations", "original")
# `pattern` 是可选字段，不入 TOP_KEYS：check_entry 要求 TOP_KEYS 全部存在，
# 而只有属于某个模式的条目才有它。
PATTERNS = os.path.join(ROOT, "catalog", "patterns")
# 模式编号带编目命名空间：RAG 把两个编目合并索引，纯 PAT-0035 在那里不唯一
# （两库各有一个，--pattern 会同时命中两边）。
PAT_RE = re.compile(r"^PAT-(INF|TRN)-\d{4}$")
LABEL_ENUMS = {
    "family": ("infra_off", "model_on", "ambiguous"),
    "trigger": ("config", "yes", "partial", "other_stack", "hw_specific", "no", "not_target"),
    "confidence": ("verified", "documented", "speculative"),
    "coverage": ("NEW", "var", "rationale", "existing"),
}
STAGES = ("pretrain", "sft", "rl", "shared")
STATUSES = ("active", "pending_audit", "merged", "retired")
EVIDENCE_KINDS = ("github_issue", "github_pr", "gitcode_issue", "gitcode_pr", "arxiv", "paper", "other")
LIMITS = {"title": 40, "component": 25, "behavior": 120, "detail": 150}
NARRATIVE_ORDER = ("mechanism", "source", "behavior", "trigger_surface", "signature", "repro", "discovery", "notes")


# ---------------------------------------------------------------- io

class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))


def dump_yaml(obj):
    return yaml.dump(obj, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=10_000)


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config():
    return load_yaml(os.path.join(CAT, "config.yaml"))


def vocab():
    return load_yaml(os.path.join(CAT, "vocab.yaml"))


def load_entries(entries_dir=ENTRIES):
    out = {}
    for name in sorted(os.listdir(entries_dir)):
        if name.endswith(".yaml"):
            e = load_yaml(os.path.join(entries_dir, name))
            out[e["id"]] = e
    return out


def load_registry(path=REGISTRY):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            next(fh)
            rows = [tuple(line.rstrip("\n").split("\t")) for line in fh if line.strip()]
    return rows


_SIBLING = []


def sibling():
    """The other catalog repo (config `sibling`, relative to this repo) as {ns, root, entries, registry};
    None when it is not configured or not checked out next to this one."""
    if not _SIBLING:
        rel = (config() or {}).get("sibling")
        root = os.path.normpath(os.path.join(ROOT, rel)) if rel else None
        cat = os.path.join(root, "catalog") if root else None
        if cat and os.path.isdir(os.path.join(cat, "entries")) and os.path.exists(os.path.join(cat, "config.yaml")):
            _SIBLING.append({"ns": load_yaml(os.path.join(cat, "config.yaml"))["namespace"], "root": root,
                             "entries": load_entries(os.path.join(cat, "entries")),
                             "registry": load_registry(os.path.join(cat, "id_registry.tsv"))})
        else:
            _SIBLING.append(None)
    return _SIBLING[0]


def today():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def out(obj, code=0):
    print(json.dumps(obj, ensure_ascii=False, indent=1))
    sys.exit(code)


# ---------------------------------------------------------------- evidence

def norm_url(u):
    return re.sub(r"[#?].*$", "", u).rstrip("/").lower()


REPO_ALIASES = {"tensorrt-llm": ("tensorrt-llm", "trt-llm", "trtllm"), "text-generation-inference": ("tgi", "text-generation-inference"),
                "megatron-lm": ("megatron-lm", "megatron"), "deepspeed": ("deepspeed",), "llama.cpp": ("llama.cpp",)}


def gh_refs(urls):
    """{(repo_short, number)} for GitHub issue/PR URLs."""
    out = set()
    for u in urls:
        m = re.search(r"github\.com/[^/]+/([^/]+)/(?:issues|pull)/(\d+)", u or "")
        if m:
            out.add((m.group(1).lower(), m.group(2)))
    return out


def mentions(entry, repo_short, number):
    """True when the entry's narrative cites repo_short #number without a link (e.g. 'vLLM #60342/#60421')."""
    names = REPO_ALIASES.get(repo_short, (repo_short,))
    text = "\n".join(str(v) for v in (entry.get("original") or {}).values()).lower()
    for line in text.splitlines():
        if re.search(rf"#{number}\b", line) and any(n in line for n in names):
            return True
    return False


def evidence_of(source_text):
    urls = []
    for u in URL_RE.findall(source_text or ""):
        u = u.rstrip(".,;:")
        if u not in urls:
            urls.append(u)
    for aid in ARXIV_ID_RE.findall(source_text or ""):
        if not any(aid in u for u in urls):
            urls.append(f"https://arxiv.org/abs/{aid}")
    return [{"url": u, "kind": url_kind(u)} for u in urls]


def url_kind(u):
    if re.search(r"github\.com/[^/]+/[^/]+/pull/\d+", u):
        return "github_pr"
    if re.search(r"github\.com/[^/]+/[^/]+/issues/\d+", u):
        return "github_issue"
    # GitCode (atomgit.com is the same site); merge requests are also linked as /pull/ or /pulls/
    if re.search(r"(?:gitcode|atomgit)\.com/[^/]+/[^/]+/(?:merge_requests|pulls?)/\d+", u, re.I):
        return "gitcode_pr"
    if re.search(r"(?:gitcode|atomgit)\.com/[^/]+/[^/]+/issues/\w+", u, re.I):
        return "gitcode_issue"
    if "arxiv.org" in u:
        return "arxiv"
    if re.search(r"aclanthology|doi\.org|usenix|acm\.org|ieee|openreview", u, re.I):
        return "paper"
    return "other"


# ---------------------------------------------------------------- validation

def check_entry(e, voc, strict_lengths=False):
    """-> (errors, warnings). strict_lengths turns length overruns into errors (new entries)."""
    errors, warnings = [], []
    ph = set(voc["placeholders"])
    is_ph = lambda v: isinstance(v, str) and v in ph

    def enum(value, allowed, field):
        if not is_ph(value) and not (isinstance(value, str) and value in allowed):
            errors.append(f"{field}={value!r} 不在词表内")

    def length(value, field, limit):
        if isinstance(value, str) and not is_ph(value) and len(value) > limit:
            (errors if strict_lengths else warnings).append(f"{field} 长 {len(value)} 字，上限 {limit} 字")

    missing = [k for k in TOP_KEYS if k not in e]
    if missing:
        return [f"缺少字段 {missing}"], warnings
    if not ID_RE.match(str(e["id"])):
        errors.append("id 格式应为 INF-0001 / TRN-0001")
    enum(e["kind"], voc["kind"], "kind")
    enum(e["status"], STATUSES, "status")
    if e["status"] == "merged" and not e.get("merged_into"):
        errors.append("status=merged 但缺 merged_into")
    if not isinstance(e.get("slug"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", e["slug"]):
        errors.append("slug 应为 snake_case")
    length(e["title"], "title", LIMITS["title"])

    cause, effect = e["cause"] if isinstance(e["cause"], dict) else {}, e["effect"] if isinstance(e["effect"], dict) else {}
    objs = cause.get("objects")
    if not is_ph(objs):
        if not isinstance(objs, list) or not objs or not all(isinstance(o, dict) for o in objs):
            errors.append("cause.objects 应为非空列表")
        else:
            for o in objs:
                enum(o.get("subsystem"), voc["subsystem"], "cause.objects.subsystem")
                if not is_ph(o.get("stack")) and o.get("stack") not in voc["stack"]:
                    warnings.append(f"stack={o.get('stack')!r} 不在词表内（半受控）")
                length(o.get("component"), "cause.objects.component", LIMITS["component"])
    if not isinstance(cause.get("behavior"), str):
        errors.append("缺少 cause.behavior")
    length(cause.get("behavior"), "cause.behavior", LIMITS["behavior"])
    eobjs = effect.get("objects")
    if not is_ph(eobjs):
        if not isinstance(eobjs, list) or not 1 <= len(eobjs) <= 3:
            errors.append("effect.objects 应为 1–3 项列表")
        else:
            for o in eobjs:
                enum(o, voc["effect_object"], "effect.objects")
    if not isinstance(effect.get("detail"), str):
        errors.append("缺少 effect.detail")
    length(effect.get("detail"), "effect.detail", LIMITS["detail"])
    if "visibility" not in effect:
        errors.append("缺少 effect.visibility")
    else:
        enum(effect["visibility"], voc["visibility"], "effect.visibility")
    if not isinstance(e["mitigation"], dict) or not {"detect", "fix"} <= set(e["mitigation"]):
        errors.append("mitigation 需含 detect 与 fix")

    lab = e["labels"] if isinstance(e["labels"], dict) else {}
    for k, allowed in LABEL_ENUMS.items():
        enum(lab.get(k), allowed, f"labels.{k}")
    stage = lab.get("stage")
    if not is_ph(stage) and (not isinstance(stage, list) or not set(stage) <= set(STAGES)):
        errors.append(f"labels.stage={stage!r} 非法")
    disc = lab.get("discovered")
    if not is_ph(disc) and not DATE_RE.match(str(disc)):
        errors.append(f"labels.discovered={disc!r} 不是日期")

    ev = e["evidence"]
    if is_ph(ev) or not ev:
        errors.append("evidence 为空：至少需要一个来源链接或文字引用")
    else:
        for item in ev:
            if item.get("kind") == "citation":
                if not str(item.get("text", "")).strip():
                    errors.append(f"citation 缺少 text: {item}")
            elif item.get("kind") not in EVIDENCE_KINDS or not str(item.get("url", "")).startswith("http"):
                errors.append(f"evidence 项非法: {item}")
    for r in e["relations"] or []:
        if r.get("kind") not in ("var", "cross"):
            errors.append(f"relation kind 非法: {r}")
    return errors, warnings


# ---------------------------------------------------------------- relations

def resolve_ref(ref, entries, registry):
    """Entry id or old Markdown number (4.28, HW.01) -> (current id or None, candidate ids)."""
    ref = str(ref or "").strip()
    if ref in entries:
        return ref, [ref]
    hits = {r[0] for r in registry if len(r) > 1 and r[1] == ref and r[0] in entries}
    hits |= {i for i, e in entries.items() if ref in (e.get("legacy_ids") or [])}
    live = {i for i in hits if entries[i].get("status") not in ("merged", "retired")}
    pick = live if len(live) == 1 else hits
    return (next(iter(pick)) if len(pick) == 1 else None), sorted(hits)


def resolve_relations(relations, entries, registry, ns, self_id=None, sib=None):
    """Map relation targets written as old numbers to current ids.

    Returns (relations, echo, errors); echo lists each target's title so the caller can check
    it is the intended entry (the 2026-10-13 scan meant 4.28 but wrote INF-0328, which is 4.12).
    `cross` targets outside this catalog are looked up in the sibling catalog (`sib`, see sibling());
    without one they are kept as written.
    """
    out_rel, echo, errors = [], [], []
    for r in relations or []:
        if not isinstance(r, dict):
            errors.append(f"relation 必须是对象: {r}")
            continue
        given = r.get("target")
        if r.get("kind") == "cross" and sib and not str(given).startswith(ns + "-"):
            tid, cands = resolve_ref(given, sib["entries"], sib["registry"])
            if tid is None:
                why = f"对应多个条目 {', '.join(cands)}，请写新编号" if len(cands) > 1 else "不存在"
                errors.append(f"cross 目标 {given} 在另一编目（{sib['ns']}）中{why}"
                              f"（用 python3 {os.path.join(sib['root'], 'tools', 'catalog_tool.py')} find 查）")
                out_rel.append(r)
                continue
            out_rel.append({**r, "target": tid})
            item = {"kind": "cross", "target": tid, "catalog": sib["ns"], "title": sib["entries"][tid].get("title")}
            if sib["entries"][tid].get("status") != "active":
                item["status"] = sib["entries"][tid].get("status")
            if tid != given:
                item["from"] = given
            echo.append(item)
            continue
        tid, cands = resolve_ref(given, entries, registry)
        local = r.get("kind") == "var" or str(given).startswith(ns + "-") or bool(cands)
        if tid is None and local:
            why = f"对应多个条目 {', '.join(cands)}，请写新编号" if len(cands) > 1 else "不存在（可写新编号或旧编号，用 find 查）"
            errors.append(f"{r.get('kind')} 目标 {given} {why}")
            out_rel.append(r)
            continue
        if tid and tid == self_id:
            errors.append(f"{r.get('kind')} 目标 {given} 指向条目自己")
        rel = {**r, "target": tid or given}
        out_rel.append(rel)
        item = {"kind": rel.get("kind"), "target": rel["target"]}
        if tid:
            item["title"] = entries[tid].get("title")
            if entries[tid].get("status") != "active":
                item["status"] = entries[tid].get("status")
        else:
            item["title"] = "（不在本编目，按原样保留）"
        if tid and tid != given:
            item["from"] = given
        echo.append(item)
    return out_rel, echo, errors


# ---------------------------------------------------------------- new

def next_id(ns, entries, registry):
    nums = [int(i.split("-")[1]) for i in entries if i.startswith(ns + "-")]
    nums += [int(r[0].split("-")[1]) for r in registry if r[0].startswith(ns + "-")]
    return f"{ns}-{max(nums, default=0) + 1:04d}"


def cmd_new(a):
    cfg, voc = config(), vocab()
    raw = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        out({"ok": False, "errors": [f"JSON 无法解析: {exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）"]}, 2)
    if not isinstance(data, dict):
        out({"ok": False, "errors": ["输入必须是一个 JSON 对象"]}, 2)

    errors = []
    original = data.get("original") if isinstance(data.get("original"), dict) else {}
    for k in cfg["narrative_required"]:
        if not str(original.get(k, "")).strip():
            errors.append(f"original.{k} 不能为空（扫描的原始说明）")
    # the narrative is the full write-up the audit checks against; the structured fields compress it
    cause = data.get("cause") if isinstance(data.get("cause"), dict) else {}
    effect = data.get("effect") if isinstance(data.get("effect"), dict) else {}
    hint = "原始说明要写全（代码/配置层面的原因、版本、证据、触发条件），结构化字段只是它的压缩，其中每个事实都必须能在原始说明里找到"
    # behavior is what users search by (retrieval embeds the narrative), so it needs the observed
    # symptom in their words, not a one-line restatement of the mechanism
    symptom = ("behavior 写使用者能直接观察到的现象：输出/指标/日志里看到什么、在什么条件下出现、影响多大，"
               "用使用者描述问题时会用的说法；原因与代码细节写在 mechanism")
    for k, floor in (("mechanism", 80), ("behavior", 80)):
        n = len(str(original.get(k, "") or ""))
        if 0 < n < floor:
            errors.append(f"original.{k} 只有 {n} 字，至少 {floor} 字：{hint if k == 'mechanism' else symptom}")
    told = sum(len(str(original.get(k, "") or "")) for k in ("mechanism", "behavior", "trigger_surface"))
    summary = len(str(cause.get("behavior") or "")) + len(str(effect.get("detail") or ""))
    if told and told < summary:
        errors.append(f"original 的机制+行为效应+触发面共 {told} 字，短于结构化 cause.behavior+effect.detail 的 {summary} 字：{hint}")
    evidence = evidence_of(original.get("source", ""))
    if not evidence:
        errors.append("original.source 必须含至少一个可点击链接（GitHub issue/PR、arXiv 或论文 URL）")

    entries, registry = load_entries(), load_registry()
    if a.legacy_id and any(len(r) > 1 and r[1] == a.legacy_id for r in registry):
        errors.append(f"旧编号 {a.legacy_id} 已登记")
    if evidence and not a.allow_duplicate_source:
        seen = {norm_url(x["url"]) for x in evidence}
        refs = gh_refs(x["url"] for x in evidence)
        dup = sorted(i for i, e in entries.items()
                     if (isinstance(e["evidence"], list) and seen & {norm_url(x["url"]) for x in e["evidence"] if x.get("url")})
                     or any(mentions(e, repo, num) for repo, num in refs))
        if dup:
            errors.append(f"来源已被已有条目引用：{', '.join(dup)}。同一机制请用 update 补充证据；"
                          "确属不同机制时加 --allow-duplicate-source 重新提交")
    labels = data.get("labels") if isinstance(data.get("labels"), dict) else {}
    relations, rel_echo, rel_errors = resolve_relations(data.get("relations"), entries, registry, cfg["namespace"],
                                                       sib=sibling())
    errors += rel_errors
    if labels.get("coverage") == "var" and not any(r.get("kind") == "var" for r in relations):
        errors.append("coverage=var 时 relations 必须写 [{\"kind\": \"var\", \"target\": \"<已有条目的新编号或旧编号>\"}]")

    mitigation = data.get("mitigation") if isinstance(data.get("mitigation"), dict) else {}
    entry = {
        "id": next_id(cfg["namespace"], entries, registry),
        "legacy_ids": [a.legacy_id] if a.legacy_id else [],
        "slug": data.get("slug"),
        "aliases": [],
        "title": data.get("title"),
        "kind": data.get("kind"),
        "status": "pending_audit",
        "cause": data.get("cause"),
        "effect": data.get("effect"),
        "mitigation": {"detect": mitigation.get("detect", TBD), "fix": mitigation.get("fix", TBD)},
        "labels": labels,
        "remarks": data.get("remarks", "") or "",
        "evidence": evidence,
        "relations": relations,
        "original": {k: original[k] for k in NARRATIVE_ORDER if k in original}
                    | {k: v for k, v in original.items() if k not in NARRATIVE_ORDER},
        "migration": {"method": "scan", "scanned": today()},
    }
    if any(e["slug"] == entry["slug"] for e in entries.values()):
        errors.append(f"slug {entry['slug']} 已存在，换一个")
    errs, warns = check_entry(entry, voc, strict_lengths=True)
    errors += errs
    if errors:
        out({"ok": False, "errors": errors, "warnings": warns}, 2)
    shown = {"relations": rel_echo} if rel_echo else {}
    if a.dry_run:
        out({"ok": True, "dry_run": True, "id": entry["id"], **shown, "warnings": warns})
    path = os.path.join(ENTRIES, f"{entry['id']}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(entry))
    with open(REGISTRY, "a", encoding="utf-8") as fh:
        fh.write(f"{entry['id']}\t{a.legacy_id or ''}\n")
    out({"ok": True, "id": entry["id"], "path": os.path.relpath(path, ROOT), "status": "pending_audit", **shown,
         "warnings": warns})


# ---------------------------------------------------------------- update

def set_path(obj, dotted, value):
    keys = dotted.split(".")
    cur = obj
    for k in keys[:-1]:
        if k.isdigit():
            cur = cur[int(k)]
        else:
            cur = cur.setdefault(k, {})
    last = keys[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


def cmd_update(a):
    voc = vocab()
    path = os.path.join(ENTRIES, f"{a.id}.yaml")
    if not os.path.exists(path):
        out({"ok": False, "errors": [f"{a.id} 不存在（用 find 查编号）"]}, 2)
    e = load_yaml(path)
    changed = []
    for url in a.add_evidence or []:
        items = evidence_of(url)
        if not items:
            out({"ok": False, "errors": [f"不是可识别的链接: {url}"]}, 2)
        if not isinstance(e["evidence"], list):
            e["evidence"] = []
        known = {norm_url(x["url"]) for x in e["evidence"] if x.get("url")}
        for it in items:
            if norm_url(it["url"]) not in known:
                e["evidence"].append(it)
                changed.append("evidence")
    if a.note:
        notes = e["original"].setdefault("notes", [])
        notes.append(f"{today()} {a.note}")
        changed.append("original.notes")
    for item in a.set or []:
        if "=" not in item:
            out({"ok": False, "errors": [f"--set 需要 path=value: {item}"]}, 2)
        k, v = item.split("=", 1)
        if k.split(".")[0] in ("id", "legacy_ids", "evidence", "original", "migration"):
            out({"ok": False, "errors": [f"{k} 不能用 --set 修改"]}, 2)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        set_path(e, k, v)
        changed.append(k)
    shown = {}
    if a.relation or a.drop_relation or any(k.split(".")[0] == "relations" for k in changed):
        entries, registry = load_entries(), load_registry()
        rels = list(e.get("relations") or [])
        errors = []
        for spec in a.drop_relation or []:
            kind, _, ref = spec.partition(":")
            tid, _ = resolve_ref(ref, entries, registry)

            def matches(r, kind=kind, ref=ref, tid=tid):
                if r.get("kind") != kind:
                    return False
                target = r.get("target")
                if target:
                    return target in (ref, tid)
                # Migration leftovers carry only target_raw -- the slug the extractor meant to
                # point at, which for all 45 of them never existed as any entry. Drop those by
                # that raw value. Guard the falsy case: an unresolvable ref (tid None) used to
                # match every target-less relation on the entry and drop it silently.
                return r.get("target_raw") == ref
            keep = [r for r in rels if not matches(r)]
            if len(keep) == len(rels):
                shown_rels = ", ".join(r.get("kind", "") + ":" + str(r.get("target") or r.get("target_raw"))
                                       for r in rels) or "无"
                errors.append(f"没有关系 {spec} 可删除（现有：{shown_rels}）")
            rels = keep
        for spec in a.relation or []:
            kind, _, ref = spec.partition(":")
            if kind not in ("var", "cross") or not ref:
                errors.append(f"--relation 写成 var:<编号> 或 cross:<编号>: {spec}")
                continue
            rels.append({"kind": kind, "target": ref})
        rels, echo, rel_errors = resolve_relations(rels, entries, registry, config()["namespace"], self_id=a.id,
                                                   sib=sibling())
        errors += rel_errors
        seen = [(r.get("kind"), r.get("target")) for r in rels]
        errors += [f"重复的关系 {k}:{t}" for (k, t), n in Counter(seen).items() if n > 1]
        if (e.get("labels") or {}).get("coverage") == "var" and not any(r.get("kind") == "var" for r in rels):
            errors.append("coverage=var 的条目至少要保留一个 var 关系")
        if errors:
            out({"ok": False, "errors": errors}, 2)
        if rels != (e.get("relations") or []):
            changed.append("relations")
        e["relations"] = rels
        shown = {"relations": echo}
    if not changed:
        out({"ok": False, "errors": ["没有要修改的内容"]}, 2)
    errs, warns = check_entry(e, voc)
    if errs:
        out({"ok": False, "errors": errs}, 2)
    e.setdefault("migration", {}).setdefault("updates", []).append({"date": today(), "fields": sorted(set(changed))})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(e))
    out({"ok": True, "id": a.id, "changed": sorted(set(changed)), **shown, "warnings": warns})


# ---------------------------------------------------------------- find

def cmd_find(a):
    q = a.query.strip()
    ql = q.lower()
    gh = re.fullmatch(r"([\w.\-]+/[\w.\-]+)#(\d+)", q)
    hits = []
    for eid, e in load_entries().items():
        urls = [x.get("url", "") for x in e["evidence"]] if isinstance(e["evidence"], list) else []
        if (q == eid or q in e.get("legacy_ids", []) or ql in str(e.get("slug", "")).lower()
                or any(ql in u.lower() for u in urls)
                or (gh and any(re.search(rf"github\.com/{re.escape(gh.group(1))}/(issues|pull)/{gh.group(2)}\b", u, re.I) for u in urls))
                or (gh and mentions(e, gh.group(1).split("/")[1].lower(), gh.group(2)))
                or ql in str(e.get("title", "")).lower() or ql in str(e["original"].get("source", "")).lower()):
            hits.append({"id": eid, "status": e["status"], "slug": e["slug"], "title": e["title"],
                         "legacy_ids": e.get("legacy_ids", []), "evidence": urls[:3]})
    out({"query": q, "count": len(hits), "hits": hits[:50]})


# ---------------------------------------------------------------- views

CSV_COLUMNS = ["id", "legacy_ids", "status", "kind", "title", "subsystem", "stack", "component", "cause_behavior",
               "effect_objects", "effect_detail", "visibility", "mitigation_detect", "mitigation_fix",
               "family", "trigger", "stage", "confidence", "coverage", "discovered", "evidence", "relations", "slug"]


def flat(e):
    objs = e["cause"]["objects"] if isinstance(e["cause"]["objects"], list) else []
    lab = e["labels"]
    j = lambda v: "；".join(map(str, v)) if isinstance(v, list) else str(v)
    return {
        "id": e["id"], "legacy_ids": j(e.get("legacy_ids", [])), "status": e["status"], "kind": e["kind"], "title": e["title"],
        "subsystem": j([o.get("subsystem") for o in objs]), "stack": j([o.get("stack") for o in objs]),
        "component": j([o.get("component") for o in objs]), "cause_behavior": e["cause"]["behavior"],
        "effect_objects": j(e["effect"]["objects"]), "effect_detail": e["effect"]["detail"],
        "visibility": e["effect"].get("visibility"), "mitigation_detect": e["mitigation"]["detect"],
        "mitigation_fix": e["mitigation"]["fix"], "family": lab.get("family"), "trigger": lab.get("trigger"),
        "stage": j(lab.get("stage")), "confidence": lab.get("confidence"), "coverage": lab.get("coverage"),
        "discovered": lab.get("discovered"),
        "evidence": j([x.get("url") or f"引用：{x.get('text', '')}" for x in e["evidence"]]) if isinstance(e["evidence"], list) else str(e["evidence"]),
        "relations": j([f"{r.get('kind')}:{r.get('target', r.get('target_raw'))}" for r in e["relations"] or []]),
        "slug": e["slug"],
    }


def render_markdown(entries, cfg, voc):
    order = list(voc["subsystem"]) + list(voc["placeholders"])
    groups = {}
    for e in entries.values():
        objs = e["cause"]["objects"] if isinstance(e["cause"]["objects"], list) else [{}]
        groups.setdefault(objs[0].get("subsystem", TBD), []).append(e)
    status = Counter(e["status"] for e in entries.values())
    L = [f"# {cfg['title']}（结构化视图）", "",
         "> 本文件由 `python3 tools/catalog_tool.py build` 从 `catalog/entries/*.yaml` 生成，**不要手改**。",
         "> 字段说明见 `catalog/SCHEMA.md`，受控词表见 `catalog/vocab.yaml`。", "",
         f"共 {len(entries)} 条；状态：" + "、".join(f"{k} {v}" for k, v in sorted(status.items())), "",
         "| 原因子系统 | 条数 |", "|---|---|"]
    L += [f"| {g} | {len(groups[g])} |" for g in order if g in groups] + [""]
    for g in order:
        if g not in groups:
            continue
        L += [f"## {g}", ""]
        for e in sorted(groups[g], key=lambda x: x["id"]):
            f = flat(e)
            lab = [x for x in (f["family"], f["trigger"], f["stage"], f["confidence"], f["coverage"]) if x not in (None, NA, "None")]
            L += [f"### {e['id']} `{e['slug']}` {e['title']}", "",
                  f"`{e['kind']}` · `{e['status']}` · " + " · ".join(f"`{x}`" for x in lab)
                  + f" · 发现 {f['discovered']}" + (f" · 旧号 {f['legacy_ids']}" if f["legacy_ids"] else ""), "",
                  f"- **原因对象**：" + "；".join(f"{o.get('subsystem')} / {o.get('stack')} / {o.get('component')}"
                                              for o in (e["cause"]["objects"] if isinstance(e["cause"]["objects"], list) else [])),
                  f"- **原因行为**：{f['cause_behavior']}",
                  f"- **影响对象**：{f['effect_objects']}（{f['visibility']}）",
                  f"- **影响详情**：{f['effect_detail']}",
                  f"- **改进措施**：检测 {f['mitigation_detect']}；修复 {f['mitigation_fix']}",
                  f"- **来源**：{f['evidence']}"]
            if f["relations"]:
                L.append(f"- **关联**：{f['relations']}")
            L.append("")
    return "\n".join(L)


def render_csv(entries):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for eid in sorted(entries):
        w.writerow(flat(entries[eid]))
    return buf.getvalue()


def write_xlsx(entries, path):
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return False
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fault Catalog"
    ws.append(CSV_COLUMNS)
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="2F5496")
    for eid in sorted(entries):
        ws.append([flat(entries[eid])[k] for k in CSV_COLUMNS])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes, ws.auto_filter.ref = "A2", ws.dimensions
    wb.properties.created = wb.properties.modified = dt.datetime(2026, 1, 1)
    buf = io.BytesIO()
    wb.save(buf)
    # openpyxl stamps every zip member with the current time; re-pack with a fixed timestamp so the
    # same entries always give byte-identical files (no binary churn in every scan/audit commit)
    import zipfile
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            fixed = zipfile.ZipInfo(info.filename, date_time=(2026, 1, 1, 0, 0, 0))
            fixed.compress_type, fixed.external_attr = zipfile.ZIP_DEFLATED, info.external_attr
            data = src.read(info.filename)
            if info.filename == "docProps/core.xml":     # openpyxl overwrites `modified` with now() on save
                data = re.sub(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)", rb"\g<1>2026-01-01T00:00:00Z\g<2>", data)
            dst.writestr(fixed, data)
    return True


def xlsx_rows(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True).active
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def build_views(target_dir):
    cfg, voc, entries = config(), vocab(), load_entries()
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "CATALOG.zh.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(entries, cfg, voc))
    with open(os.path.join(target_dir, "fault_catalog.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(render_csv(entries))
    xlsx = write_xlsx(entries, os.path.join(target_dir, "fault_catalog.xlsx"))
    return len(entries), xlsx


def cmd_build(a):
    n, xlsx = build_views(VIEWS)
    out({"ok": True, "entries": n, "views": ["views/CATALOG.zh.md", "views/fault_catalog.csv"] + (["views/fault_catalog.xlsx"] if xlsx else [])})


# ---------------------------------------------------------------- validate / stats

def cmd_validate(a):
    cfg, voc, entries, sib = config(), vocab(), load_entries(), sibling()
    errors, warnings = [], []
    for name in sorted(os.listdir(ENTRIES)):
        if name.endswith(".yaml") and name[:-5] not in entries:
            errors.append(f"{name}: 文件名与 id 不一致")
    for eid, e in entries.items():
        errs, warns = check_entry(e, voc)
        errors += [f"{eid}: {m}" for m in errs]
        warnings += [f"{eid}: {m}" for m in warns]
        if not eid.startswith(cfg["namespace"] + "-"):
            errors.append(f"{eid}: 命名空间应为 {cfg['namespace']}")
        for r in e["relations"] or []:
            t = r.get("target")
            if t and t.startswith(cfg["namespace"] + "-") and t not in entries:
                errors.append(f"{eid}: relation 目标不存在 {t}")
            elif t and sib and t.startswith(sib["ns"] + "-") and t not in sib["entries"]:
                warnings.append(f"{eid}: cross 目标 {t} 在另一编目中不存在")
            if "target_raw" in r:
                warnings.append(f"{eid}: {r['kind']} 目标无法解析 {r['target_raw']}")
    pats = load_patterns()
    for name in (sorted(os.listdir(PATTERNS)) if os.path.isdir(PATTERNS) else []):
        if name.endswith(".yaml") and name[:-5] not in pats:
            errors.append(f"patterns/{name}: 文件名与 id 不一致")
    want = {}
    for pid, p in sorted(pats.items()):
        if not PAT_RE.match(pid):
            errors.append(f"{pid}: 模式编号格式应为 PAT-0001")
        if p.get("status") == "merged":
            if not p.get("merged_into"):
                errors.append(f"{pid}: status=merged 但缺 merged_into")
            continue
        ms = p.get("members") or []
        if len(ms) < 2:
            errors.append(f"{pid}: 模式至少要有 2 个成员，现有 {len(ms)}")
        for m in ms:
            if m not in entries:
                errors.append(f"{pid}: 成员不存在 {m}")
            elif want.setdefault(m, pid) != pid:
                errors.append(f"{m}: 同时属于 {want[m]} 与 {pid}")
        if p.get("representative") not in ms:
            errors.append(f"{pid}: representative {p.get('representative')} 不在成员里")
        if p.get("title") == TBD or p.get("shared_mechanism") == TBD:
            warnings.append(f"{pid}: title/shared_mechanism 仍是占位符，待生成")
    for eid, e in entries.items():
        pid = e.get("pattern")
        if pid is None:
            if eid in want:
                errors.append(f"{eid}: 属于 {want[eid]} 但缺 pattern 字段：先运行 pattern-sync")
        elif pid not in pats:
            errors.append(f"{eid}: pattern 指向不存在的模式 {pid}")
        elif want.get(eid) != pid:
            errors.append(f"{eid}: pattern={pid} 与模式成员表不一致：先运行 pattern-sync")
    registry = load_registry()
    reg_ids = Counter(r[0] for r in registry)
    errors += [f"{i}: 编号表重复登记" for i, n in reg_ids.items() if n > 1]
    errors += [f"{i}: 不在 id_registry.tsv" for i in entries if i not in reg_ids]
    legacy = Counter(old for e in entries.values() for old in e.get("legacy_ids", []))
    errors += [f"旧编号 {o} 出现在 {n} 个条目中" for o, n in legacy.items() if n > 1]
    slugs = Counter(e["slug"] for e in entries.values())
    warnings += [f"slug {s} 被 {n} 个条目使用" for s, n in slugs.items() if n > 1]
    with tempfile.TemporaryDirectory() as tmp:
        build_views(tmp)
        for name in ("CATALOG.zh.md", "fault_catalog.csv"):
            mine = os.path.join(VIEWS, name)
            if not os.path.exists(mine) or open(mine, encoding="utf-8").read() != open(os.path.join(tmp, name), encoding="utf-8").read():
                errors.append(f"views/{name} 与条目不一致：先运行 python3 tools/catalog_tool.py build")
        fresh, mine = os.path.join(tmp, "fault_catalog.xlsx"), os.path.join(VIEWS, "fault_catalog.xlsx")
        if os.path.exists(fresh) and (not os.path.exists(mine) or xlsx_rows(mine) != xlsx_rows(fresh)):
            errors.append("views/fault_catalog.xlsx 与条目不一致：先运行 python3 tools/catalog_tool.py build")
    res = {"ok": not errors, "entries": len(entries), "patterns": len([p for p in pats.values() if p.get("status") != "merged"]),
           "errors": errors[:200], "n_errors": len(errors),
           "n_warnings": len(warnings), "pending_audit": sum(e["status"] == "pending_audit" for e in entries.values())}
    if a.verbose:
        res["warnings"] = warnings
    out(res, 0 if not errors else 1)


def cmd_cross_sync(a):
    """Give every entry here the reverse `cross` link of each sibling entry that points at it.

    The daily scan only writes its own repo, so a link made by one scan has no way back; the daily
    audit runs this in both repos (deterministic, no gateway calls).
    """
    sib = sibling()
    if sib is None:
        out({"ok": True, "added": [], "skipped": "没有配置或找不到另一编目（config.yaml 的 sibling）"})
    entries, add = load_entries(), {}
    for sid, se in sorted(sib["entries"].items()):
        if se.get("status") in ("merged", "retired"):
            continue
        for r in se.get("relations") or []:
            t = r.get("target")
            if r.get("kind") == "cross" and t in entries and entries[t].get("status") not in ("merged", "retired") \
                    and not any(x.get("kind") == "cross" and x.get("target") == sid for x in entries[t].get("relations") or []):
                add.setdefault(t, []).append(sid)
    if not a.dry_run:
        for t, sids in add.items():
            e = entries[t]
            e["relations"] = list(e.get("relations") or []) + [{"kind": "cross", "target": s} for s in sids]
            e.setdefault("migration", {}).setdefault("updates", []).append({"date": today(), "fields": ["relations"]})
            with open(os.path.join(ENTRIES, f"{t}.yaml"), "w", encoding="utf-8") as fh:
                fh.write(dump_yaml(e))
    out({"ok": True, "dry_run": a.dry_run,
         "added": [{"id": t, "cross": s, "title": sib["entries"][s].get("title")} for t, sids in sorted(add.items()) for s in sids]})


def load_patterns():
    if not os.path.isdir(PATTERNS):
        return {}
    out_ = {}
    for name in sorted(os.listdir(PATTERNS)):
        if name.endswith(".yaml"):
            d = load_yaml(os.path.join(PATTERNS, name))
            if isinstance(d, dict) and d.get("id"):
                out_[d["id"]] = d
    return out_


def pattern_trees(entries):
    """var 关系构成的森林：每棵树一个模式。

    每个条目最多一条 var 出边，所以这是森林而非一般图。注意树是连通分量，
    因此深度 >1 的树隐含了「var 可传递」这个假设——它并不总成立，深链的模式
    要人工复核（见 RAG normalize/pattern/PLAN.md）。
    """
    live = {i: e for i, e in entries.items() if e.get("status") not in ("merged", "retired")}
    edges = []
    for i, e in sorted(live.items()):
        for r in e.get("relations") or []:
            t = r.get("target")
            if r.get("kind") == "var" and t in live:
                edges.append((i, t))
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_, b_ in edges:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    indeg = Counter(b for _, b in edges)
    trees = {}
    for n in list(parent):
        trees.setdefault(find(n), []).append(n)
    return [{"members": sorted(ms),
             "representative": min(sorted(ms), key=lambda i: (-indeg[i], i))}
            for ms in trees.values()], indeg


def place_pattern_key(e, pid):
    """把 pattern 插在 relations 之后，其余键的相对顺序不变（diff 最小）。"""
    new = {}
    for k, v in e.items():
        if k == "pattern":
            continue
        new[k] = v
        if k == "relations" and pid is not None:
            new["pattern"] = pid
    if pid is not None and "pattern" not in new:
        new["pattern"] = pid
    return new


def cmd_pattern_sync(a):
    """按 var 森林同步 catalog/patterns/ 与条目的 pattern 字段。

    成员关系**派生自 var 关系**，不手工编辑——这样成员不会和关系漂移，而且每日扫描
    新增一条 var 关系后，这条命令就把新条目归入它的族（由 daily_audit.sh 调用，
    与 cross-sync 同理，不花网关调用）。title / shared_mechanism 是生成内容，这里只保留。
    """
    entries, pats = load_entries(), load_patterns()
    trees, _ = pattern_trees(entries)
    live_pats = {i: p for i, p in pats.items() if p.get("status") != "merged"}
    used, assign, created, absorbed, changed = set(), {}, [], [], []
    # 按与既有模式的成员重叠度匹配，保证编号稳定
    for tree in sorted(trees, key=lambda t: (-len(t["members"]), t["members"][0])):
        ms = set(tree["members"])
        cands = sorted(((len(ms & set(p.get("members") or [])), i) for i, p in live_pats.items()
                        if i not in used and ms & set(p.get("members") or [])), reverse=True)
        if cands:
            pid = cands[0][1]
            for _, other in cands[1:]:
                absorbed.append((other, pid))
                used.add(other)
        else:
            ns = config()["namespace"]
            nums = [int(i.rsplit("-", 1)[1]) for i in pats if PAT_RE.match(i)] or [0]
            pid = f"PAT-{ns}-{max(nums) + 1:04d}"
            pats[pid] = {"id": pid, "title": "待定", "shared_mechanism": "待定",
                         "representative": tree["representative"], "members": tree["members"]}
            created.append(pid)
        used.add(pid)
        assign[pid] = tree
    # 写模式文件
    if not a.dry_run:
        os.makedirs(PATTERNS, exist_ok=True)
    for pid, tree in assign.items():
        doc = pats.get(pid) or {"id": pid, "title": "待定", "shared_mechanism": "待定"}
        before = (doc.get("members"), doc.get("representative"))
        doc = {"id": pid, "title": doc.get("title", "待定"),
               "shared_mechanism": doc.get("shared_mechanism", "待定"),
               "representative": tree["representative"], "members": tree["members"]}
        if before != (doc["members"], doc["representative"]):
            changed.append(pid)
        if not a.dry_run:
            with open(os.path.join(PATTERNS, f"{pid}.yaml"), "w", encoding="utf-8") as fh:
                fh.write(dump_yaml(doc))
    for old, into in absorbed:
        if not a.dry_run:
            doc = dict(pats[old], status="merged", merged_into=into, members=[], representative=None)
            with open(os.path.join(PATTERNS, f"{old}.yaml"), "w", encoding="utf-8") as fh:
                fh.write(dump_yaml(doc))
    # 写条目的 pattern 字段
    want = {m: pid for pid, tree in assign.items() for m in tree["members"]}
    touched = []
    for eid, e in sorted(entries.items()):
        now, should = e.get("pattern"), want.get(eid)
        if now == should:
            continue
        touched.append({"id": eid, "from": now, "to": should})
        if not a.dry_run:
            doc = place_pattern_key(e, should)
            doc.setdefault("migration", {}).setdefault("updates", []).append(
                {"date": today(), "fields": ["pattern"]})
            with open(os.path.join(ENTRIES, f"{eid}.yaml"), "w", encoding="utf-8") as fh:
                fh.write(dump_yaml(doc))
    out({"ok": True, "dry_run": a.dry_run, "patterns": len(assign), "covered": len(want),
         "created": created, "membership_changed": changed,
         "absorbed": [{"pattern": o, "into": i} for o, i in absorbed],
         "entries_touched": len(touched), "touched": touched[:50]})


def cmd_pattern_set(a):
    """写模式的 title / shared_mechanism。成员与代表由 pattern-sync 派生，这里不改。"""
    pats = load_patterns()
    if a.id not in pats:
        out({"ok": False, "errors": [f"没有模式 {a.id}"]}, 2)
    doc, errs = pats[a.id], []
    if a.title is not None:
        if len(a.title) > LIMITS["title"]:
            errs.append(f"title 长 {len(a.title)} 字，上限 {LIMITS['title']} 字")
        doc["title"] = a.title
    if a.shared_mechanism is not None:
        doc["shared_mechanism"] = a.shared_mechanism
    if errs:
        out({"ok": False, "errors": errs}, 2)
    doc = {k: doc[k] for k in ("id", "title", "shared_mechanism", "representative", "members")
           if k in doc} | {k: v for k, v in doc.items()
                           if k not in ("id", "title", "shared_mechanism", "representative", "members")}
    if not a.dry_run:
        with open(os.path.join(PATTERNS, f"{a.id}.yaml"), "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(doc))
    out({"ok": True, "dry_run": a.dry_run, "id": a.id, "title": doc.get("title"),
         "members": len(doc.get("members") or [])})


def cmd_stats(a):
    entries = load_entries()
    sub = Counter((e["cause"]["objects"][0] or {}).get("subsystem") for e in entries.values() if isinstance(e["cause"]["objects"], list))
    out({"entries": len(entries), "status": dict(Counter(e["status"] for e in entries.values())),
         "kind": dict(Counter(e["kind"] for e in entries.values())), "subsystem": dict(sub.most_common())})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("new")
    p.add_argument("--file")
    p.add_argument("--legacy-id")
    p.add_argument("--allow-duplicate-source", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("update")
    p.add_argument("id")
    p.add_argument("--add-evidence", action="append")
    p.add_argument("--note")
    p.add_argument("--set", action="append")
    p.add_argument("--relation", action="append", metavar="KIND:REF",
                   help="add var:/cross: relation; REF = current id or old number (4.28), echoed with its title")
    p.add_argument("--drop-relation", action="append", metavar="KIND:REF")
    p = sp.add_parser("find")
    p.add_argument("query")
    sp.add_parser("build")
    p = sp.add_parser("validate")
    p.add_argument("--verbose", action="store_true")
    sp.add_parser("stats")
    p = sp.add_parser("pattern-sync")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("pattern-set")
    p.add_argument("id")
    p.add_argument("--title")
    p.add_argument("--shared-mechanism", dest="shared_mechanism")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("cross-sync")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"new": cmd_new, "update": cmd_update, "find": cmd_find, "build": cmd_build,
     "validate": cmd_validate, "stats": cmd_stats, "cross-sync": cmd_cross_sync,
     "pattern-sync": cmd_pattern_sync, "pattern-set": cmd_pattern_set}[a.cmd](a)


if __name__ == "__main__":
    main()
