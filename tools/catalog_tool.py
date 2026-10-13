#!/usr/bin/env python3
"""Structured fault catalog: the only way the daily scan writes entries (identical file in both catalog repos).

  python3 tools/catalog_tool.py new [--file entry.json] [--legacy-id 1.295] [--allow-duplicate-source] [--dry-run]
  python3 tools/catalog_tool.py update INF-0044 [--add-evidence URL ...] [--note TEXT] [--set path=value ...]
  python3 tools/catalog_tool.py find <ID | legacy id | slug | URL | owner/repo#N | text>
  python3 tools/catalog_tool.py build           # regenerate views/ from catalog/entries
  python3 tools/catalog_tool.py validate        # all entries + registry + views up to date; exit 1 on error
  python3 tools/catalog_tool.py stats

`new` reads ONE JSON object (stdin or --file): the structured fields plus the scan's narrative under
"original". On a problem it prints {"ok": false, "errors": [...]} and exits 2 — fix the JSON and resubmit.
On success it assigns the next ID, writes catalog/entries/<ID>.yaml with status pending_audit and
prints {"ok": true, "id": ...}. Layout and fields: catalog/SCHEMA.md; vocabulary: catalog/vocab.yaml;
per-repo settings: catalog/config.yaml.
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
LABEL_ENUMS = {
    "family": ("infra_off", "model_on", "ambiguous"),
    "trigger": ("config", "yes", "partial", "other_stack", "hw_specific", "no", "not_target"),
    "confidence": ("verified", "documented", "speculative"),
    "coverage": ("NEW", "var", "rationale", "existing"),
}
STAGES = ("pretrain", "sft", "rl", "shared")
STATUSES = ("active", "pending_audit", "merged", "retired")
EVIDENCE_KINDS = ("github_issue", "github_pr", "arxiv", "paper", "other")
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


def load_entries():
    out = {}
    for name in sorted(os.listdir(ENTRIES)):
        if name.endswith(".yaml"):
            e = load_yaml(os.path.join(ENTRIES, name))
            out[e["id"]] = e
    return out


def load_registry():
    rows = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as fh:
            next(fh)
            rows = [tuple(line.rstrip("\n").split("\t")) for line in fh if line.strip()]
    return rows


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
    items = []
    for u in urls:
        if re.search(r"github\.com/[^/]+/[^/]+/pull/\d+", u):
            kind = "github_pr"
        elif re.search(r"github\.com/[^/]+/[^/]+/issues/\d+", u):
            kind = "github_issue"
        elif "arxiv.org" in u:
            kind = "arxiv"
        elif re.search(r"aclanthology|doi\.org|usenix|acm\.org|ieee|openreview", u, re.I):
            kind = "paper"
        else:
            kind = "other"
        items.append({"url": u, "kind": kind})
    return items


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
    for k, floor in (("mechanism", 80), ("behavior", 40)):
        n = len(str(original.get(k, "") or ""))
        if 0 < n < floor:
            errors.append(f"original.{k} 只有 {n} 字，至少 {floor} 字：{hint}")
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
    relations = data.get("relations") or []
    if labels.get("coverage") == "var":
        targets = [r.get("target") for r in relations if isinstance(r, dict) and r.get("kind") == "var"]
        if not targets:
            errors.append("coverage=var 时 relations 必须写 [{\"kind\": \"var\", \"target\": \"<已有条目编号>\"}]")
        errors += [f"var 目标 {t} 不存在（用 find 查编号）" for t in targets if t not in entries]

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
    if a.dry_run:
        out({"ok": True, "dry_run": True, "id": entry["id"], "warnings": warns})
    path = os.path.join(ENTRIES, f"{entry['id']}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(entry))
    with open(REGISTRY, "a", encoding="utf-8") as fh:
        fh.write(f"{entry['id']}\t{a.legacy_id or ''}\n")
    out({"ok": True, "id": entry["id"], "path": os.path.relpath(path, ROOT), "status": "pending_audit", "warnings": warns})


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
    if not changed:
        out({"ok": False, "errors": ["没有要修改的内容"]}, 2)
    errs, warns = check_entry(e, voc)
    if errs:
        out({"ok": False, "errors": errs}, 2)
    e.setdefault("migration", {}).setdefault("updates", []).append({"date": today(), "fields": sorted(set(changed))})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(e))
    out({"ok": True, "id": a.id, "changed": sorted(set(changed)), "warnings": warns})


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
    cfg, voc, entries = config(), vocab(), load_entries()
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
            if "target_raw" in r:
                warnings.append(f"{eid}: {r['kind']} 目标无法解析 {r['target_raw']}")
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
    res = {"ok": not errors, "entries": len(entries), "errors": errors[:200], "n_errors": len(errors),
           "n_warnings": len(warnings), "pending_audit": sum(e["status"] == "pending_audit" for e in entries.values())}
    if a.verbose:
        res["warnings"] = warnings
    out(res, 0 if not errors else 1)


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
    p = sp.add_parser("find")
    p.add_argument("query")
    sp.add_parser("build")
    p = sp.add_parser("validate")
    p.add_argument("--verbose", action="store_true")
    sp.add_parser("stats")
    a = ap.parse_args()
    {"new": cmd_new, "update": cmd_update, "find": cmd_find, "build": cmd_build,
     "validate": cmd_validate, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
