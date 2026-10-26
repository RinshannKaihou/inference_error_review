"""Tests for tools/catalog_audit.py; each runs against a throwaway catalog with the gateway stubbed out.

    python3 -m unittest tools/test_catalog_audit.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import catalog_audit as A  # noqa: E402
from test_catalog_tool import good_entry  # noqa: E402

# Runs catalog_audit.main() with call() replaced: replies come from a JSON script keyed "<model>|<task>",
# consumed in order (the last one repeats). A reply is {"content": "..."} or {"err": "http 502"}.
STUB = r'''
import json, os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_audit as A
script, used, lock = json.load(open(sys.argv[1])), {}, threading.Lock()
def call(model, system, user, key):
    task = "C" if "分类审核员" in system else "F" if "忠实性审核员" in system else "fix"
    k = f"{model}|{task}"
    with lock:
        n = used[k] = used.get(k, 0) + 1
    replies = script.get(k) or script[f"*|{task}"]
    r = replies[min(n, len(replies)) - 1]
    return (r["content"], None) if "content" in r else (None, r["err"])
A.call = call
sys.argv = ["catalog_audit.py", *sys.argv[2:]]
A.main()
'''

CAT_OK = {"kind": "故障", "subsystem": "算子/内核", "effect": "数值结果", "visibility": "静默错误"}
F_OK = {f: {"ok": True, "problem": "none", "quote": "", "evidence": ""} for f in A.TEXT_FIELDS}


def reply(obj):
    return {"content": json.dumps(obj, ensure_ascii=False)}


class DecideTest(unittest.TestCase):
    def test_too_few_valid_votes_is_short_not_split(self):
        self.assertEqual(A.decide_cat(["量化", None, None], panel=False), ("量化", "short"))
        self.assertEqual(A.decide_cat(["量化", "量化", None, None, None, "量化"], panel=True), ("量化", "short"))
        self.assertEqual(A.decide_cat(["量化", "算子/内核", "量化", "量化"], panel=True), ("量化", "panel6"))
        self.assertEqual(A.decide_cat(["量化", "算子/内核"], panel=False), ("量化", "split"))
        self.assertEqual(A.decide_cat([None, None], panel=False), (None, "unresolved"))
        self.assertEqual(A.decide_bin([True, None, None], panel=False), (True, "short"))
        self.assertEqual(A.decide_bin([True, True, False, False], panel=True), (False, "uncertain"))


class AuditRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "tools"))
        os.makedirs(os.path.join(self.tmp, "catalog", "entries"))
        for f in ("catalog_tool.py", "catalog_audit.py"):
            shutil.copy(os.path.join(HERE, f), os.path.join(self.tmp, "tools"))
        with open(os.path.join(self.tmp, "tools", "stub_run.py"), "w") as fh:
            fh.write(STUB)
        shutil.copy(os.path.join(REPO, "catalog", "vocab.yaml"), os.path.join(self.tmp, "catalog"))
        with open(os.path.join(REPO, "catalog", "config.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        cfg.pop("sibling", None)
        with open(os.path.join(self.tmp, "catalog", "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True)
        with open(os.path.join(self.tmp, "catalog", "id_registry.tsv"), "w") as fh:
            fh.write("new_id\tlegacy_id\n")
        p = subprocess.run([sys.executable, os.path.join(self.tmp, "tools", "catalog_tool.py"), "new"],
                           input=json.dumps(good_entry()), capture_output=True, text=True)
        self.eid = json.loads(p.stdout)["id"]
        self.day = json.loads(subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import catalog_tool as T, json; print(json.dumps(T.today()))",
             os.path.join(self.tmp, "tools")], capture_output=True, text=True).stdout)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def audit(self, script, *args):
        path = os.path.join(self.tmp, "script.json")
        with open(path, "w") as fh:
            json.dump(script, fh, ensure_ascii=False)
        p = subprocess.run([sys.executable, os.path.join(self.tmp, "tools", "stub_run.py"), path, "run", *args],
                           capture_output=True, text=True, env=dict(os.environ, JUDGE_API_KEY="test"))
        return p.returncode, json.loads(p.stdout) if p.stdout.strip().startswith("{") else p.stdout + p.stderr

    def entry(self):
        with open(os.path.join(self.tmp, "catalog", "entries", f"{self.eid}.yaml")) as fh:
            return yaml.safe_load(fh)

    def votes(self):
        path = os.path.join(self.tmp, "catalog", "audit", self.day, "votes.jsonl")
        with open(path) as fh:
            return {r["job"]: r for r in map(json.loads, fh)}

    def test_void_replies_retried_archived_and_counted_short(self):
        code, res = self.audit({
            "glm-5.3-flash|C": [{"content": "我觉得是算子问题"}, reply(CAT_OK)],      # unparsable, then fine on retry
            "deepseek-v4.1-flash|C": [{"err": "http 502"}],                          # void both times
            "qwen3.8-flash|C": [reply(dict(CAT_OK, subsystem="硬件故障"))],           # out-of-vocab label: rejected
            "glm-5.3|C": [{"err": "ReadTimeout"}],
            "*|C": [reply(CAT_OK)], "*|F": [reply(F_OK)]})
        self.assertEqual(code, 0, res)
        self.assertEqual((res["void_replies"], res["uncertain_fields"], res["uncertain_short_fields"]), (3, 4, 4))
        v = self.votes()
        self.assertEqual(v[f"baseC|{self.eid}|glm-5.3-flash"]["attempts"], 2)
        self.assertEqual(v[f"baseC|{self.eid}|glm-5.3-flash"]["result"], CAT_OK)
        self.assertEqual(v[f"baseC|{self.eid}|deepseek-v4.1-flash"]["fail"], "http 502")
        self.assertEqual(v[f"baseC|{self.eid}|qwen3.8-flash"]["fail"], "rejected")
        self.assertIn("硬件故障", v[f"baseC|{self.eid}|qwen3.8-flash"]["raw"])
        self.assertNotIn("fail", v[f"baseF|{self.eid}|deepseek-v4.1-flash"])
        e = self.entry()
        self.assertEqual(e["status"], "active")
        rec = e["migration"]["audit"]
        self.assertEqual(rec["round"], f"daily-audit-{self.day}")
        self.assertEqual(rec["uncertain_short"], list(A.CAT_FIELDS))
        self.assertNotIn("updates", e["migration"])

    def test_reaudit_keeps_previous_record_and_logs_update(self):
        code, res = self.audit({"*|C": [reply(CAT_OK)], "*|F": [reply(F_OK)]})
        self.assertEqual((code, res["uncertain_fields"]), (0, 0), res)
        first = self.entry()["migration"]["audit"]
        quant = {"*|C": [reply(dict(CAT_OK, subsystem="量化"))], "*|F": [reply(F_OK)]}
        code, res = self.audit(quant, "--ids", self.eid, "--dry-run")
        self.assertEqual(code, 0, res)
        self.assertEqual(res["records"][self.eid]["corrected"], {"subsystem": {"from": "算子/内核", "to": "量化"}})
        self.assertEqual(self.entry()["cause"]["objects"][0]["subsystem"], "算子/内核")   # dry run writes nothing
        code, res = self.audit(quant, "--ids", self.eid)
        self.assertEqual(code, 0, res)
        e = self.entry()
        self.assertEqual(e["cause"]["objects"][0]["subsystem"], "量化")
        self.assertEqual(e["migration"]["audit"]["round"], f"reaudit-{self.day}")
        self.assertEqual(e["migration"]["audit"]["previous"], first)
        self.assertEqual(e["migration"]["updates"], [{"date": self.day, "fields": ["cause.objects"]}])
        self.assertEqual(self.votes()[f"baseC|{self.eid}|qwen3.8-flash"]["round"], f"reaudit-{self.day}")

    def test_refuses_entries_that_are_not_pending_or_active(self):
        code, res = self.audit({"*|C": [reply(CAT_OK)], "*|F": [reply(F_OK)]}, "--ids", "ZZZ-0001")
        self.assertEqual(code, 2)
        self.assertIn("ZZZ-0001", res["errors"][0])


if __name__ == "__main__":
    unittest.main()
