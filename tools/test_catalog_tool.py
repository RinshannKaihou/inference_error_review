"""Tests for tools/catalog_tool.py; each runs against a throwaway catalog in a temp dir.

    python3 -m unittest tools/test_catalog_tool.py
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


def good_entry(**over):
    e = {
        "slug": "demo_topk_ignores_indices", "title": "topk 单核路径忽略传入索引", "kind": "故障",
        "cause": {"objects": [{"subsystem": "算子/内核", "stack": "tt-metal", "component": "ttnn.topk 单核 factory"}],
                  "behavior": "单核 factory 硬编码 GENERATE_INDICES，忽略调用方 indices_tensor"},
        "effect": {"objects": ["数值结果"], "detail": "单核形状下返回 iota 而非传入索引，无 crash", "visibility": "静默错误"},
        "mitigation": {"detect": "待定", "fix": "修复 PR #59639 open（未发布）"},
        "labels": {"family": "infra_off", "trigger": "hw_specific", "trigger_review": False, "trigger_note": "tt-metal 专用",
                   "stage": "N/A", "confidence": "documented", "coverage": "NEW", "discovered": "2026-10-13"},
        "relations": [], "remarks": "",
        "original": {"mechanism": "ttnn.topk 的单核 topk_single_core_program_factory 把 GENERATE_INDICES 硬编码为 1 # 注意这里有井号，忽略调用方传入的 indices_tensor payload，只有多核 factory 才读取该张量；单核形状包括 W<8192、非 2 的幂等。",
                     "source": "tt-metal #59639 https://github.com/tenstorrent/tt-metal/issues/59639",
                     "behavior": "单核形状下静默返回 generated iota [0,1,2,...] 而非调用方提供的索引，无 crash、无 NaN；下游按这些索引 gather 时取到错误的行，"
                                 "生成结果悄悄错乱，只有把同一输入换成多核形状或 CPU 参考实现对比时才看得出差异。", "trigger_surface": "tt-metal 单核", "discovery": "2026-10-13 每日扫描"},
    }
    e.update(over)
    return e


class CatalogToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "tools"))
        os.makedirs(os.path.join(self.tmp, "catalog", "entries"))
        shutil.copy(os.path.join(HERE, "catalog_tool.py"), os.path.join(self.tmp, "tools"))
        shutil.copy(os.path.join(REPO, "catalog", "vocab.yaml"), os.path.join(self.tmp, "catalog"))
        cfg = yaml.safe_load(open(os.path.join(REPO, "catalog", "config.yaml")))
        cfg.pop("sibling", None)                                       # tests never read the real other repo
        with open(os.path.join(self.tmp, "catalog", "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True)
        with open(os.path.join(self.tmp, "catalog", "id_registry.tsv"), "w") as fh:
            ns0 = yaml.safe_load(open(os.path.join(self.tmp, "catalog", "config.yaml")))["namespace"]
            fh.write(f"new_id\tlegacy_id\n{ns0}-0007\t1.7\n")
        ns = yaml.safe_load(open(os.path.join(self.tmp, "catalog", "config.yaml")))["namespace"]
        self.ns = ns
        self.run_tool("new", "--legacy-id", "1.8", stdin=json.dumps(good_entry(
            slug="seed_entry", original=dict(good_entry()["original"], source="https://github.com/o/r/issues/1"))))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_tool(self, *args, stdin=None):
        p = subprocess.run([sys.executable, os.path.join(self.tmp, "tools", "catalog_tool.py"), *args],
                           input=stdin, capture_output=True, text=True)
        return p.returncode, json.loads(p.stdout) if p.stdout.strip().startswith("{") else p.stdout

    def entry(self, eid):
        return yaml.safe_load(open(os.path.join(self.tmp, "catalog", "entries", f"{eid}.yaml")))

    def test_new_assigns_next_id_and_keeps_hash_text(self):
        code, res = self.run_tool("new", stdin=json.dumps(good_entry()))
        self.assertEqual(code, 0, res)
        self.assertEqual(res["id"], f"{self.ns}-0009")          # registry already holds 0007, seed got 0008
        e = self.entry(res["id"])
        self.assertEqual(e["status"], "pending_audit")
        self.assertEqual(e["mitigation"]["fix"], "修复 PR #59639 open（未发布）")   # '#' survives
        self.assertIn("# 注意这里有井号", e["original"]["mechanism"])
        self.assertEqual(e["evidence"], [{"url": "https://github.com/tenstorrent/tt-metal/issues/59639", "kind": "github_issue"}])

    def test_new_rejects_bad_entry_without_writing(self):
        bad = good_entry(title="题" * 41)
        bad["effect"] = {"objects": ["不存在的对象"], "detail": "x"}
        bad["cause"]["objects"][0]["subsystem"] = "硬件故障"
        bad["original"] = dict(bad["original"], source="只有项目名，没有链接")
        before = sorted(os.listdir(os.path.join(self.tmp, "catalog", "entries")))
        code, res = self.run_tool("new", stdin=json.dumps(bad))
        self.assertEqual(code, 2)
        text = " ".join(res["errors"])
        for needle in ("title 长 41", "effect.objects", "缺少 effect.visibility", "cause.objects.subsystem", "可点击链接"):
            self.assertIn(needle, text)
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "catalog", "entries"))), before)

    def test_thin_narrative_rejected(self):
        thin = good_entry(slug="thin", original=dict(good_entry()["original"], mechanism="硬编码索引", behavior="返回错",
                                                     source="https://github.com/o/r/issues/77"))
        code, res = self.run_tool("new", stdin=json.dumps(thin))
        self.assertEqual(code, 2)
        self.assertTrue(any(m.startswith("original.mechanism 只有") for m in res["errors"]))
        self.assertTrue(any(m.startswith("original.behavior 只有") for m in res["errors"]))

    def test_duplicate_source_and_var_target(self):
        dup = good_entry(slug="other", original=dict(good_entry()["original"], source="https://github.com/o/r/issues/1"))
        code, res = self.run_tool("new", stdin=json.dumps(dup))
        self.assertEqual(code, 2)
        self.assertIn(f"{self.ns}-0008", res["errors"][0])
        code, _ = self.run_tool("new", "--allow-duplicate-source", stdin=json.dumps(dup))
        self.assertEqual(code, 0)
        var = good_entry(slug="variant", labels=dict(good_entry()["labels"], coverage="var"), relations=[{"kind": "var", "target": "ZZZ-9999"}])
        code, res = self.run_tool("new", stdin=json.dumps(var))
        self.assertEqual(code, 2)
        self.assertTrue(any("var 目标" in m for m in res["errors"]))

    def test_relations_accept_old_numbers_and_echo_titles(self):
        seed, other_ns = f"{self.ns}-0008", "TRN" if self.ns == "INF" else "INF"
        var = good_entry(slug="variant", labels=dict(good_entry()["labels"], coverage="var"),
                         original=dict(good_entry()["original"], source="https://github.com/o/r/issues/5"),
                         relations=[{"kind": "var", "target": "1.8"}])            # old number of the seed
        code, res = self.run_tool("new", stdin=json.dumps(var))
        self.assertEqual(code, 0, res)
        vid = res["id"]
        self.assertEqual(self.entry(vid)["relations"], [{"kind": "var", "target": seed}])
        self.assertEqual(res["relations"], [{"kind": "var", "target": seed, "title": "topk 单核路径忽略传入索引",
                                             "status": "pending_audit", "from": "1.8"}])   # non-active status shown

        code, res = self.run_tool("update", vid, "--relation", "var:1.7")           # registered, but no entry file
        self.assertEqual(code, 2)
        self.assertTrue(any("var 目标 1.7" in m for m in res["errors"]))
        code, res = self.run_tool("update", vid, "--relation", f"var:{vid}")
        self.assertEqual(code, 2)
        self.assertTrue(any("指向条目自己" in m for m in res["errors"]))
        code, res = self.run_tool("update", vid, "--drop-relation", "var:1.8")       # would leave coverage=var bare
        self.assertEqual(code, 2)
        self.assertTrue(any("至少要保留一个 var" in m for m in res["errors"]))
        self.assertEqual(self.entry(vid)["relations"], [{"kind": "var", "target": seed}])   # unchanged on error

        code, res = self.run_tool("update", vid, "--relation", f"cross:{other_ns}-0001")
        self.assertEqual(code, 0, res)                                              # other catalog: kept as written
        self.assertEqual(res["changed"], ["relations"])
        self.assertEqual(res["relations"][1]["target"], f"{other_ns}-0001")
        code, res = self.run_tool("update", vid, "--drop-relation", f"cross:{other_ns}-0001")
        self.assertEqual(code, 0, res)
        self.assertEqual(self.entry(vid)["relations"], [{"kind": "var", "target": seed}])
        code, res = self.run_tool("update", vid, "--set", 'relations=[{"kind": "var", "target": "ZZZ-1"}]')
        self.assertEqual(code, 2)                                                   # --set goes through the same check

    def test_behavior_needs_80_chars_of_observed_symptom(self):
        short = good_entry(slug="short_behavior", original=dict(good_entry()["original"], behavior="单核形状下静默返回 iota 而非传入索引，无报错，下游取到错误的行。",
                                                               source="https://github.com/o/r/issues/78"))
        code, res = self.run_tool("new", stdin=json.dumps(short))
        self.assertEqual(code, 2)
        self.assertTrue(any(m.startswith("original.behavior 只有") and "使用者能直接观察到的现象" in m for m in res["errors"]))

    def test_gitcode_evidence_kinds(self):
        src = ("https://gitcode.com/cann/ops-nn/issues/6482 修复 https://gitcode.com/cann/ops-nn/merge_requests/11564 "
               "https://atomgit.com/mindspore/mindformers/merge_requests/8868 https://gitcode.com/cann/ops-nn/pull/12")
        code, res = self.run_tool("new", stdin=json.dumps(good_entry(slug="npu", original=dict(good_entry()["original"], source=src))))
        self.assertEqual(code, 0, res)
        self.assertEqual([x["kind"] for x in self.entry(res["id"])["evidence"]],
                         ["gitcode_issue", "gitcode_pr", "gitcode_pr", "gitcode_pr"])

    def test_cross_links_check_the_other_catalog_and_sync_back(self):
        other_ns = "TRN" if self.ns == "INF" else "INF"
        parent = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, parent)
        roots = {}
        for name, ns, sib in (("a", self.ns, "../b"), ("b", other_ns, "../a")):
            root = roots[name] = os.path.join(parent, name)
            os.makedirs(os.path.join(root, "tools"))
            os.makedirs(os.path.join(root, "catalog", "entries"))
            shutil.copy(os.path.join(HERE, "catalog_tool.py"), os.path.join(root, "tools"))
            shutil.copy(os.path.join(REPO, "catalog", "vocab.yaml"), os.path.join(root, "catalog"))
            cfg = dict(yaml.safe_load(open(os.path.join(REPO, "catalog", "config.yaml"))), namespace=ns, sibling=sib)
            with open(os.path.join(root, "catalog", "config.yaml"), "w") as fh:
                yaml.safe_dump(cfg, fh, allow_unicode=True)
            with open(os.path.join(root, "catalog", "id_registry.tsv"), "w") as fh:
                fh.write("new_id\tlegacy_id\n")

        def tool(name, *args, stdin=None):
            p = subprocess.run([sys.executable, os.path.join(roots[name], "tools", "catalog_tool.py"), *args],
                               input=stdin, capture_output=True, text=True)
            return p.returncode, json.loads(p.stdout)

        code, res = tool("b", "new", stdin=json.dumps(good_entry(slug="b_side")))
        self.assertEqual(code, 0, res)
        bid = res["id"]
        code, res = tool("a", "new", stdin=json.dumps(good_entry(slug="a_side", relations=[{"kind": "cross", "target": f"{other_ns}-0999"}])))
        self.assertEqual(code, 2)                                                    # unknown in the other catalog
        self.assertTrue(any("在另一编目" in m for m in res["errors"]))
        code, res = tool("a", "new", stdin=json.dumps(good_entry(slug="a_side", relations=[{"kind": "cross", "target": bid}])))
        self.assertEqual(code, 0, res)
        aid = res["id"]
        self.assertEqual(res["relations"], [{"kind": "cross", "target": bid, "catalog": other_ns,
                                             "title": "topk 单核路径忽略传入索引", "status": "pending_audit"}])
        code, res = tool("b", "cross-sync", "--dry-run")
        self.assertEqual([x["id"] for x in res["added"]], [bid])
        self.assertEqual(yaml.safe_load(open(os.path.join(roots["b"], "catalog", "entries", f"{bid}.yaml")))["relations"], [])
        code, res = tool("b", "cross-sync")
        self.assertEqual((code, res["added"][0]["cross"]), (0, aid))
        self.assertEqual(yaml.safe_load(open(os.path.join(roots["b"], "catalog", "entries", f"{bid}.yaml")))["relations"],
                         [{"kind": "cross", "target": aid}])
        _, res = tool("b", "cross-sync")
        self.assertEqual(res["added"], [])                                            # idempotent

    def test_duplicate_detected_from_unlinked_mention(self):
        # an existing entry cites "vLLM #60342/#60421" in its narrative without links
        old = good_entry(slug="parser_audit", original=dict(good_entry()["original"],
                         source="vLLM #60342/#60343/#60421 三个 parser 审计 https://github.com/vllm-project/vllm/issues/60342"))
        code, _ = self.run_tool("new", stdin=json.dumps(old))
        self.assertEqual(code, 0)
        dup = good_entry(slug="olmo3_parser", original=dict(good_entry()["original"],
                         source="https://github.com/vllm-project/vllm/issues/60421"))
        code, res = self.run_tool("new", stdin=json.dumps(dup))
        self.assertEqual(code, 2)
        self.assertIn("来源已被已有条目引用", res["errors"][0])
        _, res = self.run_tool("find", "vllm-project/vllm#60421")
        self.assertEqual(len(res["hits"]), 1)

    def test_update_find_and_validate(self):
        sid = f"{self.ns}-0008"
        code, res = self.run_tool("update", sid, "--add-evidence", "https://github.com/o/r/pull/2", "--note", "回访：已修复",
                                  "--set", 'mitigation.fix="PR #2 已合入"')
        self.assertEqual(code, 0, res)
        e = self.entry(sid)
        self.assertEqual(len(e["evidence"]), 2)
        self.assertTrue(e["original"]["notes"][0].endswith("回访：已修复"))
        code, res = self.run_tool("update", sid, "--set", "effect.visibility=很严重")
        self.assertEqual(code, 2)
        self.assertEqual(self.entry(sid)["effect"]["visibility"], "静默错误")      # unchanged on error
        _, res = self.run_tool("find", "o/r#2")
        self.assertEqual([h["id"] for h in res["hits"]], [sid])
        _, res = self.run_tool("find", "1.8")
        self.assertEqual([h["id"] for h in res["hits"]], [sid])
        code, res = self.run_tool("validate")
        self.assertEqual(code, 1)
        self.assertTrue(any("views/" in m for m in res["errors"]))                 # views not built yet
        self.run_tool("build")
        code, res = self.run_tool("validate")
        self.assertEqual(code, 0, res)
        xlsx = os.path.join(self.tmp, "views", "fault_catalog.xlsx")
        if os.path.exists(xlsx):                                     # openpyxl installed
            first = open(xlsx, "rb").read()
            self.run_tool("build")
            self.assertEqual(open(xlsx, "rb").read(), first)          # byte-identical rebuilds
            os.remove(xlsx)
            code, res = self.run_tool("validate")
            self.assertTrue(any("fault_catalog.xlsx" in m for m in res["errors"]))


if __name__ == "__main__":
    unittest.main()
