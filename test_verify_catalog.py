import re
import unittest
from pathlib import Path

import verify_catalog
from export_faults import parse_entries


class CatalogVerificationTest(unittest.TestCase):
    def setUp(self):
        self.text = Path(__file__).with_name("FAULT_MASTER_REFERENCE.zh.md").read_text()

    def test_rejects_wrong_rolling_total(self):
        match = re.search(r"(## 滚动汇总（共 )(\d+)( 条)", self.text)
        self.assertIsNotNone(match)
        broken = self.text[:match.start(2)] + str(int(match[2]) + 1) + self.text[match.end(2):]
        with self.assertRaisesRegex(ValueError, "滚动汇总"):
            verify_catalog.check(broken)

    def test_rejects_stale_directory_count(self):
        match = re.search(r"^(\|\| 1 \|[^\n]*\| )(\d+)( \|)", self.text, re.M)
        assert match is not None
        broken = self.text[:match.start(2)] + str(int(match[2]) + 1) + self.text[match.end(2):]
        with self.assertRaisesRegex(ValueError, "目录"):
            verify_catalog.check(broken)

    def test_parser_does_not_swallow_separator(self):
        snippet = (
            "### 1.999 `example`\n\n"
            "`infra_off` · Cov: `NEW` · 置信度 `documented`\n\n"
            "- **发现来源**：2026-09-23 每日扫描。\n\n---\n\n"
            "### 1.1000 `next`\n\n"
            "`infra_off` · Cov: `NEW` · 置信度 `documented`\n\n"
            "- **发现来源**：2026-09-23 每日扫描。\n"
        )
        self.assertEqual(parse_entries(snippet)[0]["discovery_date"], "2026-09-23 每日扫描。")

    def test_rejects_polluted_discovery_date(self):
        rows = [{"discovery_date": "2026-09-23 每日扫描\n\n---\n# DATA"}]
        with self.assertRaisesRegex(ValueError, "discovery_date"):
            verify_catalog.check_dates(rows)

    def test_topk_source_has_one_canonical_mechanism(self):
        rows = [row for row in parse_entries(self.text)
                if row["source"] == "https://github.com/tenstorrent/tt-metal/issues/51921"]
        self.assertEqual([row["id"] for row in rows], ["1.44"])
        self.assertIn("GENERATE_INDICES", rows[0]["mechanism"])
        self.assertIn("1.70 → 1.44", self.text)

    def test_dspark_source_has_one_evidence_based_mechanism(self):
        rows = [row for row in parse_entries(self.text)
                if row["source"] == "https://github.com/sgl-project/sglang/issues/33493"]
        self.assertEqual([row["id"] for row in rows], ["11.24"])
        self.assertIn('getattr(sampling_info, "acc_linear_penalties", None)', rows[0]["mechanism"])
        self.assertIn("acc_additive_penalties", rows[0]["mechanism"])
        self.assertIn("不是 `KeyError`", rows[0]["mechanism"])
        self.assertIn("min_new_tokens", rows[0]["behavior"])
        self.assertIn("11.30 → 11.24", self.text)


if __name__ == "__main__":
    unittest.main()
