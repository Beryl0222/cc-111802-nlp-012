import json
import tempfile
import unittest
from pathlib import Path

from nlp_dependency_tree_workbench.annotations import coerce_tokens, split_sentences
from nlp_dependency_tree_workbench.repository import EventStore
from nlp_dependency_tree_workbench.workbench import AnnotationWorkbench

TEXT = "老师把作业本发给了每个学生。"
TOKENS_V1 = [[0, 2], [2, 3], [3, 6], [6, 8], [8, 9], [9, 11], [11, 13], [13, 14]]
# 修订后的分词方案：作业本 → 作业 / 本，其余词项跨度不变
TOKENS_V2 = [[0, 2], [2, 3], [3, 5], [5, 6], [6, 8], [8, 9], [9, 11], [11, 13], [13, 14]]

RUN1 = [
    {"dependent": [0, 2], "head": [6, 8], "label": "nsubj"},
    {"dependent": [2, 3], "head": [3, 6], "label": "case"},
    {"dependent": [3, 6], "head": [0, 2], "label": "nmod"},
    {"dependent": [6, 8], "head": None, "label": "ROOT"},
    {"dependent": [8, 9], "head": [6, 8], "label": "aux"},
    {"dependent": [9, 11], "head": [11, 13], "label": "det"},
    {"dependent": [11, 13], "head": [6, 8], "label": "iobj"},
    {"dependent": [13, 14], "head": [6, 8], "label": "punct"},
]
MANUAL_FIX = [{"dependent": [3, 6], "head": [6, 8], "label": "dobj"}]


def auto(run="run-1"):
    return {"kind": "auto", "run_id": run}


def manual(actor="教研员-李"):
    return {"kind": "manual", "actor": actor}


def flatten(report, field):
    return [entry for sentence in report["sentences"] for entry in sentence[field]]


def by_label(entries, label):
    matches = [entry for entry in entries if entry["label"] == label]
    assert len(matches) == 1, f"label={label} 命中 {len(matches)} 条"
    return matches[0]


class WorkbenchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EventStore(Path(self.tmp.name) / "events.json")
        self.workbench = AnnotationWorkbench(self.store)
        registered = self.workbench.register_segmentation(
            TEXT, TOKENS_V1, segmenter="manual-v1", record_id="lesson-ba-01", at="2026-11-18T09:10:00Z"
        )
        self.text_version_id = registered["text_version_id"]
        self.seg_v1 = registered["segmentation_id"]
        self.run1_ids = self.workbench.submit_relations(RUN1, auto("run-1"), at="2026-11-18T09:11:00Z")["created"]
        self.manual_ids = self.workbench.submit_relations(MANUAL_FIX, manual(), at="2026-11-18T09:20:00Z")["created"]

    def relation_id(self, label):
        report = self.workbench.report()
        for field in ("valid_relations", "expired_relations"):
            for entry in flatten(report, field):
                if entry["label"] == label:
                    return entry["relation_id"]
        raise AssertionError(f"找不到 label={label} 的关系")


class LayerLinkAndBasisTests(WorkbenchTestCase):
    def test_layers_are_linked_and_report_shows_token_basis(self):
        agent = self.workbench.submit_extraction(
            "施事",
            [{"dependent": [0, 2], "label": "nsubj"}],
            {"text": "老师", "span": [0, 2]},
            manual(),
            at="2026-11-18T09:25:00Z",
        )
        patient = self.workbench.submit_extraction(
            "对象",
            [{"dependent": [3, 6], "label": "dobj"}],
            {"text": "作业本", "span": [3, 6]},
            manual(),
            at="2026-11-18T09:26:00Z",
        )
        self.assertTrue(agent["created"] and patient["created"])

        report = self.workbench.report()
        self.assertEqual(report["current"]["text_version_id"], self.text_version_id)
        self.assertEqual(report["current"]["segmentation_id"], self.seg_v1)
        self.assertEqual(len(report["sentences"]), 1)
        sentence = report["sentences"][0]
        self.assertEqual(sentence["span"], [0, 14])
        self.assertEqual([token["text"] for token in sentence["tokens"]], ["老师", "把", "作业本", "发给", "了", "每个", "学生", "。"])

        valid = sentence["valid_relations"]
        # 人工 dobj 取代自动 nmod；每条有效关系都带当前词项依据
        dobj = by_label(valid, "dobj")
        self.assertEqual(dobj["dependent"], {"index": 2, "text": "作业本", "start": 3, "end": 6})
        self.assertEqual(dobj["head"], {"index": 3, "text": "发给", "start": 6, "end": 8})
        self.assertEqual(dobj["origin"]["kind"], "manual")
        self.assertEqual(dobj["segmentation_id"], self.seg_v1)
        self.assertFalse(dobj["carried_forward"])
        nsubj = by_label(valid, "nsubj")
        self.assertEqual(nsubj["dependent"]["text"], "老师")
        self.assertEqual(nsubj["head"]["text"], "发给")
        self.assertEqual(len(valid), 8)

        expired = sentence["expired_relations"]
        nmod = by_label(expired, "nmod")
        self.assertEqual(nmod["status"], "superseded")
        self.assertIn("人工校订", nmod["reason"])
        self.assertIn(dobj["relation_id"], nmod["reason"])

        self.assertEqual(len(sentence["valid_extractions"]), 2)
        kinds = {entry["kind"] for entry in sentence["valid_extractions"]}
        self.assertEqual(kinds, {"施事", "对象"})
        patient_entry = next(entry for entry in sentence["valid_extractions"] if entry["kind"] == "对象")
        self.assertEqual(patient_entry["relation_ids"], [dobj["relation_id"]])
        self.assertEqual(report["summary"]["expired_relations"], 1)
        self.assertEqual(report["summary"]["expired_extractions"], 0)

    def test_identical_resubmission_is_idempotent(self):
        again = self.workbench.submit_relations(RUN1, auto("run-1"), at="2026-11-18T09:30:00Z")
        self.assertEqual(again["created"], [])
        self.assertEqual(sorted(again["existing"]), sorted(self.run1_ids))
        events_before = len(self.store.read())
        extraction = self.workbench.submit_extraction(
            "施事", [{"dependent": [0, 2], "label": "nsubj"}], {"text": "老师"}, manual()
        )
        repeat = self.workbench.submit_extraction(
            "施事", [{"dependent": [0, 2], "label": "nsubj"}], {"text": "老师"}, manual()
        )
        self.assertTrue(extraction["created"])
        self.assertFalse(repeat["created"])
        self.assertEqual(extraction["extraction_id"], repeat["extraction_id"])
        self.assertEqual(len(self.store.read()), events_before + 1)


class ManualProtectionTests(WorkbenchTestCase):
    def test_auto_rerun_cannot_override_manual_correction(self):
        # 自动重跑对同一 dependent 给出新关系，也不得覆盖人工校订
        result = self.workbench.submit_relations(
            [{"dependent": [3, 6], "head": [11, 13], "label": "nmod"}],
            auto("run-2"),
            at="2026-11-18T09:40:00Z",
        )
        self.assertEqual(len(result["created"]), 1)
        report = self.workbench.report()
        valid = flatten(report, "valid_relations")
        dobj = by_label(valid, "dobj")
        self.assertEqual(dobj["origin"]["kind"], "manual")
        self.assertEqual(dobj["head"]["text"], "发给")
        expired = flatten(report, "expired_relations")
        blocked = [entry for entry in expired if entry["relation_id"] == result["created"][0]]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["status"], "superseded")
        self.assertIn("人工校订优先", blocked[0]["reason"])
        self.assertIn("自动重跑不得静默覆盖", blocked[0]["reason"])

    def test_auto_rerun_supersedes_earlier_auto_and_stales_its_extraction(self):
        nsubj_id = self.relation_id("nsubj")
        extraction = self.workbench.submit_extraction(
            "施事", [nsubj_id], {"text": "老师"}, auto("extract-1"), at="2026-11-18T09:30:00Z"
        )
        self.assertTrue(extraction["created"])
        rerun = self.workbench.submit_relations(
            [{"dependent": [0, 2], "head": [11, 13], "label": "nsubj"}],
            auto("run-2"),
            at="2026-11-18T09:41:00Z",
        )
        report = self.workbench.report()
        valid = flatten(report, "valid_relations")
        nsubj = by_label(valid, "nsubj")
        self.assertEqual(nsubj["relation_id"], rerun["created"][0])
        self.assertEqual(nsubj["head"]["text"], "学生")
        expired = flatten(report, "expired_relations")
        old = [entry for entry in expired if entry["relation_id"] == nsubj_id]
        self.assertEqual(old[0]["status"], "superseded")
        self.assertIn("较新的自动分析", old[0]["reason"])
        # 依据被取代关系的抽取随之过期，不得继续出现在有效结论中
        expired_ext = flatten(report, "expired_extractions")
        self.assertEqual([entry["extraction_id"] for entry in expired_ext], [extraction["extraction_id"]])
        self.assertIn(nsubj_id, expired_ext[0]["reason"])
        # 基于新的有效关系可以重新登记同名结论
        renewed = self.workbench.submit_extraction(
            "施事", [rerun["created"][0]], {"text": "老师"}, auto("extract-2"), at="2026-11-18T09:42:00Z"
        )
        self.assertTrue(renewed["created"])
        self.assertNotEqual(renewed["extraction_id"], extraction["extraction_id"])


class BoundaryChangeTests(WorkbenchTestCase):
    def setUp(self):
        super().setUp()
        self.agent_ext = self.workbench.submit_extraction(
            "施事", [{"dependent": [0, 2], "label": "nsubj"}], {"text": "老师"}, manual(), at="2026-11-18T09:25:00Z"
        )
        self.patient_ext = self.workbench.submit_extraction(
            "对象", [{"dependent": [3, 6], "label": "dobj"}], {"text": "作业本"}, manual(), at="2026-11-18T09:26:00Z"
        )
        registered = self.workbench.register_segmentation(
            TEXT, TOKENS_V2, segmenter="manual-v2", at="2026-11-18T09:40:00Z"
        )
        self.seg_v2 = registered["segmentation_id"]

    def test_boundary_change_stales_manual_relation_and_its_extraction(self):
        report = self.workbench.report()
        self.assertEqual(report["current"]["segmentation_id"], self.seg_v2)
        valid = flatten(report, "valid_relations")
        expired = flatten(report, "expired_relations")

        # 边界未变的词项关系沿用有效，并标注为跨方案沿用
        nsubj = by_label(valid, "nsubj")
        self.assertTrue(nsubj["carried_forward"])
        self.assertEqual(nsubj["dependent"]["text"], "老师")
        self.assertEqual(nsubj["segmentation_id"], self.seg_v1)

        # 人工确认过的“作业本”跨度边界已变，不能假装仍然有效
        stale_manual = [entry for entry in expired if entry["relation_id"] == self.manual_ids[0]]
        self.assertEqual(len(stale_manual), 1)
        entry = stale_manual[0]
        self.assertEqual(entry["status"], "stale")
        self.assertEqual(entry["origin"]["kind"], "manual")
        self.assertIn("[3,6)“作业本”", entry["reason"])
        self.assertIn("作业", entry["reason"])
        self.assertIn("本", entry["reason"])

        # head 落在被改边界上的自动关系同样过期
        case_entry = by_label(expired, "case")
        self.assertEqual(case_entry["status"], "stale")
        self.assertIn("head", case_entry["reason"])

        # 下游：施事仍有效，对象随依据关系过期
        valid_ext = flatten(report, "valid_extractions")
        self.assertEqual([entry["kind"] for entry in valid_ext], ["施事"])
        expired_ext = flatten(report, "expired_extractions")
        self.assertEqual([entry["kind"] for entry in expired_ext], ["对象"])
        self.assertIn(self.manual_ids[0], expired_ext[0]["reason"])

    def test_extraction_on_stale_relation_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.workbench.submit_extraction("对象", [self.manual_ids[0]], {"text": "作业本"}, manual())
        self.assertIn("禁止据此生成结构化结果", str(caught.exception))
        with self.assertRaises(ValueError):
            self.workbench.submit_extraction("对象", [{"dependent": [3, 6], "label": "dobj"}], {}, manual())
        with self.assertRaises(ValueError):
            self.workbench.submit_extraction("对象", ["rel-不存在的编号"], {}, manual())
        # 施事依据仍然有效，可以正常登记
        ok = self.workbench.submit_extraction("施事复核", [self.relation_id("nsubj")], {}, manual())
        self.assertTrue(ok["created"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EventStore(Path(self.tmp.name) / "events.json")
        self.workbench = AnnotationWorkbench(self.store)
        self.workbench.register_segmentation(TEXT, TOKENS_V1, segmenter="manual-v1")

    def test_relation_span_must_match_a_token(self):
        events_before = len(self.store.read())
        with self.assertRaises(ValueError) as caught:
            self.workbench.submit_relations([{"dependent": [3, 5], "head": [6, 8], "label": "x"}], auto())
        self.assertIn("不对应分词方案", str(caught.exception))
        with self.assertRaises(ValueError):
            self.workbench.submit_relations([{"dependent": [0, 2], "head": [0, 2], "label": "x"}], auto())
        with self.assertRaises(ValueError):
            self.workbench.submit_relations([{"dependent": [0, 2], "head": None, "label": ""}], auto())
        self.assertEqual(len(self.store.read()), events_before)

    def test_origin_rules(self):
        with self.assertRaises(ValueError):
            self.workbench.submit_relations([{"dependent": [0, 2], "head": None, "label": "ROOT"}], {"kind": "robot"})
        with self.assertRaises(ValueError):
            self.workbench.submit_relations([{"dependent": [0, 2], "head": None, "label": "ROOT"}], {"kind": "manual"})
        with self.assertRaises(ValueError):
            self.workbench.submit_relations([{"dependent": [0, 2], "head": None, "label": "ROOT"}], {"kind": "auto"})

    def test_token_scheme_validation(self):
        with self.assertRaises(ValueError):
            coerce_tokens(TEXT, [[0, 2], [1, 3]])
        with self.assertRaises(ValueError):
            coerce_tokens(TEXT, [[0, 99]])
        with self.assertRaises(ValueError):
            coerce_tokens(TEXT, [{"text": "学生", "start": 0, "end": 2}])
        tokens = coerce_tokens(TEXT, [[2, 3], [0, 2]])
        self.assertEqual([token["index"] for token in tokens], [0, 1])
        self.assertEqual(tokens[0]["text"], "老师")

    def test_default_tokens_and_sentence_split(self):
        result = self.workbench.register_segmentation("老师提问。学生回答。")
        self.assertEqual(result["token_count"], 10)
        self.assertEqual(result["sentence_count"], 2)
        sentences = split_sentences("老师提问。学生回答。")
        self.assertEqual([(s["start"], s["end"]) for s in sentences], [(0, 5), (5, 10)])

    def test_relations_grouped_per_sentence(self):
        workbench = AnnotationWorkbench(EventStore(Path(self.tmp.name) / "other.json"))
        workbench.register_segmentation(
            "老师提问。学生回答。", [[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10]], segmenter="manual-v1"
        )
        workbench.submit_relations(
            [
                {"dependent": [0, 2], "head": [2, 4], "label": "nsubj"},
                {"dependent": [5, 7], "head": [7, 9], "label": "nsubj"},
            ],
            auto(),
        )
        report = workbench.report()
        self.assertEqual(len(report["sentences"]), 2)
        first, second = report["sentences"]
        self.assertEqual(first["valid_relations"][0]["dependent"]["text"], "老师")
        self.assertEqual(second["valid_relations"][0]["dependent"]["text"], "学生")


class HistoryTests(WorkbenchTestCase):
    def test_history_reproduces_earlier_reports(self):
        report_a = self.workbench.report()
        self.workbench.submit_extraction(
            "施事", [{"dependent": [0, 2], "label": "nsubj"}], {"text": "老师"}, manual(), at="2026-11-18T09:25:00Z"
        )
        report_b = self.workbench.report()
        self.workbench.register_segmentation(TEXT, TOKENS_V2, segmenter="manual-v2", at="2026-11-18T09:40:00Z")
        report_c = self.workbench.report()
        self.assertNotEqual(report_a["current"]["segmentation_id"], report_c["current"]["segmentation_id"])

        history = self.workbench.history()
        self.assertEqual([event["seq"] for event in history], list(range(1, 7)))
        self.assertEqual(
            [event["type"] for event in history],
            ["text_registered", "segmentation_registered", "relations_submitted", "relations_submitted",
             "extraction_registered", "segmentation_registered"],
        )
        self.assertTrue(all(event["summary"] for event in history))

        # 回放前 4 个事件 == 当初的报道；回放前 5 个事件 == 登记抽取后的报道
        def normalized(report):
            return json.dumps({**report, "at_seq": None}, sort_keys=True)

        self.assertEqual(normalized(self.workbench.report(at_seq=4)), normalized(report_a))
        self.assertEqual(normalized(self.workbench.report(at_seq=5)), normalized(report_b))
        empty = self.workbench.report(at_seq=0)
        self.assertIsNone(empty["current"])
        self.assertEqual(empty["sentences"], [])
        with self.assertRaises(ValueError):
            self.workbench.report(at_seq=7)

    def test_state_survives_reload(self):
        self.workbench.submit_extraction(
            "施事", [{"dependent": [0, 2], "label": "nsubj"}], {"text": "老师"}, manual()
        )
        reloaded = AnnotationWorkbench(EventStore(self.store.path))
        self.assertEqual(
            json.dumps(reloaded.report(), sort_keys=True),
            json.dumps(self.workbench.report(), sort_keys=True),
        )
        self.assertEqual(reloaded.history(), self.workbench.history())


if __name__ == "__main__":
    unittest.main()
