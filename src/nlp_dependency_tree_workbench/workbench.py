"""分层句法校订工作台。

原文版本、分词方案、句法关系、下游结构化抽取逐层绑定：
每层记录其上游的内容身份；上游修订后按词项跨度重新判效，
不再有效的下游结论进入过期区，不得继续参与报告或派生新结论；
人工校订优先于自动分析，自动重跑不得静默覆盖人工关系。
"""
from __future__ import annotations

from datetime import datetime, timezone

from .annotations import (
    build_relation,
    coerce_tokens,
    make_extraction_id,
    make_segmentation_id,
    make_text_version_id,
    parse_span,
    split_sentences,
    validate_origin,
)
from .text import normalize_text


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _span_problem(role, span, current_segmentation):
    overlapping = [
        token
        for token in current_segmentation["tokens"]
        if token["start"] < span["end"] and span["start"] < token["end"]
    ]
    now = "、".join(f"[{token['start']},{token['end']})“{token['text']}”" for token in overlapping)
    if not now:
        now = "无词项覆盖该区域"
    return (
        f"{role} 词项 [{span['start']},{span['end']})“{span['text']}” 在当前分词方案 "
        f"{current_segmentation['segmentation_id']} 中已不存在；该区域现为 {now}"
    )


def compute_view(state):
    """按当前原文版本与分词方案为全部关系、抽取判效。

    关系两端词项跨度在当前方案中仍存在则有效（可跨方案沿用）；
    同一 dependent 上人工校订优先，其次按提交先后新者胜出；
    抽取仅当其依据关系全部当前有效时才有效。
    """
    view = {
        "current": None,
        "valid_relations": [],
        "expired_relations": [],
        "valid_extractions": [],
        "expired_extractions": [],
        "status_by_id": {},
    }
    seg_id = state["current_segmentation"]
    if not seg_id:
        return view
    segmentation = state["segmentations"][seg_id]
    text_version = state["text_versions"][segmentation["text_version_id"]]
    view["current"] = {
        "text_version_id": text_version["text_version_id"],
        "text": text_version["text"],
        "segmentation_id": seg_id,
        "segmenter": segmentation["segmenter"],
    }
    span_index = {(token["start"], token["end"]): token for token in segmentation["tokens"]}

    candidates = []
    for relation in state["relations"].values():
        rel_segmentation = state["segmentations"][relation["segmentation_id"]]
        problems = []
        dep_token = span_index.get((relation["dependent"]["start"], relation["dependent"]["end"]))
        if dep_token is None:
            problems.append(_span_problem("dependent", relation["dependent"], segmentation))
        head_token = None
        if relation["head"] is not None:
            head_token = span_index.get((relation["head"]["start"], relation["head"]["end"]))
            if head_token is None:
                problems.append(_span_problem("head", relation["head"], segmentation))
        entry = {
            "relation_id": relation["relation_id"],
            "label": relation["label"],
            "dependent": relation["dependent"],
            "head": relation["head"],
            "origin": relation["origin"],
            "seq": relation["seq"],
            "segmentation_id": relation["segmentation_id"],
            "carried_forward": relation["segmentation_id"] != seg_id,
        }
        if problems:
            if rel_segmentation["text_version_id"] != text_version["text_version_id"]:
                problems.append(
                    f"原文已自 {rel_segmentation['text_version_id']} 修订为 {text_version['text_version_id']}"
                )
            entry["status"] = "stale"
            entry["reason"] = "；".join(problems)
            view["expired_relations"].append(entry)
        else:
            # 有效关系展示当前词项，使界面看到它仍描述同一段原文
            entry["dependent"] = dep_token
            entry["head"] = head_token
            candidates.append(entry)

    by_dependent = {}
    for entry in candidates:
        key = (entry["dependent"]["start"], entry["dependent"]["end"])
        by_dependent.setdefault(key, []).append(entry)
    for group in by_dependent.values():
        group.sort(key=lambda item: (0 if item["origin"]["kind"] == "manual" else 1, -item["seq"]))
        winner = group[0]
        winner["status"] = "valid"
        view["valid_relations"].append(winner)
        for loser in group[1:]:
            if winner["origin"]["kind"] == "manual" and loser["origin"]["kind"] == "auto":
                reason = f"已被人工校订 {winner['relation_id']} 取代；人工校订优先，自动重跑不得静默覆盖"
            elif winner["origin"]["kind"] == "manual":
                reason = f"已被较新的人工校订 {winner['relation_id']} 取代"
            else:
                reason = f"已被较新的自动分析 {winner['relation_id']}（run {winner['origin']['run_id']}）取代"
            loser["status"] = "superseded"
            loser["reason"] = reason
            view["expired_relations"].append(loser)

    view["valid_relations"].sort(key=lambda item: (item["dependent"]["start"], item["seq"]))
    view["expired_relations"].sort(key=lambda item: item["seq"])
    view["status_by_id"] = {
        entry["relation_id"]: entry for entry in view["valid_relations"] + view["expired_relations"]
    }

    for extraction in state["extractions"].values():
        problems = []
        for relation_id in extraction["relation_ids"]:
            relation = view["status_by_id"].get(relation_id)
            if relation is None:
                problems.append(f"依据关系 {relation_id} 不存在")
            elif relation["status"] != "valid":
                problems.append(f"依据关系 {relation_id} 已失效（{relation['reason']}）")
        entry = {
            "extraction_id": extraction["extraction_id"],
            "kind": extraction["kind"],
            "payload": extraction["payload"],
            "relation_ids": extraction["relation_ids"],
            "origin": extraction["origin"],
            "seq": extraction["seq"],
        }
        if problems:
            entry["status"] = "stale"
            entry["reason"] = "；".join(problems)
            view["expired_extractions"].append(entry)
        else:
            entry["status"] = "valid"
            view["valid_extractions"].append(entry)
    view["valid_extractions"].sort(key=lambda item: item["seq"])
    view["expired_extractions"].sort(key=lambda item: item["seq"])
    return view


def _sentence_index_for(sentences, offset):
    for sentence in sentences:
        if sentence["start"] <= offset < sentence["end"]:
            return sentence["index"]
    later = [sentence for sentence in sentences if sentence["start"] >= offset]
    if later:
        return later[0]["index"]
    if sentences:
        return sentences[-1]["index"]
    return None


def _summarize_event(event):
    kind = event["type"]
    if kind == "text_registered":
        version = event["text_version"]
        suffix = f"，记录 {event['record_id']}" if event.get("record_id") else ""
        return f"登记原文版本 {version['text_version_id']}（{len(version['text'])} 字，{len(version['sentences'])} 句{suffix}）"
    if kind == "segmentation_registered":
        seg = event["segmentation"]
        return (
            f"登记分词方案 {seg['segmentation_id']}（{len(seg['tokens'])} 个词项，"
            f"segmenter={seg['segmenter']}，基于原文 {seg['text_version_id']}）"
        )
    if kind == "relations_submitted":
        origin = event["origin"]
        who = f"人工 {origin['actor']}" if origin["kind"] == "manual" else f"自动 {origin['run_id']}"
        return f"提交 {len(event['relations'])} 条关系（{who}，基于分词方案 {event['segmentation_id']}）"
    if kind == "extraction_registered":
        extraction = event["extraction"]
        return f"登记抽取 {extraction['extraction_id']}（{extraction['kind']}，依据 {len(extraction['relation_ids'])} 条关系）"
    return kind


class AnnotationWorkbench:
    """分层标注的命令入口与查询入口；状态由事件库回放得出。"""

    def __init__(self, store, clock=None):
        self.store = store
        self.clock = clock or _utc_now

    # ---- 命令 ----

    def register_segmentation(self, text, tokens=None, *, segmenter="default", record_id="", at=None):
        """登记原文版本及其分词方案；两者内容寻址，重复登记为幂等空操作。"""
        at = self._at(at)
        normalized = normalize_text(text)
        if not normalized.strip():
            raise ValueError("原文不能为空")
        token_list = coerce_tokens(normalized, tokens)
        text_version_id = make_text_version_id(normalized)
        segmentation_id = make_segmentation_id(text_version_id, token_list, segmenter)
        state = self._state()
        events = []
        if text_version_id != state["current_text_version"]:
            events.append(
                {
                    "type": "text_registered",
                    "at": at,
                    "record_id": str(record_id).strip(),
                    "text_version": {
                        "text_version_id": text_version_id,
                        "text": normalized,
                        "sentences": split_sentences(normalized),
                    },
                }
            )
        if segmentation_id != state["current_segmentation"]:
            events.append(
                {
                    "type": "segmentation_registered",
                    "at": at,
                    "segmentation": {
                        "segmentation_id": segmentation_id,
                        "text_version_id": text_version_id,
                        "segmenter": str(segmenter).strip() or "default",
                        "tokens": token_list,
                    },
                }
            )
        self.store.append(events)
        return {
            "text_version_id": text_version_id,
            "segmentation_id": segmentation_id,
            "registered_events": len(events),
            "token_count": len(token_list),
            "sentence_count": len(split_sentences(normalized)),
        }

    def submit_relations(self, relations, origin, *, segmentation_id=None, at=None):
        """提交一批句法关系；每条关系的跨度必须命中所基于分词方案中的词项。"""
        at = self._at(at)
        origin = validate_origin(origin)
        state = self._state()
        seg_id = segmentation_id or state["current_segmentation"]
        if not seg_id or seg_id not in state["segmentations"]:
            raise ValueError("分词方案不存在，请先登记原文与分词方案")
        if not isinstance(relations, list) or not relations:
            raise ValueError("relations 必须是非空列表")
        segmentation = state["segmentations"][seg_id]
        built = [build_relation(segmentation, spec, origin, at, order) for order, spec in enumerate(relations)]
        created, seen = [], set()
        for relation in built:
            if relation["relation_id"] in seen:
                continue
            seen.add(relation["relation_id"])
            if relation["relation_id"] not in state["relations"]:
                created.append(relation)
        if created:
            self.store.append(
                [
                    {
                        "type": "relations_submitted",
                        "at": at,
                        "segmentation_id": seg_id,
                        "origin": origin,
                        "relations": created,
                    }
                ]
            )
        return {
            "segmentation_id": seg_id,
            "created": [relation["relation_id"] for relation in created],
            "existing": [
                relation["relation_id"] for relation in built if relation["relation_id"] in state["relations"]
            ],
        }

    def submit_extraction(self, kind, relation_refs, payload=None, origin=None, *, at=None):
        """登记下游结构化结果；依据关系必须全部当前有效，否则拒绝登记。"""
        at = self._at(at)
        origin = validate_origin(origin)
        kind = str(kind).strip()
        if not kind:
            raise ValueError("抽取缺少 kind")
        if not isinstance(relation_refs, list) or not relation_refs:
            raise ValueError("relation_refs 必须是非空列表")
        state = self._state()
        view = compute_view(state)
        relation_ids = [self._resolve_relation_ref(state, view, ref) for ref in relation_refs]
        extraction_id = make_extraction_id(kind, relation_ids, payload or {})
        if extraction_id in state["extractions"]:
            return {"extraction_id": extraction_id, "created": False, "relation_ids": relation_ids}
        record = {
            "extraction_id": extraction_id,
            "kind": kind,
            "relation_ids": relation_ids,
            "payload": payload or {},
            "origin": {**origin, "at": at},
        }
        self.store.append([{"type": "extraction_registered", "at": at, "extraction": record}])
        return {"extraction_id": extraction_id, "created": True, "relation_ids": relation_ids}

    # ---- 查询 ----

    def report(self, at_seq=None):
        """按句给出当前有效分析、过期内容及每条关系依据的词项；at_seq 可回放历史。"""
        state = self._state(at_seq)
        view = compute_view(state)
        summary = {
            "valid_relations": len(view["valid_relations"]),
            "expired_relations": len(view["expired_relations"]),
            "valid_extractions": len(view["valid_extractions"]),
            "expired_extractions": len(view["expired_extractions"]),
        }
        if view["current"] is None:
            return {
                "at_seq": at_seq,
                "event_count": state["event_count"],
                "current": None,
                "sentences": [],
                "unlocated": None,
                "summary": summary,
            }
        text_version = state["text_versions"][view["current"]["text_version_id"]]
        segmentation = state["segmentations"][view["current"]["segmentation_id"]]
        sentences = text_version["sentences"]
        buckets = {
            sentence["index"]: {
                "index": sentence["index"],
                "text": sentence["text"],
                "span": [sentence["start"], sentence["end"]],
                "tokens": [
                    token
                    for token in segmentation["tokens"]
                    if sentence["start"] <= token["start"] and token["end"] <= sentence["end"]
                ],
                "valid_relations": [],
                "expired_relations": [],
                "valid_extractions": [],
                "expired_extractions": [],
            }
            for sentence in sentences
        }
        unlocated = {"valid_relations": [], "expired_relations": [], "valid_extractions": [], "expired_extractions": []}

        def place(offset, entry, field):
            index = _sentence_index_for(sentences, offset)
            if index is None:
                unlocated[field].append(entry)
            else:
                buckets[index][field].append(entry)

        for entry in view["valid_relations"]:
            place(entry["dependent"]["start"], entry, "valid_relations")
        for entry in view["expired_relations"]:
            place(entry["dependent"]["start"], entry, "expired_relations")
        for entry in view["valid_extractions"] + view["expired_extractions"]:
            field = "valid_extractions" if entry["status"] == "valid" else "expired_extractions"
            basis = view["status_by_id"].get(entry["relation_ids"][0])
            if basis is None:
                unlocated[field].append(entry)
            else:
                place(basis["dependent"]["start"], entry, field)

        return {
            "at_seq": at_seq,
            "event_count": state["event_count"],
            "current": view["current"],
            "sentences": [buckets[index] for index in sorted(buckets)],
            "unlocated": unlocated if any(unlocated.values()) else None,
            "summary": summary,
        }

    def history(self):
        """列出全部事件（版本），供复现与审计。"""
        return [
            {"seq": event["seq"], "type": event["type"], "at": event["at"], "summary": _summarize_event(event)}
            for event in self.store.read()
        ]

    # ---- 内部 ----

    def _at(self, at):
        if at is None:
            return self.clock()
        datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        return str(at)

    def _state(self, at_seq=None):
        events = self.store.read()
        if at_seq is not None:
            if not 0 <= at_seq <= len(events):
                raise ValueError(f"事件序号 {at_seq} 超出范围（0..{len(events)}）")
            events = events[:at_seq]
        state = {
            "text_versions": {},
            "segmentations": {},
            "relations": {},
            "extractions": {},
            "current_text_version": None,
            "current_segmentation": None,
            "event_count": len(events),
        }
        for event in events:
            self._apply(state, event)
        return state

    @staticmethod
    def _apply(state, event):
        seq = event["seq"]
        kind = event["type"]
        if kind == "text_registered":
            version = event["text_version"]
            state["text_versions"][version["text_version_id"]] = version
            state["current_text_version"] = version["text_version_id"]
        elif kind == "segmentation_registered":
            segmentation = event["segmentation"]
            state["segmentations"][segmentation["segmentation_id"]] = segmentation
            state["current_segmentation"] = segmentation["segmentation_id"]
        elif kind == "relations_submitted":
            for order, relation in enumerate(event["relations"]):
                record = {**relation, "seq": seq, "order": order}
                state["relations"].setdefault(record["relation_id"], record)
        elif kind == "extraction_registered":
            record = {**event["extraction"], "seq": seq}
            state["extractions"].setdefault(record["extraction_id"], record)
        else:
            raise ValueError(f"未知事件类型 {kind}")

    def _resolve_relation_ref(self, state, view, ref):
        if isinstance(ref, str):
            if ref not in state["relations"]:
                raise ValueError(f"关系 {ref} 不存在")
            relation_id = ref
        else:
            relation_id = self._match_valid_relation(view, ref)
        relation = view["status_by_id"].get(relation_id)
        if relation is None or relation["status"] != "valid":
            reason = relation["reason"] if relation else "关系不存在"
            raise ValueError(f"关系 {relation_id} 当前无效（{reason}），禁止据此生成结构化结果")
        return relation_id

    @staticmethod
    def _match_valid_relation(view, ref):
        dep_span = parse_span(ref.get("dependent"), "dependent")
        label = str(ref.get("label", "")).strip()
        head_value = ref.get("head")
        matches = []
        for entry in view["valid_relations"]:
            if (entry["dependent"]["start"], entry["dependent"]["end"]) != dep_span:
                continue
            if label and entry["label"] != label:
                continue
            if head_value is not None:
                if entry["head"] is None:
                    continue
                if (entry["head"]["start"], entry["head"]["end"]) != parse_span(head_value, "head"):
                    continue
            matches.append(entry)
        if not matches:
            raise ValueError(f"没有与 {ref} 匹配的当前有效关系")
        if len(matches) > 1:
            raise ValueError(f"{ref} 匹配到多条当前有效关系，请改用关系 id")
        return matches[0]["relation_id"]
