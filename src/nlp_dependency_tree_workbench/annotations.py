"""分层标注模型：原文版本、分词方案、句法关系与下游抽取的对应规则。

各层身份均由内容哈希派生，偏移一律以规范化后的原文为准。
"""
from __future__ import annotations

import hashlib
import json
import re

from .text import TOKEN_RE

SENTENCE_END_RE = re.compile(r"[。！？；!?…]+")


def content_hash(*parts):
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def split_sentences(text):
    """按句末标点切句，返回带原文偏移的句子列表。"""
    sentences = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        if text[start:end].strip():
            sentences.append({"index": len(sentences), "start": start, "end": end, "text": text[start:end]})
        start = end
    if start < len(text) and text[start:].strip():
        sentences.append({"index": len(sentences), "start": start, "end": len(text), "text": text[start:]})
    return sentences


def default_tokens(text):
    """缺省分词：按字符/词正则切分，词项带原文偏移。"""
    return [
        {"text": match.group(0), "start": match.start(), "end": match.end()}
        for match in TOKEN_RE.finditer(text)
    ]


def coerce_tokens(text, tokens):
    """校验并规范化分词方案。

    tokens 为 [start, end] 或 {"text", "start", "end"} 列表；为 None 时使用缺省分词。
    词项必须落在原文范围内、与原文文字一致、互不重叠。
    """
    if tokens is None:
        tokens = default_tokens(text)
    coerced = []
    for item in tokens:
        if isinstance(item, dict):
            start, end = int(item["start"]), int(item["end"])
            expected = str(item.get("text", ""))
        else:
            start, end = int(item[0]), int(item[1])
            expected = ""
        if not 0 <= start < end <= len(text):
            raise ValueError(f"词项跨度 [{start},{end}) 超出原文范围")
        actual = text[start:end]
        if expected and expected != actual:
            raise ValueError(f"词项文本“{expected}”与原文 [{start},{end}) 处“{actual}”不一致")
        coerced.append({"text": actual, "start": start, "end": end})
    coerced.sort(key=lambda token: (token["start"], token["end"]))
    for previous, current in zip(coerced, coerced[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(
                f"词项跨度 [{previous['start']},{previous['end']}) 与 [{current['start']},{current['end']}) 重叠"
            )
    for index, token in enumerate(coerced):
        token["index"] = index
    return coerced


def make_text_version_id(text):
    return "tv-" + content_hash("text", text)


def make_segmentation_id(text_version_id, tokens, segmenter):
    spans = [[token["start"], token["end"], token["text"]] for token in tokens]
    return "seg-" + content_hash("segmentation", text_version_id, spans, segmenter)


def make_relation_id(segmentation_id, dependent, head, label, kind):
    dep = [dependent["start"], dependent["end"]]
    hd = None if head is None else [head["start"], head["end"]]
    return "rel-" + content_hash("relation", segmentation_id, dep, hd, label, kind)


def make_extraction_id(kind, relation_ids, payload):
    return "ext-" + content_hash("extraction", kind, sorted(relation_ids), payload)


def validate_origin(origin):
    """校验来源：auto 必须给 run_id，manual 必须给 actor。"""
    if not isinstance(origin, dict):
        raise TypeError("origin 必须是对象")
    kind = str(origin.get("kind", "")).strip()
    if kind not in ("auto", "manual"):
        raise ValueError("origin.kind 必须是 auto（自动分析）或 manual（人工校订）")
    actor = str(origin.get("actor", "")).strip()
    run_id = str(origin.get("run_id", "")).strip()
    if kind == "manual" and not actor:
        raise ValueError("人工校订必须提供 actor")
    if kind == "auto" and not run_id:
        raise ValueError("自动分析必须提供 run_id")
    return {"kind": kind, "actor": actor, "run_id": run_id}


def parse_span(value, role):
    if isinstance(value, dict):
        return int(value["start"]), int(value["end"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{role} 跨度格式无效：{value!r}")


def build_relation(segmentation, spec, origin, at, order):
    """把关系规格解析为绑定分词方案词项的关系记录；跨度必须命中方案中的词项。"""
    if not isinstance(spec, dict):
        raise ValueError("关系条目必须是对象")
    seg_id = segmentation["segmentation_id"]
    label = str(spec.get("label", "")).strip()
    if not label:
        raise ValueError("关系缺少 label")
    index = {(token["start"], token["end"]): token for token in segmentation["tokens"]}
    dep_span = parse_span(spec.get("dependent"), "dependent")
    dependent = index.get(dep_span)
    if dependent is None:
        raise ValueError(f"dependent 跨度 [{dep_span[0]},{dep_span[1]}) 不对应分词方案 {seg_id} 中的任何词项")
    head_value = spec.get("head")
    if head_value is None:
        head = None
    else:
        head_span = parse_span(head_value, "head")
        head = index.get(head_span)
        if head is None:
            raise ValueError(f"head 跨度 [{head_span[0]},{head_span[1]}) 不对应分词方案 {seg_id} 中的任何词项")
        if head_span == dep_span:
            raise ValueError("dependent 与 head 不能是同一词项")
    relation_id = make_relation_id(seg_id, dependent, head, label, origin["kind"])
    return {
        "relation_id": relation_id,
        "segmentation_id": seg_id,
        "dependent": dict(dependent),
        "head": dict(head) if head is not None else None,
        "label": label,
        "origin": {**origin, "at": at},
        "order": order,
    }
