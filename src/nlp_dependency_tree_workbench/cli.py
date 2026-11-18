import argparse
import json
import sys
from pathlib import Path

from .repository import EventStore, RecordRepository
from .workbench import AnnotationWorkbench
from .workflow import TextWorkflow


def main():
    parser = argparse.ArgumentParser(description="本地教研文本处理工具")
    parser.add_argument("action", choices=("ingest", "list", "segment", "relate", "extract", "report", "history"))
    parser.add_argument("path", nargs="?", default="fixtures/records.json")
    parser.add_argument("--state", default="var/records.json")
    parser.add_argument("--store", default="var/annotation_events.json")
    parser.add_argument("--at", type=int, default=None, help="report 回放至指定事件序号")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run(args):
    if args.action in ("ingest", "list"):
        workflow = TextWorkflow(RecordRepository(args.state))
        if args.action == "ingest":
            return workflow.ingest(json.loads(Path(args.path).read_text(encoding="utf-8")))
        return workflow.list_records()
    workbench = AnnotationWorkbench(EventStore(args.store))
    if args.action == "segment":
        spec = json.loads(Path(args.path).read_text(encoding="utf-8"))
        return workbench.register_segmentation(
            spec["text"],
            spec.get("tokens"),
            segmenter=spec.get("segmenter", "default"),
            record_id=spec.get("record_id", ""),
            at=spec.get("at"),
        )
    if args.action == "relate":
        spec = json.loads(Path(args.path).read_text(encoding="utf-8"))
        return workbench.submit_relations(
            spec["relations"], spec["origin"], segmentation_id=spec.get("segmentation_id"), at=spec.get("at")
        )
    if args.action == "extract":
        spec = json.loads(Path(args.path).read_text(encoding="utf-8"))
        items = spec.get("extractions", [spec] if "kind" in spec else [])
        return [
            workbench.submit_extraction(
                item["kind"], item["relation_refs"], item.get("payload", {}), item["origin"], at=item.get("at")
            )
            for item in items
        ]
    if args.action == "report":
        return workbench.report(at_seq=args.at)
    if args.action == "history":
        return workbench.history()
    raise ValueError(f"未知操作 {args.action}")


if __name__ == "__main__":
    main()
