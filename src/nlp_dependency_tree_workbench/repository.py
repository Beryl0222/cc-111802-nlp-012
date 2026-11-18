import json
from pathlib import Path


class RecordRepository:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(value, dict) or not isinstance(value.get("records"), list):
            raise ValueError("记录库格式无效")
        return value["records"]

    def replace(self, records):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class EventStore:
    """只增不改的事件存储：分层标注的全部版本都由事件序列回放得出，可复现历史。"""

    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(value, dict) or not isinstance(value.get("events"), list):
            raise ValueError("事件库格式无效")
        return value["events"]

    def append(self, events):
        events = list(events)
        if not events:
            return
        existing = self.read()
        sequence = len(existing)
        stamped = []
        for event in events:
            if "seq" in event:
                raise ValueError("事件不允许自带序号")
            sequence += 1
            stamped.append({"seq": sequence, **event})
        self._write(existing + stamped)

    def _write(self, events):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
