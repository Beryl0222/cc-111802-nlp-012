# 句法关系校订台

句法关系校订台用于处理本地合成教研文本，保存来源版本、文本片段和处理记录，供校内复核使用。

## 环境

需要 Python 3.11 或更新版本。应用运行不依赖第三方包；运行测试和构建前，安装固定版本的开发工具：`python -m pip install -r requirements-dev.txt`。

## 运行

执行 `PYTHONPATH=src python -m nlp_dependency_tree_workbench.cli ingest fixtures/records.json` 导入演练记录，再执行 `PYTHONPATH=src python -m nlp_dependency_tree_workbench.cli list` 查看当前文本处理结果。运行数据保存在 `var/records.json`。

## 分层校订（原文—分词—句法—抽取）

句法样例的四层标注逐层绑定，每层记录其上游的内容身份，上游修订后按词项跨度重新判效：

- **原文版本**：规范化文本按内容哈希登记为 `tv-…`，偏移一律以规范化后的文本为准。
- **分词方案**：登记为 `seg-…` 并绑定原文版本；词项必须与原文文字一致、互不重叠。
- **句法关系**：登记为 `rel-…`，绑定分词方案中的词项跨度（提交时跨度对不上词项会被拒绝）；来源区分 `auto`（需 `run_id`）与 `manual`（需 `actor`）。
- **下游抽取**（施事、对象等结构化结果）：登记为 `ext-…`，绑定关系 id；仅当依据关系全部当前有效时才允许登记，过期依据会被拒绝。

判效规则在查询时计算，不存标志位：

- 上游修订后，关系两端词项跨度在当前分词方案中仍存在的沿用有效（报告中标注 `carried_forward`）；边界已变的进入过期区并注明当前词项，人工确认过的跨度也不例外。
- 同一 dependent 上人工校订优先于自动分析；自动重跑不得静默覆盖人工关系，被取代者保留在过期区并注明原因。
- 抽取依据的关系全部有效时抽取才有效；任一依据失效（边界变化或已被取代），抽取随之过期，不再出现在有效结论中。

演示流程（状态保存在 `var/annotation_events.json`，可用 `--store` 更改）：

```sh
CLI="python -m nlp_dependency_tree_workbench.cli"
PYTHONPATH=src $CLI segment fixtures/annotation/01_segment_v1.json    # 登记原文与分词方案 v1
PYTHONPATH=src $CLI relate  fixtures/annotation/02_relations_run1.json # 自动分析 run1
PYTHONPATH=src $CLI relate  fixtures/annotation/03_relations_manual.json # 人工校订：作业本→发给 dobj
PYTHONPATH=src $CLI extract fixtures/annotation/04_extractions.json    # 登记施事、对象抽取
PYTHONPATH=src $CLI report                                            # 按句查看有效分析与词项依据
PYTHONPATH=src $CLI segment fixtures/annotation/05_segment_v2.json    # 修订分词：作业本→作业/本
PYTHONPATH=src $CLI report                                            # 人工 dobj 与对象抽取已过期
PYTHONPATH=src $CLI extract fixtures/annotation/04_extractions.json   # 失败：过期依据禁止复用
PYTHONPATH=src $CLI relate  fixtures/annotation/06_relations_run2.json # 自动重跑 run2
PYTHONPATH=src $CLI report                                            # 旧自动关系被取代，人工关系仍过期
PYTHONPATH=src $CLI history                                           # 查看全部版本事件
PYTHONPATH=src $CLI report --at 4                                     # 回放前 4 个事件，复现历史报告
```

## 测试

执行 `python -m pytest -q` 检查文本规范化、记录处理、状态读取及本地持久化。
