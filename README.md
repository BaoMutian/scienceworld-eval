# ScienceWorld LLM Agent Evaluation

基于 ScienceWorld 基准的 LLM Agent 评测框架，支持 Baseline 和 ReasoningBank 经验库模式。

## 功能特点

- **30种科学任务**: 涵盖物态变化、测量、电学、分类、植物生长、化学混合等10个主题
- **ReAct Agent**: 使用 Think-Action 格式的 ReAct 风格 Agent
- **ReasoningBank 经验库**: 从成功/失败轨迹中提取可复用策略
- **断点续传**: 支持中断后继续评测
- **配置化设计**: YAML 配置文件，支持命令行覆盖
- **多种记忆模式**: baseline / retrieve_only / retrieve_and_extract

## 安装

```bash
# 1. 创建 conda 环境
conda create -n scienceworld python=3.10
conda activate scienceworld

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
export OPENROUTER_API_KEY="your-api-key"
# 或在 config/default.yaml 中设置
```

## 快速开始

```bash
# 运行默认配置
python run_eval.py

# 使用自定义配置
python run_eval.py --config config/my_config.yaml

# 指定任务和模型
python run_eval.py --model qwen/qwen3-8b --task-ids 4-1 4-2 --num-episodes 3

# 开启 Debug 模式
python run_eval.py --debug --task-ids 4-1 --num-episodes 1

# 使用经验库
python run_eval.py --memory-mode retrieve_and_extract
```

## 配置说明

### 配置文件结构

```yaml
llm:
  api_base_url: "https://openrouter.ai/api/v1"
  model: "qwen/qwen3-8b"
  temperature: 0.3
  max_tokens: 1024

test:
  num_episodes: 5           # 每个任务变体的 episode 数
  task_ids: null            # null 表示所有任务，或 ["1-1", "4-1"]
  split: "dev"              # train/dev/test
  max_steps: 50             # 每个 episode 最大步数
  simplifications: "easy"   # 环境简化预设

memory:
  enabled: true
  mode: "retrieve_and_extract"  # baseline | retrieve_only | retrieve_and_extract
  top_k: 1                      # 检索返回的记忆数量
  similarity_threshold: 0.5    # 相似度阈值
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--config, -c` | 配置文件路径 |
| `--model, -m` | 模型名称 |
| `--task-ids, -t` | 任务 ID 列表 (如 1-1 4-1 4-2) |
| `--num-episodes, -n` | 每个变体的 episode 数 |
| `--split, -s` | 数据划分 (train/dev/test) |
| `--memory-mode` | 记忆模式 (baseline/retrieve_only/retrieve_and_extract) |
| `--debug, -d` | 开启调试模式 |

## 任务列表

| ID | 任务名 | 主题 | 说明 |
|----|--------|------|------|
| 1-1 | boil | 物质 | 沸腾 |
| 1-2 | melt | 物质 | 融化 |
| 1-3 | freeze | 物质 | 冷冻 |
| 1-4 | change-the-state-of-matter-of | 物质 | 物态变化 |
| 2-1 | use-thermometer | 测量 | 使用温度计 |
| 2-2 | measure-melting-point-known-substance | 测量 | 测熔点(已知) |
| 2-3 | measure-melting-point-unknown-substance | 测量 | 测熔点(未知) |
| 3-1 | power-component | 电学 | 创建电路 |
| 3-2 | power-component-renewable-vs-nonrenewable-energy | 电学 | 可再生能源 |
| 3-3 | test-conductivity | 电学 | 测导电性(已知) |
| 3-4 | test-conductivity-of-unknown-substances | 电学 | 测导电性(未知) |
| 4-1 | find-living-thing | 分类 | 找生物 |
| 4-2 | find-non-living-thing | 分类 | 找非生物 |
| 4-3 | find-plant | 分类 | 找植物 |
| 4-4 | find-animal | 分类 | 找动物 |
| 5-1 | grow-plant | 生物 | 种植物 |
| 5-2 | grow-fruit | 生物 | 种果实 |
| 6-1 | chemistry-mix | 化学 | 通用混合 |
| 6-2 | chemistry-mix-paint-secondary-color | 化学 | 二次色 |
| 6-3 | chemistry-mix-paint-tertiary-color | 化学 | 三次色 |
| 7-1 | lifespan-longest-lived | 生物 | 最长寿命 |
| 7-2 | lifespan-shortest-lived | 生物 | 最短寿命 |
| 7-3 | lifespan-longest-lived-then-shortest-lived | 生物 | 寿命排序 |
| 8-1 | identify-life-stages-1 | 生物 | 植物生命周期 |
| 8-2 | identify-life-stages-2 | 生物 | 动物生命周期 |
| 9-1 | inclined-plane-determine-angle | 力学 | 斜面角度 |
| 9-2 | inclined-plane-friction-named-surfaces | 力学 | 已知表面摩擦 |
| 9-3 | inclined-plane-friction-unnamed-surfaces | 力学 | 未知表面摩擦 |
| 10-1 | mendelian-genetics-known-plant | 生物 | 已知遗传学 |
| 10-2 | mendelian-genetics-unknown-plant | 生物 | 未知遗传学 |

## 记忆模式说明

### baseline (基准模式)
不使用经验库，Agent 独立完成任务。

### retrieve_only (仅检索)
从经验库检索相关经验辅助决策，但不提取新经验。

### retrieve_and_extract (检索并提取)
- 从经验库检索相关经验
- 任务完成后从轨迹提取新经验存入库

## 输出文件

```
results/
├── {run_id}_checkpoint.json     # 断点续传文件
├── {run_id}_results.json        # 最新结果
├── {run_id}_{timestamp}_results.json  # 带时间戳的结果
└── {run_id}_debug.log           # 调试日志 (debug 模式)

memory_banks/
├── scienceworld_memories.jsonl  # 记忆库
└── scienceworld_embeddings.npy  # 嵌入向量
```

## 项目结构

```
scienceworld-eval/
├── run_eval.py              # 主入口
├── requirements.txt         # 依赖
├── config/
│   └── default.yaml         # 默认配置
├── src/
│   ├── config.py            # 配置管理
│   ├── environment.py       # ScienceWorld 环境封装
│   ├── agent.py             # ReAct Agent
│   ├── evaluator.py         # 评估器
│   ├── llm_client.py        # LLM 客户端
│   ├── logging_utils.py     # 日志工具
│   ├── utils.py             # 通用工具
│   ├── prompts/
│   │   ├── system.py        # 系统提示词
│   │   └── few_shot.py      # Few-shot 示例
│   └── memory/
│       ├── schemas.py       # 数据结构
│       ├── store.py         # 记忆存储
│       ├── retriever.py     # 记忆检索
│       ├── extractor.py     # 记忆提取
│       └── embeddings.py    # 嵌入模型
├── results/                 # 结果输出
└── memory_banks/            # 记忆库存储
```

## 参考

- [ScienceWorld GitHub](https://github.com/allenai/ScienceWorld)
- [ScienceWorld Paper (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.775/)

