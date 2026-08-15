# Prompt Journal

一个独立、自包含的 Prompt 记录框架，用于记录你发给 AI 的每一条 prompt。
数据以两种形式落盘，都在本目录内：

- `prompts.jsonl` — 机器可读的源数据（一行一个 JSON 对象，可追加、可脚本处理）。
- `prompts.md` — 人类可读的 Markdown 视图（每次 `add` 自动追加，`export` 可整体重建）。

## 使用

在仓库根目录运行：

```bash
# 记录一条 prompt
python -m promptlog add "帮我分析 NVDA 的估值" --tag finance --tag nvda

# 查看最近 20 条
python -m promptlog list --limit 20

# 从 prompts.jsonl 重建 prompts.md（两者不一致时用）
python -m promptlog export
```

`add` 支持 `--tag`（可重复）与 `--note`。示例：

```bash
python -m promptlog add "可以继续 phase2 吗" --tag stockmind --note "关于后续阶段的问题"
```

## 记录格式

每条记录（JSONL 一行）：

```json
{
  "id": "a1b2c3d4",
  "timestamp": "2026-08-15T16:30:00+08:00",
  "prompt": "帮我分析 NVDA 的估值",
  "tags": ["finance", "nvda"],
  "note": null,
  "meta": {}
}
```

Markdown 视图示例：

```markdown
## 2026-08-15 16:30

Tags: `finance`, `nvda`

> 帮我分析 NVDA 的估值
```

## 隐私提示

Prompt 里可能包含敏感信息。如果不想把 `prompts.jsonl` / `prompts.md` 提交到仓库，
可把它们加入 `.gitignore`。

