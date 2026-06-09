# -*- coding: utf-8 -*-
"""
Query Optimizer — 买手 Agent 用户问句预处理层

在 Router / Retriever / Generator 之前，对商家输入做可观测、可回放的规则化改写。

能力矩阵（对应需求 + 扩展）：
  ① context_fusion      — 上下文补全（SPU/SKC 与问题分散时合并）
  ② denoise_correct     — 去噪 + 常见错别字纠正
  ③ term_align          — 跨平台术语 → Temu/全托管域内说法
  ④ structure_simplify — 长句压缩、多诉求切分标记
  ⑤ coreference_resolve — 指代消解（这个/这款/上面）
  ⑥ entity_normalize    — 实体 ID 格式统一（SPU:/SKC:/WB:）
  ⑦ intent_anchor       — 省略动作用语补锚（推进/申诉/核价）
  ⑧ placeholder_guard   — 训练/脱敏占位符原样保留
  ⑨ multi_intent_detect — 多意图并行任务标记（供下游澄清）
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "log" / "query_opt"

# 口语助词 / 噪声（保留「麻烦」「辛苦」等业务礼貌词）
FILLER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"嗯+|呃+|啊+|呀+|哈+|诶+|额+|"
    r"那个(?![A-Za-z0-9])|就是(?![A-Za-z0-9])|然后呢|你知道吧|"
    r"怎么说呢|说实话|讲真|反正|基本上"
    r")(?![A-Za-z0-9])",
    re.I,
)

NOISE_PATTERN = re.compile(
    r"\[未知消息\]|<\|redacted_im_end\>|@\S+",
    re.I,
)

# 常见错别字 / 同音字（买手群聊）
TYPO_MAP = {
    "审办": "审版",
    "审板": "审版",
    "限留": "限流",
    "高价限留": "高价限流",
    "比价款": "比价",
    "核价卡主": "核价卡住",
    "加站": "加站",
    "上架": "加站",  # Temu 域内优先「加站」；若原文含「仓库上架」则不替换
    "备贷": "备货",
    "退供": "退供",
    "欧代": "欧代",
    "土代": "土代",
    "特木": "TEMU",
    "temu": "TEMU",
}

# 跨平台 → Temu / 全托管术语对齐
TERM_ALIGN_MAP = [
    (r"直通车", "开广告"),
    (r"钻展", "站内广告"),
    (r"超级推荐", "开广告"),
    (r"淘宝客|淘客", "站外推广"),
    (r"天猫", "全托管"),
    (r"京东自营", "全托管"),
    (r"拼多多", "TEMU"),
    (r"listing", "商品链接"),
    (r"SKU(?![C:/：:\s])", "SKC"),  # 口语 SKU 多指 SKC
]

# 实体抽取
RE_SPU = re.compile(
    r"(?:SPU[:\s：]*|spu[:\s：]*)(<SPU_ID>|\d{6,})",
    re.I,
)
RE_SKC = re.compile(
    r"(?:SKC[:\s：]*|skc[:\s：]*)(<SKC_ID>|\d{6,})",
    re.I,
)
RE_WB = re.compile(
    r"(?:WB[:\s：]*|wb[:\s：]*|退货单[:\s：]*)(<WB_ID>|\d{6,})",
    re.I,
)
RE_BARE_ID = re.compile(r"(?<![:/\d])(\d{8,12})(?![:/\d])")

# 指代 / 缺 ID 的动作句
PRONOUN_PATTERN = re.compile(
    r"(这个|这款|这件|那个|那款|上面的|刚才的|这款品|这个品|这个价格|这款链接)",
    re.I,
)
ACTION_WITHOUT_ID = re.compile(
    r"(核价|申诉|加站|上架|推进|限流|谈价|调价|备货|开白|审版|寄样|下架|改图|改尺码)",
    re.I,
)

# 多意图连接词
MULTI_INTENT_SPLIT = re.compile(
    r"[+＋/／]|(?:另外|顺便|还有|以及|同时|并且)(?=[\u4e00-\u9fff])",
)

# 意图锚定：过短且缺动词
SHORT_VAGUE = re.compile(r"^[\u4e00-\u9fff\w\s:：，,。.]{0,12}$")

# 通用政策/流程问句（不强行绑历史 SPU）
GENERAL_QUESTION = re.compile(
    r"(怎么办|是什么|什么意思|有哪些|哪些类目|怎么开通|如何理解|规则|政策|介绍)",
    re.I,
)


@dataclass
class OptimizeStep:
    """单步优化记录（大厂日志可追溯）。"""

    step_id: str
    step_name: str
    before: str
    after: str
    detail: str = ""
    changed: bool = False


@dataclass
class QueryOptimizeResult:
    """优化结果包。"""

    trace_id: str
    raw_query: str
    optimized_query: str
    steps: list[OptimizeStep] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class QueryOptimizeLogger:
    """结构化优化日志：JSONL + 可选单次 trace 文件。"""

    def __init__(self, log_dir: Path | None = None, enabled: bool = True):
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.enabled = enabled
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.log_dir / "query_opt.jsonl" if self.enabled else None

    def write(self, result: QueryOptimizeResult):
        if not self.enabled or not self._jsonl:
            return
        record = {
            "trace_id": result.trace_id,
            "timestamp": result.timestamp,
            "raw_query": result.raw_query,
            "optimized_query": result.optimized_query,
            "entities": result.entities,
            "flags": result.flags,
            "latency_ms": result.latency_ms,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "changed": s.changed,
                    "detail": s.detail,
                    "before": s.before[:200],
                    "after": s.after[:200],
                }
                for s in result.steps
            ],
        }
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_trace_report(self, result: QueryOptimizeResult) -> Path | None:
        """单次请求的 Markdown 追踪（便于排查）。"""
        if not self.enabled:
            return None
        path = self.log_dir / f"trace_{result.trace_id[:8]}.md"
        lines = [
            f"# Query Optimize Trace `{result.trace_id}`",
            "",
            f"| 字段 | 值 |",
            f"| --- | --- |",
            f"| 时间 | {result.timestamp} |",
            f"| 耗时 | {result.latency_ms:.2f} ms |",
            f"| 标记 | {', '.join(result.flags) or '-'} |",
            f"| 实体 | `{json.dumps(result.entities, ensure_ascii=False)}` |",
            "",
            "## 输入 / 输出",
            "",
            f"**RAW**",
            f"```",
            result.raw_query,
            f"```",
            "",
            f"**OPTIMIZED**",
            f"```",
            result.optimized_query,
            f"```",
            "",
            "## 流水线步骤",
            "",
        ]
        for s in result.steps:
            flag = "✅" if s.changed else "⏭"
            lines.extend(
                [
                    f"### {flag} `{s.step_id}` {s.step_name}",
                    f"- {s.detail or '无变更'}",
                    f"- before: `{s.before[:120]}`",
                    f"- after: `{s.after[:120]}`",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


class QueryOptimizer:
    """
    用户 Query 优化器（规则引擎，零外部依赖，可离线运行）。

    Parameters
    ----------
    log_dir : 日志目录
    enable_file_log : 是否写 JSONL
    enable_trace_md : 是否在变更时额外写 trace md
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        enable_file_log: bool = True,
        enable_trace_md: bool = False,
    ):
        self.logger = QueryOptimizeLogger(log_dir, enabled=enable_file_log)
        self.enable_trace_md = enable_trace_md
        self._session_entities: dict[str, str] = {}

    def reset_session(self):
        self._session_entities.clear()

    def optimize(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        session_entities: dict[str, str] | None = None,
    ) -> QueryOptimizeResult:
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        raw = (query or "").strip()
        text = raw
        steps: list[OptimizeStep] = []
        flags: list[str] = []
        entities: dict[str, str] = dict(session_entities or self._session_entities)

        def run_step(step_id: str, name: str, fn, detail: str = ""):
            nonlocal text
            before = text
            text, step_detail = fn(text)
            changed = before != text
            steps.append(
                OptimizeStep(
                    step_id=step_id,
                    step_name=name,
                    before=before,
                    after=text,
                    detail=step_detail or detail,
                    changed=changed,
                )
            )

        # ⑧ 占位符保护：先提取再还原
        placeholders: list[str] = []

        def guard_placeholders(s: str) -> str:
            def _repl(m):
                placeholders.append(m.group(0))
                return f"__PH_{len(placeholders)-1}__"

            return re.sub(
                r"<[A-Z_]+>|__PH_\d+__",
                _repl,
                s,
            )

        def restore_placeholders(s: str) -> str:
            for i, ph in enumerate(placeholders):
                s = s.replace(f"__PH_{i}__", ph)
            return s

        text = guard_placeholders(text)

        # ⑨ 噪声剔除
        run_step("S09_noise_strip", "噪声剔除", self._strip_noise)

        # ② 去噪 + 纠错
        run_step("S02_denoise_correct", "去噪与错别字纠正", self._denoise_and_correct)

        # ③ 术语对齐
        run_step("S03_term_align", "电商术语对齐(Temu)", self._align_terms)

        # 从历史 + 当前句抽取实体
        merged_text = self._concat_history(history) + "\n" + text
        entities = self._extract_entities(merged_text, entities)

        # ⑤ 指代消解（轻量：指代替换为最近 SPU）
        run_step(
            "S05_coreference",
            "指代消解",
            lambda s: self._resolve_coreference(s, entities, flags),
        )

        # ① 上下文融合
        run_step(
            "S01_context_fusion",
            "上下文补全",
            lambda s: self._fuse_context(s, entities, flags),
        )

        # ④ 结构简化
        run_step("S04_structure", "结构转换与压缩", self._simplify_structure)

        # ⑥ 实体规范化
        run_step(
            "S06_entity_normalize",
            "实体格式规范化",
            lambda s: self._normalize_entities(s, entities),
        )

        # ⑦ 意图锚定
        run_step("S07_intent_anchor", "意图锚定", lambda s: self._anchor_intent(s, flags))

        # ⑨ 多意图检测
        self._detect_multi_intent(text, flags)

        text = restore_placeholders(text)
        text = re.sub(r"\s+", " ", text).strip()

        # 更新 session 实体
        if session_entities is not None:
            session_entities.update(entities)
        else:
            self._session_entities.update(entities)

        latency_ms = (time.perf_counter() - t0) * 1000
        result = QueryOptimizeResult(
            trace_id=trace_id,
            raw_query=raw,
            optimized_query=text,
            steps=steps,
            entities={k: v for k, v in entities.items() if v},
            flags=flags,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        self.logger.write(result)
        if self.enable_trace_md and (raw != text or flags):
            self.logger.write_trace_report(result)

        return result

    # ----- 步骤实现 -----

    @staticmethod
    def _strip_noise(text: str) -> tuple[str, str]:
        n = len(text)
        t = NOISE_PATTERN.sub(" ", text)
        t = re.sub(r"\s+", " ", t).strip()
        removed = n - len(t)
        return t, f"剔除 @提及/未知消息等，约 {removed} 字符" if removed else "无噪声"

    @staticmethod
    def _denoise_and_correct(text: str) -> tuple[str, str]:
        t = FILLER_PATTERN.sub(" ", text)
        fixes = []
        # 「仓库上架」不改成加站
        protected = []
        for m in re.finditer(r"仓库.{0,4}上架|催仓库", t):
            protected.append((m.start(), m.end(), m.group(0)))

        def in_protected(pos):
            return any(a <= pos < b for a, b, _ in protected)

        for wrong, right in TYPO_MAP.items():
            if wrong == "上架" and "仓库" in t:
                continue
            if wrong in t:
                t = t.replace(wrong, right)
                fixes.append(f"{wrong}→{right}")

        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"([，,])\s*([，,])+", r"\1", t)
        return t, "纠错: " + "; ".join(fixes) if fixes else "去口语助词"

    @staticmethod
    def _align_terms(text: str) -> tuple[str, str]:
        aligned = []
        t = text
        for pat, repl in TERM_ALIGN_MAP:
            if re.search(pat, t, re.I):
                t2 = re.sub(pat, repl, t, flags=re.I)
                if t2 != t:
                    aligned.append(f"{pat}→{repl}")
                    t = t2
        return t, "术语: " + "; ".join(aligned) if aligned else "无跨平台术语"

    @staticmethod
    def _concat_history(history: list[dict[str, str]] | None) -> str:
        if not history:
            return ""
        parts = []
        for m in history[-8:]:
            c = (m.get("content") or "").strip()
            if c:
                parts.append(c)
        return "\n".join(parts)

    @staticmethod
    def _extract_entities(text: str, base: dict[str, str]) -> dict[str, str]:
        ent = dict(base)
        for m in RE_SPU.finditer(text):
            ent["spu"] = m.group(1)
        for m in RE_SKC.finditer(text):
            ent["skc"] = m.group(1)
        for m in RE_WB.finditer(text):
            ent["wb"] = m.group(1)
        if "spu" not in ent:
            ids = RE_BARE_ID.findall(text)
            if ids:
                ent["spu"] = ids[-1]
        return ent

    @staticmethod
    def _resolve_coreference(
        text: str, entities: dict[str, str], flags: list[str]
    ) -> tuple[str, str]:
        if not PRONOUN_PATTERN.search(text):
            return text, "无指代词"
        spu = entities.get("spu")
        if not spu:
            flags.append("unresolved_coreference")
            return text, "有指代但未解析到 SPU"
        t = PRONOUN_PATTERN.sub(f"SPU:{spu}", text, count=1)
        return t, f"指代→SPU:{spu}"

    @staticmethod
    def _fuse_context(
        text: str, entities: dict[str, str], flags: list[str]
    ) -> tuple[str, str]:
        has_id = bool(RE_SPU.search(text) or RE_SKC.search(text) or RE_BARE_ID.search(text))
        needs_action = bool(ACTION_WITHOUT_ID.search(text))
        if has_id or not needs_action:
            return text, "已含实体或无需补全"

        # 通用问法且无指代 → 不绑历史 SPU（避免「审版怎么办」误绑上一句）
        if GENERAL_QUESTION.search(text) and not PRONOUN_PATTERN.search(text):
            return text, "通用问句跳过上下文补全"

        # 仅在有指代、或明确执行业务短句时补全
        explicit_exec = PRONOUN_PATTERN.search(text) or re.search(
            r"(帮我|麻烦|辛苦).{0,20}(推进|申诉|核价|加站|备货)",
            text,
        )
        if not explicit_exec and len(text) > 30:
            return text, "长句无指代，不自动补 SPU"

        spu = entities.get("spu")
        skc = entities.get("skc")
        if spu:
            prefix = f"SPU:{spu} "
            if not text.upper().startswith("SPU"):
                flags.append("context_fused_spu")
                return prefix + text, f"从历史补全 SPU:{spu}"
        if skc:
            flags.append("context_fused_skc")
            return f"SKC:{skc} " + text, f"从历史补全 SKC:{skc}"

        flags.append("missing_entity")
        return text, "动作明确但无可用历史 ID"

    @staticmethod
    def _simplify_structure(text: str) -> tuple[str, str]:
        parts = MULTI_INTENT_SPLIT.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            # 保留第一句为核心，其余用分号连接（检索友好）
            core = parts[0]
            rest = "；".join(parts[1:])
            merged = f"{core}；{rest}"
            if len(merged) < len(text):
                return merged, f"多诉求切分合并 {len(parts)} 段"
        if len(text) > 120:
            # 去掉冗余从句标记
            t = re.sub(r"(就是说|其实|然后我想问|我想问一下)", "", text)
            t = re.sub(r"\s+", " ", t).strip()
            return t, "长句压缩"
        return text, "结构已简洁"

    @staticmethod
    def _normalize_entities(text: str, entities: dict[str, str]) -> tuple[str, str]:
        t = text
        changes = []

        def norm_spu(m):
            return f"SPU:{m.group(1)}"

        t2 = re.sub(
            r"(?:SPU[:\s：]*|spu[:\s：]*)(<SPU_ID>|\d{6,})",
            norm_spu,
            t,
            flags=re.I,
        )
        if t2 != t:
            changes.append("SPU 格式统一")
            t = t2

        t2 = re.sub(
            r"(?:SKC[:\s：]*|skc[:\s：]*)(<SKC_ID>|\d{6,})",
            lambda m: f"SKC:{m.group(1)}",
            t,
            flags=re.I,
        )
        if t2 != t:
            changes.append("SKC 格式统一")
            t = t2

        return t, "; ".join(changes) if changes else "实体格式已规范"

    @staticmethod
    def _anchor_intent(text: str, flags: list[str]) -> tuple[str, str]:
        if SHORT_VAGUE.match(text) and not ACTION_WITHOUT_ID.search(text):
            flags.append("vague_short_query")
            return text, "过短模糊，建议下游澄清"

        # 有 ID 但极短且含「推进/处理」→ 补「帮我推进」
        if re.search(r"^(SPU|SKC)[:\s：\d<]", text, re.I) and len(text) < 25:
            if re.search(r"(推进|处理|看看|搞一下)", text):
                return text, "意图已含动作"

        if ACTION_WITHOUT_ID.search(text) and not re.search(
            r"(帮我|麻烦|辛苦|请)", text
        ):
            flags.append("intent_anchored")
            return "帮我" + text, "补充「帮我」执行语气"
        return text, "意图完整"

    @staticmethod
    def _detect_multi_intent(text: str, flags: list[str]):
        parts = MULTI_INTENT_SPLIT.split(text)
        parts = [p for p in parts if p.strip()]
        if len(parts) >= 2:
            flags.append("multi_intent")
        actions = set(ACTION_WITHOUT_ID.findall(text))
        if len(actions) >= 2:
            flags.append("multi_action")
            flags.append("multi_intent")


def optimize_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    **kwargs,
) -> QueryOptimizeResult:
    """模块级快捷入口。"""
    opt = QueryOptimizer(**kwargs)
    return opt.optimize(query, history=history)


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------

_DEMO_CASES = [
    {
        "query": "嗯那个 这个价格能核多少",
        "history": [
            {"role": "user", "content": "SPU:814473175 高价限流了"},
            {"role": "assistant", "content": "我先看下"},
        ],
    },
    {
        "query": "直通车怎么投比较好",
        "history": [],
    },
    {
        "query": "这款帮我申诉一下 不是同款",
        "history": [{"role": "user", "content": "3498535536 加绒裤"}],
    },
    {
        "query": "另外顺便把加站也推进下 还有限流申诉",
        "history": [{"role": "user", "content": "SPU:5890623888 欧区没上"}],
    },
    {
        "query": "审办不通过怎么办",
        "history": [],
    },
]


def _print_result(r: QueryOptimizeResult):
    print("-" * 60)
    print(f"trace={r.trace_id[:8]}  flags={r.flags}")
    print(f"RAW : {r.raw_query}")
    print(f"OPT : {r.optimized_query}")
    for s in r.steps:
        if s.changed:
            print(f"  [{s.step_id}] {s.step_name}: {s.detail}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Optimizer 演示 / 批测")
    parser.add_argument("--demo", action="store_true", help="运行内置样例")
    parser.add_argument("-q", "--query", type=str, help="单条 query")
    parser.add_argument("--trace-md", action="store_true", help="输出 trace md")
    args = parser.parse_args()

    optimizer = QueryOptimizer(enable_trace_md=args.trace_md)

    if args.query:
        r = optimizer.optimize(args.query)
        _print_result(r)
    elif args.demo:
        for case in _DEMO_CASES:
            optimizer.reset_session()
            r = optimizer.optimize(case["query"], history=case.get("history"))
            _print_result(r)
        print(f"\n日志目录: {DEFAULT_LOG_DIR / 'query_opt.jsonl'}")
    else:
        parser.print_help()
