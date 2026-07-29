# -*- coding: utf-8 -*-
"""
买手 Agent — 远程大模型版（不加载本地 Qwen / LoRA）。

复用：QueryOptimizer、Router、Retriever、Business API、build_messages。
生成：OpenAI 兼容 API（DeepSeek 等）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    MAX_HISTORY_ROUNDS,
    POLICY_KB_PATH,
    ENABLE_QUERY_OPTIMIZER,
    QUERY_OPT_LOG_DIR,
    ENABLE_BUSINESS_API,
    BUSINESS_API_MODE,
    BUSINESS_API_LOG_DIR,
    ENABLE_SOP_RAG,
    RAG_BM25_ONLY,
    RAG_USE_RERANK,
    RAG_SOP_TOP_K,
    ENABLE_SKILL_SELECTOR,
    ENABLE_FAQ_LAYER,
    ENABLE_RESPONSE_CACHE,
    FAQ_PUBLISHED_PATH,
    FAQ_MIN_BM25_SCORE,
    CACHE_SEMANTIC_THRESHOLD,
    RESPONSE_CACHE_PATH,
)
from generator import build_messages, log_trace
from llm_providers import LlmApiClient, LlmApiConfig
from query_optimizer import QueryOptimizer
from router import route_query, detect_subtask
from retriever import Retriever
from rag.merged_retriever import MergedRetriever
from business.executor import BusinessActionExecutor
from business.schemas import AgentActionType
from skills import get_skill, select_skill
from faq.policy_faq import PolicyFaqIndex, format_policy_direct
from cache.response_cache import BuyerResponseCache

CHAT_LOG_DIR = Path(__file__).resolve().parent / "log" / "api_chat"


class BuyerAgentApiApp:
    """与 BuyerAgentApp 同链路，仅 LLM 推理改为 API。"""

    def __init__(self, llm_config: LlmApiConfig, *, verbose: bool = True):
        self.llm = LlmApiClient(llm_config)
        self.llm_config = llm_config
        self.verbose = verbose
        if ENABLE_SOP_RAG:
            self.retriever = MergedRetriever(
                POLICY_KB_PATH,
                sop_top_k=RAG_SOP_TOP_K,
                use_rerank=RAG_USE_RERANK,
                bm25_only=RAG_BM25_ONLY,
            )
        else:
            self.retriever = Retriever(POLICY_KB_PATH)
        self.history: list[dict[str, str]] = []
        self.session_entities: dict[str, str] = {}
        self.query_optimizer = (
            QueryOptimizer(log_dir=QUERY_OPT_LOG_DIR, enable_file_log=True)
            if ENABLE_QUERY_OPTIMIZER
            else None
        )
        self.business_executor = (
            BusinessActionExecutor(
                api_mode=BUSINESS_API_MODE,
                log_dir=BUSINESS_API_LOG_DIR,
                enable_file_log=True,
            )
            if ENABLE_BUSINESS_API
            else None
        )
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.last_turn_meta: dict = {}
        self.faq_index: PolicyFaqIndex | None = None
        self.response_cache: BuyerResponseCache | None = None
        if ENABLE_FAQ_LAYER:
            self.faq_index = PolicyFaqIndex(
                FAQ_PUBLISHED_PATH,
                min_score=FAQ_MIN_BM25_SCORE,
            )
        if ENABLE_RESPONSE_CACHE:
            self.response_cache = BuyerResponseCache(
                semantic_threshold=CACHE_SEMANTIC_THRESHOLD,
                persist_path=Path(RESPONSE_CACHE_PATH),
            )

    def _try_policy_fast_path(self, working_query: str) -> Optional[str]:
        """L1/L2 缓存 → FAQ（政策直出），跳过 LLM 与买手润色。"""
        if self.response_cache:
            cached = self.response_cache.lookup(working_query)
            if cached and cached.get("response"):
                self.last_turn_meta = {
                    "path": "cache_policy_direct",
                    "cache_level": cached.get("cache_level"),
                    "faq_id": cached.get("faq_id"),
                    "response_mode": "policy_direct",
                }
                if self.verbose:
                    print(
                        f"[Cache] L{cached.get('cache_level')} hit | "
                        f"faq_id={cached.get('faq_id', '')}"
                    )
                return str(cached["response"])

        if not self.faq_index or self.faq_index.count() == 0:
            return None

        hit = self.faq_index.match(working_query)
        if not hit:
            return None

        reply = format_policy_direct(hit.entry)
        if self.response_cache:
            self.response_cache.remember_policy_direct(
                working_query,
                reply,
                faq_id=hit.faq_id,
                source="faq",
            )
        self.last_turn_meta = {
            "path": "faq_direct",
            "faq_id": hit.faq_id,
            "faq_score": hit.score,
            "response_mode": "policy_direct",
        }
        if self.verbose:
            print(f"[FAQ] policy_direct | {hit.faq_id} score={hit.score:.1f}")
        return reply

    def reset_history(self):
        self.history = []
        self.session_entities = {}
        self.last_turn_meta = {}
        if self.query_optimizer:
            self.query_optimizer.reset_session()

    def chat(
        self,
        user_input: str,
        *,
        merchant_notes: str = "",
        max_history_rounds: Optional[int] = None,
    ) -> str:
        working_query = user_input
        if self.query_optimizer:
            opt = self.query_optimizer.optimize(
                user_input,
                history=self.history,
                session_entities=self.session_entities,
            )
            working_query = opt.optimized_query
            if self.verbose and (
                opt.raw_query != opt.optimized_query or opt.flags
            ):
                print(
                    f"[QueryOpt] {opt.raw_query[:50]} -> {opt.optimized_query[:50]} "
                    f"| {opt.flags}"
                )

        scene = route_query(working_query)
        subtask = detect_subtask(scene, working_query)

        fast = self._try_policy_fast_path(working_query)
        if fast is not None:
            self.last_turn_meta["scene"] = scene
            self.last_turn_meta["subtask"] = subtask
            self.last_turn_meta["skill_id"] = None
            self.last_turn_meta["skill_reason"] = ""
            self._append_history(user_input, fast)
            self._log_turn(
                user_input,
                working_query,
                scene,
                subtask,
                fast,
                self.last_turn_meta.get("path", "faq_direct"),
                extra={
                    "faq_id": self.last_turn_meta.get("faq_id"),
                    "cache_level": self.last_turn_meta.get("cache_level"),
                    "faq_score": self.last_turn_meta.get("faq_score"),
                    "response_mode": "policy_direct",
                },
            )
            return fast

        if self.business_executor:
            action_out = self.business_executor.try_execute(
                working_query, scene, subtask
            )
            if action_out and action_out.action in (
                AgentActionType.STOCK_ORDER,
                AgentActionType.PRICE_REVIEW,
                AgentActionType.CLARIFY,
            ):
                reply = action_out.user_message
                self._append_history(user_input, reply)
                if self.verbose:
                    print(
                        f"[BusinessAPI] {action_out.action.value} "
                        f"api_called={action_out.api_called}"
                    )
                self._log_turn(user_input, working_query, scene, subtask, reply, "business_api")
                self.last_turn_meta = {
                    "scene": scene,
                    "subtask": subtask,
                    "skill_id": None,
                    "skill_reason": "business_api",
                }
                return reply

        # Skill 选择：模型看注册表目录，决定是否启用分析模板
        skill_id = None
        skill_reason = ""
        skill_body = ""
        use_rag = True
        if ENABLE_SKILL_SELECTOR:
            selection = select_skill(self.llm, working_query)
            skill_id = selection.skill_id
            skill_reason = selection.reason
            if skill_id:
                entry = get_skill(skill_id)
                if entry:
                    skill_body = entry.body
                    use_rag = False  # 分析 Skill 与政策证据约束分流
                    if self.verbose:
                        print(
                            f"[Skill] selected={skill_id} | {skill_reason} | "
                            f"title={entry.title}"
                        )
                else:
                    skill_id = None
            elif self.verbose:
                print(f"[Skill] none | {skill_reason}")

        retrieved_context = ""
        strong_hit = False
        low_confidence = False
        follow_up_question = ""
        rag_meta: dict = {
            "retrieval_method": None,
            "sop_chunk_ids": None,
        }
        if use_rag:
            retrieved_result = self.retriever.retrieve_context(
                working_query, scene=scene
            )
            retrieved_context = retrieved_result["context"]
            strong_hit = retrieved_result["strong_hit"]
            low_confidence = retrieved_result.get("low_confidence", False)
            follow_up_question = retrieved_result.get("follow_up_question", "")
            rag_meta = {
                "retrieval_method": retrieved_result.get("retrieval_method"),
                "sop_chunk_ids": retrieved_result.get("sop_chunk_ids"),
            }

        rounds = (
            max_history_rounds
            if max_history_rounds is not None
            else MAX_HISTORY_ROUNDS
        )
        messages = build_messages(
            user_input=working_query,
            scene=scene,
            history=self.history,
            max_history_rounds=rounds,
            retrieved_context=retrieved_context,
            strong_hit=strong_hit,
            subtask=subtask,
            low_confidence=low_confidence,
            follow_up_question=follow_up_question,
            skill_body=skill_body,
            use_rag_evidence=use_rag,
            merchant_notes=merchant_notes,
        )

        if self.verbose:
            print(f"[Router] scene={scene} | subtask={subtask}")
            if use_rag and rag_meta.get("retrieval_method") == "merged_policy_sop":
                print(
                    f"[RAG] merged | sop_chunks={rag_meta.get('sop_chunk_ids', [])}"
                )

        response = self.llm.chat(
            messages,
            retrieved_context=retrieved_context if use_rag else "",
        )

        if self.verbose:
            log_trace(user_input, scene, subtask, retrieved_context, response)

        self._append_history(user_input, response)
        self.last_turn_meta = {
            "scene": scene,
            "subtask": subtask,
            "skill_id": skill_id,
            "skill_reason": skill_reason,
        }
        self._log_turn(
            user_input,
            working_query,
            scene,
            subtask,
            response,
            "llm_api_skill" if skill_id else "llm_api",
            extra={
                "model": self.llm_config.resolved_model(),
                "skill_id": skill_id,
                "skill_reason": skill_reason,
                **rag_meta,
            },
        )
        return response

    def _append_history(self, user_input: str, response: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

    def _log_turn(
        self,
        raw_input: str,
        working_query: str,
        scene: str,
        subtask: str,
        response: str,
        path: str,
        extra: Optional[dict] = None,
    ):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": self.llm_config.provider,
            "model": self.llm_config.resolved_model(),
            "path": path,
            "raw_input": raw_input,
            "working_query": working_query,
            "scene": scene,
            "subtask": subtask,
            "response": response[:500],
        }
        if extra:
            record.update(extra)
        log_file = CHAT_LOG_DIR / "api_chat.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_api_app(llm_config: LlmApiConfig, **kwargs) -> BuyerAgentApiApp:
    return BuyerAgentApiApp(llm_config, **kwargs)
