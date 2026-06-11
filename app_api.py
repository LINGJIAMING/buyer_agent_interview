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
)
from generator import build_messages, log_trace
from llm_providers import LlmApiClient, LlmApiConfig
from query_optimizer import QueryOptimizer
from router import route_query, detect_subtask
from retriever import Retriever
from rag.merged_retriever import MergedRetriever
from business.executor import BusinessActionExecutor
from business.schemas import AgentActionType

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

    def reset_history(self):
        self.history = []
        self.session_entities = {}
        if self.query_optimizer:
            self.query_optimizer.reset_session()

    def chat(self, user_input: str) -> str:
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
                return reply

        retrieved_result = self.retriever.retrieve_context(working_query, scene=scene)
        retrieved_context = retrieved_result["context"]
        strong_hit = retrieved_result["strong_hit"]
        low_confidence = retrieved_result.get("low_confidence", False)
        follow_up_question = retrieved_result.get("follow_up_question", "")
        rag_meta = {
            "retrieval_method": retrieved_result.get("retrieval_method"),
            "sop_chunk_ids": retrieved_result.get("sop_chunk_ids"),
        }

        messages = build_messages(
            user_input=working_query,
            scene=scene,
            history=self.history,
            max_history_rounds=MAX_HISTORY_ROUNDS,
            retrieved_context=retrieved_context,
            strong_hit=strong_hit,
            subtask=subtask,
            low_confidence=low_confidence,
            follow_up_question=follow_up_question,
        )

        if self.verbose:
            print(f"[Router] scene={scene} | subtask={subtask}")
            if retrieved_result.get("retrieval_method") == "merged_policy_sop":
                print(
                    f"[RAG] merged | sop_chunks={retrieved_result.get('sop_chunk_ids', [])}"
                )

        response = self.llm.chat(messages, retrieved_context=retrieved_context)

        if self.verbose:
            log_trace(user_input, scene, subtask, retrieved_context, response)

        self._append_history(user_input, response)
        self._log_turn(
            user_input,
            working_query,
            scene,
            subtask,
            response,
            "llm_api",
            extra={"model": self.llm_config.resolved_model(), **rag_meta},
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
