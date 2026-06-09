# -*- coding: utf-8 -*-
"""
业务动作执行器：Router 子任务 → 槽位抽取 → Pydantic 校验 → Mock/真实 API。

在 app.chat 中于 LLM 生成前调用；若返回非 None，则直接以结构化结果回复商家。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import ValidationError

from business.apis import ApiMode, get_price_review_api, get_stock_api
from business.schemas import (
    AgentActionType,
    AgentStructuredOutput,
    PriceReviewRequest,
    StockOrderRequest,
)
from business.slot_extractor import extract_price_review_slots, extract_stock_slots

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "log" / "business_api"

# 子任务 → 可执行 API
STOCK_SUBTASKS = {"发起备货"}
PRICE_REVIEW_SUBTASKS = {
    "商品供货价太低，需谈价",
    "商品供货价太低，需涨价",
}


class BusinessActionExecutor:
    def __init__(
        self,
        api_mode: ApiMode = "mock",
        log_dir: Path | str | None = None,
        enable_file_log: bool = True,
    ):
        self.api_mode = api_mode
        self.stock_api = get_stock_api(api_mode)
        self.price_api = get_price_review_api(api_mode)
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.enable_file_log = enable_file_log
        if enable_file_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def try_execute(
        self,
        user_input: str,
        scene: str,
        subtask: str,
    ) -> Optional[AgentStructuredOutput]:
        """若命中可执行子任务且槽位足够，调 API 并返回；否则 None 走 LLM。"""
        stock_intent = subtask in STOCK_SUBTASKS or (
            scene == "inventory"
            and (
                any(k in user_input for k in ("备货", "下单", "下个单", "补货", "断货"))
                or re.search(r"备.{0,2}货", user_input)
            )
        )
        if stock_intent:
            return self._handle_stock(user_input, subtask)

        price_intent = (
            subtask in PRICE_REVIEW_SUBTASKS
            or self._looks_like_price_review(user_input)
            or any(
                k in user_input
                for k in ("核价", "核一下", "报价", "调价", "供货价")
            )
        )
        if price_intent and not stock_intent:
            return self._handle_price_review(user_input, subtask)

        return None

    @staticmethod
    def _looks_like_stock(text: str) -> bool:
        t = text.lower()
        return any(
            k in t
            for k in ("下个单", "备货", "补货", "下单", "件", "skc", "sku")
        ) and bool(re.search(r"\d", t))

    @staticmethod
    def _looks_like_price_review(text: str) -> bool:
        t = text.lower()
        return ("核价" in t or "申诉内容" in t or "报价" in t) and (
            "skc" in t or re.search(r"\d{6,}", t)
        )

    def _handle_stock(self, user_input: str, subtask: str) -> AgentStructuredOutput:
        slots = extract_stock_slots(user_input)
        if slots.missing:
            return self._clarify_stock(slots)

        try:
            req = StockOrderRequest(
                skc=slots.skc,
                quantity=slots.quantity,
                size=slots.size,
                warehouse_code=slots.warehouse_code,
                remark=slots.remark,
            )
        except ValidationError as e:
            return AgentStructuredOutput(
                action=AgentActionType.CLARIFY,
                user_message=f"备货信息校验未通过：{e.errors()[0]['msg']}，请补全 SKC 和数量。",
                missing_fields=[x["loc"][0] for x in e.errors()],
            )

        api_resp = self.stock_api.create_order(req)
        out = AgentStructuredOutput(
            action=AgentActionType.STOCK_ORDER,
            user_message=api_resp.message,
            payload=req.model_dump(),
            api_called=True,
            api_result=api_resp.model_dump(),
        )
        self._log_action(user_input, subtask, out)
        return out

    def _handle_price_review(
        self, user_input: str, subtask: str
    ) -> AgentStructuredOutput:
        slots = extract_price_review_slots(user_input)
        if slots.missing:
            return self._clarify_price_review(slots)

        try:
            req = PriceReviewRequest(
                skc=slots.skc,
                target_price=slots.target_price,
                currency=slots.currency,
                reason=slots.reason or "",
            )
        except ValidationError as e:
            return AgentStructuredOutput(
                action=AgentActionType.CLARIFY,
                user_message="核价字段不完整或格式不对，请按 SKC + 报价 + 理由（工艺/材质）发我。",
                missing_fields=slots.missing,
            )

        api_resp = self.price_api.submit_review(req)
        out = AgentStructuredOutput(
            action=AgentActionType.PRICE_REVIEW,
            user_message=api_resp.message,
            payload=req.model_dump(mode="json"),
            api_called=True,
            api_result=api_resp.model_dump(mode="json"),
        )
        self._log_action(user_input, subtask, out)
        return out

    @staticmethod
    def _clarify_stock(slots) -> AgentStructuredOutput:
        hints = []
        if "skc" in slots.missing:
            hints.append("SKC 编号")
        if "quantity" in slots.missing:
            hints.append("备货数量（件）")
        msg = f"帮你下备货单还差：{'、'.join(hints)}，例如：SKC12345678 L码10件"
        return AgentStructuredOutput(
            action=AgentActionType.CLARIFY,
            user_message=msg,
            missing_fields=slots.missing,
        )

    @staticmethod
    def _clarify_price_review(slots) -> AgentStructuredOutput:
        hints = []
        if "skc" in slots.missing:
            hints.append("SKC")
        if "target_price" in slots.missing:
            hints.append("目标价/报价")
        if "reason" in slots.missing:
            hints.append("核价理由（工艺、材质等）")
        msg = (
            f"核价申请还差：{'、'.join(hints)}。"
            "可参考：*需核价货品SKC：xxx *币种：人民币 *申诉内容：工艺+材质+报价xx元"
        )
        return AgentStructuredOutput(
            action=AgentActionType.CLARIFY,
            user_message=msg,
            missing_fields=slots.missing,
        )

    def _log_action(self, user_input: str, subtask: str, out: AgentStructuredOutput):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subtask": subtask,
            "user_input": user_input,
            "action": out.action.value,
            "api_called": out.api_called,
            "user_message": out.user_message,
            "payload": out.payload,
            "api_result": out.api_result,
            "api_mode": self.api_mode,
        }
        logger.info("[BusinessAPI] %s", json.dumps(record, ensure_ascii=False))
        if not self.enable_file_log:
            return
        path = self.log_dir / "business_api.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
