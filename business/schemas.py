# -*- coding: utf-8 -*-
"""业务 API 请求/响应 — Pydantic 约束，便于对接真实 HTTP 接口。"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Currency(str, Enum):
    CNY = "CNY"
    USD = "USD"


# ---------------------------------------------------------------------------
# 备货下单
# ---------------------------------------------------------------------------


class StockOrderRequest(BaseModel):
    """备货 API 入参：商家提供 SKC + 数量（可选尺码/仓库）。"""

    skc: str = Field(..., description="商品 SKC 编号", min_length=4, max_length=32)
    quantity: int = Field(..., description="备货数量（件）", gt=0, le=100_000)
    size: Optional[str] = Field(None, description="尺码，如 L / M / 42")
    warehouse_code: Optional[str] = Field(None, description="目标仓库编码")
    remark: Optional[str] = Field(None, max_length=500)

    @field_validator("skc", mode="before")
    @classmethod
    def normalize_skc(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if s.startswith("SKC:"):
            s = s[4:].strip()
        return s


class StockOrderResponse(BaseModel):
    success: bool
    order_id: Optional[str] = None
    status: Literal["submitted", "failed", "pending"] = "pending"
    message: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 核价申请
# ---------------------------------------------------------------------------


class PriceReviewRequest(BaseModel):
    """核价 API 入参：SKC + 目标价 + 币种 + 核价理由。"""

    skc: str = Field(..., description="需核价 SKC", min_length=4, max_length=32)
    target_price: Decimal = Field(..., description="商家期望供货价", gt=0)
    currency: Currency = Field(Currency.CNY, description="币种")
    reason: str = Field(
        ...,
        description="核价理由，如工艺/材质/成本说明",
        min_length=2,
        max_length=2000,
    )

    @field_validator("skc", mode="before")
    @classmethod
    def normalize_skc(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if s.startswith("SKC:"):
            s = s[4:].strip()
        return s

    @field_validator("target_price", mode="before")
    @classmethod
    def parse_price(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        s = str(v).strip().replace("人民币", "").replace("元", "").replace(",", "")
        m = __import__("re").search(r"(\d+(?:\.\d+)?)", s)
        if not m:
            raise ValueError("无法解析价格")
        return Decimal(m.group(1))


class PriceReviewResponse(BaseModel):
    success: bool
    review_id: Optional[str] = None
    status: Literal["submitted", "failed", "pending", "rejected"] = "pending"
    message: str = ""
    suggested_price: Optional[Decimal] = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent 统一动作信封（模型可输出 / 日志可落库）
# ---------------------------------------------------------------------------


class AgentActionType(str, Enum):
    NONE = "none"
    STOCK_ORDER = "stock_order"
    PRICE_REVIEW = "price_review"
    CLARIFY = "clarify"


class AgentStructuredOutput(BaseModel):
    """
    买手 Agent 结构化输出（对接业务 API 前的标准形态）。

    - action=clarify 时仅返回追问，不调 API
    - action=stock_order / price_review 时 payload 为对应 Request 的 dict
    """

    action: AgentActionType = AgentActionType.NONE
    user_message: str = Field(..., description="返回给商家的自然语言")
    payload: Optional[dict[str, Any]] = Field(
        None, description="通过 Pydantic 校验后的 API 请求体"
    )
    missing_fields: list[str] = Field(default_factory=list)
    api_called: bool = False
    api_result: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_payload_action_match(self):
        if self.action == AgentActionType.STOCK_ORDER and self.payload:
            StockOrderRequest.model_validate(self.payload)
        elif self.action == AgentActionType.PRICE_REVIEW and self.payload:
            PriceReviewRequest.model_validate(self.payload)
        return self


def stock_order_json_schema() -> dict:
    return StockOrderRequest.model_json_schema()


def price_review_json_schema() -> dict:
    return PriceReviewRequest.model_json_schema()
