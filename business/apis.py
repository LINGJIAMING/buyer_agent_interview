# -*- coding: utf-8 -*-
"""业务 API 抽象 + Mock 实现（真实 HTTP 接口就绪后替换）。"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Literal

from business.schemas import (
    PriceReviewRequest,
    PriceReviewResponse,
    StockOrderRequest,
    StockOrderResponse,
)

ApiMode = Literal["mock", "http"]


class StockOrderApiClient(ABC):
    @abstractmethod
    def create_order(self, req: StockOrderRequest) -> StockOrderResponse:
        ...


class PriceReviewApiClient(ABC):
    @abstractmethod
    def submit_review(self, req: PriceReviewRequest) -> PriceReviewResponse:
        ...


class MockStockOrderApi(StockOrderApiClient):
    """模拟备货下单 API。"""

    def create_order(self, req: StockOrderRequest) -> StockOrderResponse:
        order_id = f"STK{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"
        size_part = f" 尺码{req.size}" if req.size else ""
        msg = (
            f"备货单已提交：SKC {req.skc}{size_part} × {req.quantity} 件，"
            f"单号 {order_id}，预计 1 个工作日内审核。"
        )
        return StockOrderResponse(
            success=True,
            order_id=order_id,
            status="submitted",
            message=msg,
            raw={
                "order_id": order_id,
                "skc": req.skc,
                "quantity": req.quantity,
                "size": req.size,
                "warehouse_code": req.warehouse_code,
                "mock": True,
            },
        )


class MockPriceReviewApi(PriceReviewApiClient):
    """模拟核价申请 API。"""

    def submit_review(self, req: PriceReviewRequest) -> PriceReviewResponse:
        review_id = f"PRV{uuid.uuid4().hex[:10].upper()}"
        # Mock：简单规则给出参考价
        suggested = req.target_price
        if req.target_price > 50:
            suggested = req.target_price * Decimal("0.9")
        status = "submitted"
        msg = (
            f"核价申请已提交：SKC {req.skc}，申报价 {req.target_price}{req.currency.value}，"
            f"单号 {review_id}，系统参考价约 {suggested.quantize(Decimal('0.01'))}，请等待审核。"
        )
        return PriceReviewResponse(
            success=True,
            review_id=review_id,
            status=status,
            message=msg,
            suggested_price=suggested,
            raw={
                "review_id": review_id,
                "skc": req.skc,
                "target_price": str(req.target_price),
                "currency": req.currency.value,
                "reason": req.reason,
                "mock": True,
            },
        )


# 未来 HTTP 实现示例骨架：
# class HttpStockOrderApi(StockOrderApiClient):
#     def __init__(self, base_url: str, api_key: str): ...
#     def create_order(self, req: StockOrderRequest) -> StockOrderResponse:
#         resp = requests.post(f"{self.base_url}/v1/stock/orders", json=req.model_dump(), ...)
#         resp.raise_for_status()
#         return StockOrderResponse.model_validate(resp.json())


def get_stock_api(mode: ApiMode = "mock") -> StockOrderApiClient:
    if mode == "mock":
        return MockStockOrderApi()
    raise NotImplementedError("HTTP 备货 API 尚未配置，请设置 BUSINESS_API_MODE=mock")


def get_price_review_api(mode: ApiMode = "mock") -> PriceReviewApiClient:
    if mode == "mock":
        return MockPriceReviewApi()
    raise NotImplementedError("HTTP 核价 API 尚未配置，请设置 BUSINESS_API_MODE=mock")
