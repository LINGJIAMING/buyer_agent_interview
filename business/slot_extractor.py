# -*- coding: utf-8 -*-
"""从商家原话中抽取备货 / 核价槽位（规则引擎，不依赖 LLM）。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from business.schemas import Currency

RE_SKC = re.compile(
    r"(?:SKC[:\s：]*|skc[:\s：]*|需核价货品SKC[：:\s]*)(<SKC_ID>|\d{6,})",
    re.I,
)
RE_SKU_SIZE_QTY = re.compile(
    r"(?:SKC|SKU|skc|sku)[:\s：]*([A-Za-z0-9_<>\d]+).*?"
    r"(?:([Xx]{0,2}[Ss]?[MmLl]{0,2}|\d{2})\s*码\s*)?(\d+)\s*件",
    re.I | re.S,
)
RE_QTY_SIMPLE = re.compile(
    r"(\d+)\s*件|下.*?(\d+)\s*件|备货\s*(\d+)|数量\s*[:：]?\s*(\d+)",
    re.I,
)
RE_SIZE = re.compile(
    r"\b(X{0,2}[SML]|XXL|XXXL|\d{2})\s*码|\b([XSLM]{1,3})\b",
    re.I,
)
RE_PRICE_TEMPLATE = re.compile(
    r"\*?申诉内容[：:]\s*(.+?)(?:\*|$)",
    re.I | re.S,
)
RE_PRICE_NUM = re.compile(
    r"报价\s*(\d+(?:\.\d+)?)\s*(?:人民币|元|RMB)?|"
    r"目标价[：:\s]*(\d+(?:\.\d+)?)|"
    r"价格[：:\s]*(\d+(?:\.\d+)?)",
    re.I,
)
RE_CURRENCY = re.compile(r"币种.*?(人民币|美元|CNY|USD)", re.I)


@dataclass
class StockSlots:
    skc: Optional[str] = None
    quantity: Optional[int] = None
    size: Optional[str] = None
    warehouse_code: Optional[str] = None
    remark: Optional[str] = None
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skc": self.skc,
            "quantity": self.quantity,
            "size": self.size,
            "warehouse_code": self.warehouse_code,
            "remark": self.remark,
        }


@dataclass
class PriceReviewSlots:
    skc: Optional[str] = None
    target_price: Optional[Decimal] = None
    currency: Currency = Currency.CNY
    reason: Optional[str] = None
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skc": self.skc,
            "target_price": str(self.target_price) if self.target_price else None,
            "currency": self.currency.value,
            "reason": self.reason,
        }


def _pick_skc(text: str) -> Optional[str]:
    m = RE_SKC.search(text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?<![:/\d])(\d{8,12})(?![:/\d])", text)
    return m2.group(1) if m2 else None


def extract_stock_slots(text: str) -> StockSlots:
    slots = StockSlots()
    t = text.strip()

    m = RE_SKU_SIZE_QTY.search(t)
    if m:
        slots.skc = m.group(1).strip()
        slots.size = (m.group(2) or "").strip().upper() or None
        if m.group(3) and str(m.group(3)).isdigit():
            slots.quantity = int(m.group(3))
    else:
        slots.skc = _pick_skc(t)
        for pat in RE_QTY_SIMPLE.finditer(t):
            for g in pat.groups():
                if g and g.isdigit():
                    slots.quantity = int(g)
                    break
        sm = RE_SIZE.search(t)
        if sm:
            slots.size = (sm.group(1) or sm.group(2) or "").upper()

    if not slots.skc:
        slots.missing.append("skc")
    if not slots.quantity:
        slots.missing.append("quantity")
    return slots


def extract_price_review_slots(text: str) -> PriceReviewSlots:
    slots = PriceReviewSlots()
    t = text.strip()

    slots.skc = _pick_skc(t)

    rm = RE_PRICE_TEMPLATE.search(t)
    if rm:
        slots.reason = rm.group(1).strip().rstrip("*").strip()
    elif "核价" in t or "申诉" in t:
        # 取「报价」前后作为理由片段
        if "工艺" in t or "材质" in t or "涤纶" in t:
            slots.reason = re.sub(r"\*+[^*]+\*+", " ", t)
            slots.reason = re.sub(r"\s+", " ", slots.reason).strip()[:500]

    for pat in RE_PRICE_NUM.finditer(t):
        for g in pat.groups():
            if g:
                slots.target_price = Decimal(g)
                break
        if slots.target_price:
            break

    cm = RE_CURRENCY.search(t)
    if cm:
        c = cm.group(1)
        slots.currency = Currency.USD if "美" in c or "USD" in c.upper() else Currency.CNY

    if not slots.skc:
        slots.missing.append("skc")
    if slots.target_price is None:
        slots.missing.append("target_price")
    if not slots.reason or len(slots.reason) < 2:
        slots.missing.append("reason")
    return slots
