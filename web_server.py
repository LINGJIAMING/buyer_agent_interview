# -*- coding: utf-8 -*-
"""
买手 Agent · 简易网页入口（类 assistant_agent）

功能：
  - GET /              聊天页（操作员 + 商家记忆）
  - POST /chat         非流式问答（落库 messages + 注入商家便签）
  - 操作员 Key：secrets/operator_keys.json（不进 LLM 记忆）
  - GET /tools/menswear  男装全品类 Prompt 工具页

启动（建议用装了 RAG 依赖的 Python，本机多为 3.9）：

  cd mybuyer_agent-main
  py -3.9 -m uvicorn web_server:app --reload --host 127.0.0.1 --port 8001

浏览器打开 http://127.0.0.1:8001

API Key：优先 secrets/operator_keys.json（按 operator_id）；否则 .env。
本文件不改 Router。
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory.keys import (
    list_operators,
    resolve_api_key_for_operator,
    save_operator_key,
)
from memory.store import MemoryStore

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_STATIC = PROJECT_ROOT / "web" / "static"
SKILL_STATIC = (
    PROJECT_ROOT / "skills" / "menswear_full_category" / "static"
)
PROMPTS_YAML = (
    PROJECT_ROOT / "skills" / "menswear_full_category" / "prompts.yaml"
)

# 按 operator 缓存 Agent（Key 不同）；会话 history 以 SQLite 为准
_agents: Dict[str, Any] = {}
_sessions: dict[str, dict[str, Any]] = {}
_store: Optional[MemoryStore] = None

MEMORY_MSG_LIMIT = 20
MEMORY_NOTE_LIMIT = 5
MEMORY_HISTORY_ROUNDS = 10


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _get_agent(operator_id: str = ""):
    op = (operator_id or "").strip() or "_env"
    if op in _agents:
        return _agents[op]

    from app_api import create_api_app
    from llm_providers import LlmApiConfig

    api_key, provider, source = resolve_api_key_for_operator(
        "" if op == "_env" else op
    )
    if not api_key:
        raise RuntimeError(
            "未配置 API Key。请在网页保存操作员 Key，"
            "或复制 secrets/operator_keys.example.json → operator_keys.json，"
            "或在 .env 填写 DEEPSEEK_API_KEY。"
        )
    cfg = LlmApiConfig(
        provider=provider or os.getenv("API_PROVIDER", "deepseek"),
        api_key=api_key,
        base_url=os.getenv("API_BASE_URL", ""),
        model=os.getenv("API_MODEL", "") or os.getenv("LLM_MODEL", ""),
        max_tokens=int(os.getenv("API_MAX_TOKENS", "512")),
        temperature=float(os.getenv("API_TEMPERATURE", "0.6")),
    )
    agent = create_api_app(cfg, verbose=False)
    # 仅调试用：不把 key 放进 meta；source 可暴露
    agent._key_source = source  # type: ignore[attr-defined]
    _agents[op] = agent
    return agent


def _session_bundle(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {"session_entities": {}}
    return _sessions[session_id]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_dotenv()
    WEB_STATIC.mkdir(parents=True, exist_ok=True)
    get_store()
    yield


app = FastAPI(
    title="买手 Agent Web",
    description="网页问答 + 商家记忆 + 操作员本地 Key（未改 Router）",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_STATIC)), name="static")
if SKILL_STATIC.exists():
    app.mount(
        "/tools/menswear/assets",
        StaticFiles(directory=str(SKILL_STATIC)),
        name="menswear_assets",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None  # 兼容旧字段 = session_id
    session_id: Optional[str] = None
    merchant_id: str = Field(..., min_length=1)
    operator_id: str = ""


class ChatResponse(BaseModel):
    thread_id: str
    session_id: str
    merchant_id: str
    operator_id: str
    reply: str
    scene: Optional[str] = None
    subtask: Optional[str] = None
    skill_id: Optional[str] = None
    skill_reason: Optional[str] = None
    key_source: Optional[str] = None
    response_path: Optional[str] = None


class OperatorKeyRequest(BaseModel):
    operator_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    provider: str = "deepseek"


class NoteRequest(BaseModel):
    merchant_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source: str = "manual"


@app.get("/")
async def index():
    index_path = WEB_STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "web/static/index.html 缺失")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    api_key, _, source = resolve_api_key_for_operator(None)
    store = get_store()
    return {
        "status": "ok",
        "api_key_configured": bool(api_key),
        "key_source": source if api_key else "missing",
        "memory": store.stats(),
        "operators": list_operators(),
        "menswear_html": (SKILL_STATIC / "全品类.html").exists(),
        "prompts_yaml": PROMPTS_YAML.exists(),
    }


@app.get("/api/operators")
async def api_list_operators():
    return {"operators": list_operators()}


@app.post("/api/operators/key")
async def api_save_operator_key(body: OperatorKeyRequest):
    try:
        meta = save_operator_key(
            body.operator_id,
            body.api_key,
            provider=body.provider,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # 清掉该操作员缓存的 Agent，下次用新 Key
    op = body.operator_id.strip()
    _agents.pop(op, None)
    return {"ok": True, "operator": meta}


@app.get("/api/sessions")
async def api_list_sessions(merchant_id: str):
    if not (merchant_id or "").strip():
        raise HTTPException(400, "merchant_id 必填")
    return {"sessions": get_store().list_sessions(merchant_id.strip())}


@app.get("/api/notes")
async def api_list_notes(merchant_id: str, limit: int = MEMORY_NOTE_LIMIT):
    if not (merchant_id or "").strip():
        raise HTTPException(400, "merchant_id 必填")
    return {
        "notes": get_store().list_notes(merchant_id.strip(), limit=max(1, min(limit, 50)))
    }


@app.post("/api/notes")
async def api_add_note(body: NoteRequest):
    try:
        note_id = get_store().add_note(
            body.merchant_id,
            body.content,
            source=body.source or "manual",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "id": note_id}


@app.get("/tools/menswear")
async def menswear_tool_page():
    html_path = SKILL_STATIC / "全品类.html"
    if not html_path.exists():
        raise HTTPException(404, "全品类.html 未找到，请先运行 skills/_extract_menswear_prompts.py")
    return FileResponse(html_path)


@app.get("/api/skills/menswear/prompts")
async def list_menswear_prompts():
    if not PROMPTS_YAML.exists():
        raise HTTPException(404, "prompts.yaml 不存在")
    try:
        import yaml
    except ImportError as exc:
        raise HTTPException(500, "请 pip install pyyaml") from exc
    data = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts") or []
    summary = [
        {
            "id": p.get("id"),
            "title": p.get("title"),
            "category": p.get("category"),
            "needs_image": p.get("needs_image", False),
            "body_preview": (p.get("body") or "")[:160],
        }
        for p in prompts
    ]
    return {"skill": data.get("skill"), "count": len(summary), "prompts": summary}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    if len(message) > 4000:
        raise HTTPException(400, "消息过长")

    merchant_id = body.merchant_id.strip()
    operator_id = (body.operator_id or "").strip()
    session_id = (body.session_id or body.thread_id or "").strip() or None

    store = get_store()
    try:
        session_id = store.ensure_session(
            merchant_id=merchant_id,
            operator_id=operator_id,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        agent = _get_agent(operator_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    # 从 SQLite 恢复对话；entities 仍按 session 放内存
    bundle = _session_bundle(session_id)
    agent.history = store.get_recent_messages(session_id, limit=MEMORY_MSG_LIMIT)
    agent.session_entities = bundle["session_entities"]
    notes_block = store.notes_as_prompt_block(
        merchant_id, limit=MEMORY_NOTE_LIMIT
    )

    try:
        from router import route_query, detect_subtask

        scene = route_query(message)
        subtask = detect_subtask(scene, message)
        reply = agent.chat(
            message,
            merchant_notes=notes_block,
            max_history_rounds=MEMORY_HISTORY_ROUNDS,
        )
        meta = getattr(agent, "last_turn_meta", {}) or {}
    except Exception as exc:
        raise HTTPException(500, f"对话失败: {exc}") from exc

    store.append_message(session_id, "user", message)
    store.append_message(session_id, "assistant", reply)
    bundle["session_entities"] = agent.session_entities

    return ChatResponse(
        thread_id=session_id,
        session_id=session_id,
        merchant_id=merchant_id,
        operator_id=operator_id,
        reply=reply,
        scene=meta.get("scene", scene),
        subtask=meta.get("subtask", subtask),
        skill_id=meta.get("skill_id"),
        skill_reason=meta.get("skill_reason"),
        key_source=getattr(agent, "_key_source", None),
        response_path=meta.get("path"),
    )


@app.post("/chat/reset")
async def reset_chat(body: ChatRequest):
    merchant_id = (body.merchant_id or "").strip()
    if not merchant_id:
        raise HTTPException(400, "merchant_id 必填")
    operator_id = (body.operator_id or "").strip()
    session_id = (body.session_id or body.thread_id or "").strip() or str(uuid.uuid4())
    store = get_store()
    try:
        session_id = store.ensure_session(
            merchant_id=merchant_id,
            operator_id=operator_id,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.clear_session_messages(session_id)
    _sessions[session_id] = {"session_entities": {}}
    # 新开一场会话，避免与已清空会话混淆
    new_id = store.ensure_session(
        merchant_id=merchant_id,
        operator_id=operator_id,
        session_id=None,
    )
    return {
        "ok": True,
        "thread_id": new_id,
        "session_id": new_id,
        "merchant_id": merchant_id,
    }
