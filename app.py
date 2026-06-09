from config import (
    MODEL_ID,
    ADAPTER_PATH,
    resolve_device_dtype,
    MAX_HISTORY_ROUNDS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    POLICY_KB_PATH,
)

from model_loader import load_model_and_tokenizer
from router import route_query, detect_subtask
from retriever import Retriever
from generator import build_messages, generate_reply, log_trace
from query_optimizer import QueryOptimizer
from business.executor import BusinessActionExecutor
from business.schemas import AgentActionType
from config import (
    ENABLE_QUERY_OPTIMIZER,
    QUERY_OPT_LOG_DIR,
    ENABLE_BUSINESS_API,
    BUSINESS_API_MODE,
    BUSINESS_API_LOG_DIR,
)


class BuyerAgentApp:
    def __init__(self):
        self.tokenizer, self.model = load_model_and_tokenizer(
            model_id=MODEL_ID,
            adapter_path=ADAPTER_PATH,
            device_dtype=resolve_device_dtype()
        )
        self.retriever = Retriever(POLICY_KB_PATH)
        self.history = []
        self.session_entities = {}
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

    def chat(self, user_input: str) -> str:
        working_query = user_input
        if self.query_optimizer:
            opt = self.query_optimizer.optimize(
                user_input,
                history=self.history,
                session_entities=self.session_entities,
            )
            working_query = opt.optimized_query
            if opt.raw_query != opt.optimized_query or opt.flags:
                print(
                    f"[QueryOpt] {opt.raw_query[:40]} -> {opt.optimized_query[:40]} "
                    f"| flags={opt.flags}"
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
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": reply})
                print(
                    f"[BusinessAPI] action={action_out.action.value} "
                    f"api_called={action_out.api_called}"
                )
                return reply

        retrieved_context = ""
        retrieved_result = self.retriever.retrieve_context(
            working_query, scene=scene
        )
        retrieved_context = retrieved_result["context"]
        strong_hit = retrieved_result["strong_hit"]
        low_confidence = retrieved_result.get("low_confidence", False)
        follow_up_question = retrieved_result.get("follow_up_question", "")

        messages = build_messages(
            user_input=working_query,
            scene=scene,
            history=self.history,
            max_history_rounds=MAX_HISTORY_ROUNDS,
            retrieved_context=retrieved_context,
            strong_hit=strong_hit,
            subtask=subtask,
            low_confidence=low_confidence,
            follow_up_question=follow_up_question
        )

        response = generate_reply(
            tokenizer=self.tokenizer,
            model=self.model,
            messages=messages,
            retrieved_context=retrieved_context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY
        )

        log_trace(
            user_input=user_input,
            scene=scene,
            subtask=subtask,
            retrieved_context=retrieved_context,
            response=response
        )

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


def create_app() -> BuyerAgentApp:
    return BuyerAgentApp()