import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import DEVICE_DTYPE_NAME


def resolve_device_dtype(name: str | None = None):
    key = (name or DEVICE_DTYPE_NAME).lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(key, torch.bfloat16)


def load_model_and_tokenizer(model_id, adapter_path, device_dtype=None):
    if device_dtype is None:
        device_dtype = resolve_device_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=device_dtype,
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        local_files_only=True,
    )

    model.eval()
    return tokenizer, model