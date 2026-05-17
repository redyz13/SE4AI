import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REGISTRY = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "Aratako/Ministral-3-8B-Instruct-2512-TextOnly",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


def get_model_id(model_key: str) -> str:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")


def detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(model_id: str):
    backend = detect_backend()

    print(f"Backend: {backend}")
    print(f"Loading model: {model_id}")

    if "Ministral-3" in model_id or "ministral3" in model_id.lower():
        cuda_dtype = torch.bfloat16
    else:
        cuda_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if backend == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=cuda_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    elif backend == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("mps")

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")

    model.eval()
    return tokenizer, model, backend