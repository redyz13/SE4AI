import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REGISTRY = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "Aratako/Ministral-3-8B-Instruct-2512-TextOnly",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


SUPPORTED_QUANTIZATION_MODES = {"none", "8bit", "4bit"}


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


def get_cuda_dtype(model_id: str):
    if "Ministral-3" in model_id or "ministral3" in model_id.lower():
        return torch.bfloat16

    return torch.float16


def build_quantization_config(quantization: str):
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "Quantized loading requires bitsandbytes support. "
            "Install it with: pip install -U bitsandbytes accelerate transformers"
        ) from error

    if quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    raise ValueError(f"Unsupported quantization mode: {quantization}")


def load_model(model_id: str, quantization: str = "none"):
    backend = detect_backend()

    if quantization not in SUPPORTED_QUANTIZATION_MODES:
        available = ", ".join(sorted(SUPPORTED_QUANTIZATION_MODES))
        raise ValueError(
            f"Unsupported quantization value '{quantization}'. "
            f"Available: {available}"
        )

    print(f"Backend: {backend}")
    print(f"Loading model: {model_id}")
    print(f"Quantization: {quantization}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if backend == "cuda":
        cuda_dtype = get_cuda_dtype(model_id)

        if quantization in {"8bit", "4bit"}:
            quantization_config = build_quantization_config(quantization)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                dtype=cuda_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

    elif backend == "mps":
        if quantization != "none":
            raise RuntimeError(
                "8-bit and 4-bit quantization are supported only on CUDA in this pipeline."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("mps")

    else:
        if quantization != "none":
            raise RuntimeError(
                "8-bit and 4-bit quantization are supported only on CUDA in this pipeline."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")

    model.eval()
    return tokenizer, model, backend