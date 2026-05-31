from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    text = text.strip().upper()
    choice_re = "".join(re.escape(c) for c in choices)
    patterns = [
        rf"^\s*\(?([{choice_re}])\)?\s*[\).:]?\s*$",
        rf"(?:ANSWER|ОТВЕТ|CORRECT ANSWER|ПРАВИЛЬНЫЙ ОТВЕТ)\s*[:\-]?\s*\(?([{choice_re}])\)?",
        rf"\b([{choice_re}])\s*[\).:]",
        rf"\(([{choice_re}])\)",
        rf"\b([{choice_re}])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(str(name), torch.float32)


def _load_adapter(model: MathVLM, path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    if any(k.startswith("adapter.") for k in state):
        state = {k.removeprefix("adapter."): v for k, v in state.items() if k.startswith("adapter.")}
    model.adapter.load_state_dict(state, strict=False)


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    inf_cfg = config.get("inference", {})

    manifest = data_cfg.get("eval_manifest", "assets/toy_math_vqa/manifest.jsonl")
    split = data_cfg.get("split", "dev")
    dataset = MathVQADataset(manifest, split=split, max_samples=data_cfg.get("max_samples"))

    device = torch.device(inf_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(inf_cfg.get("dtype", "float32"))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"], trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vision = AutoModel.from_pretrained(model_cfg["vision_encoder"], torch_dtype=dtype, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"], torch_dtype=dtype, trust_remote_code=True)
    lm.resize_token_embeddings(len(tokenizer))

    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**proc_cfg))
    cfg = ModelConfig(
        vision_hidden_size=int(getattr(vision.config, "hidden_size")),
        text_hidden_size=int(lm.get_input_embeddings().embedding_dim),
        num_image_tokens=int(proc_cfg.get("num_image_tokens", 49)),
        image_token_id=int(tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)),
    )
    model = MathVLM(vision, lm, cfg).to(device)
    model.freeze_backbones()
    if model_cfg.get("adapter_path"):
        _load_adapter(model, model_cfg["adapter_path"])
    model.eval()

    rows: list[dict[str, Any]] = []
    for sample in dataset:
        batch = processor.collate([processor(sample)])
        batch = {k: v.to(device) for k, v in batch.items()}
        gen = model.generate(
            batch,
            max_new_tokens=int(inf_cfg.get("max_new_tokens", 16)),
            do_sample=bool(inf_cfg.get("do_sample", False)),
            temperature=float(inf_cfg.get("temperature", 1.0)) if inf_cfg.get("do_sample", False) else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = parse_mc_answer(text) or normalize_text(text)
        rows.append({"id": sample.id, "prediction": pred, "answer": sample.answer, "subject": sample.subject, "output": text})

    output_path = inf_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
