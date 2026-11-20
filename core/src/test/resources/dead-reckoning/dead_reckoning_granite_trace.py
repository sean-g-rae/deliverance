"""Dead Reckoning trace for Granite 4.1 LLMON prompts.

This is intentionally separate from the generic trace runner because LLMON prompts must be
tokenized as already-rendered raw text. Applying the tokenizer chat template again would test a
different prompt. The JSONL checkpoint schema matches ``dead_reckoning_trace.py``.

Example:

    python examples/dead_reckoning_granite_trace.py \
      --model /path/to/granite-4.1-3b-llmon-preview \
      --prompt-file /path/to/rendered-llmon.txt \
      --output target/dead-reckoning-hf/granite41-llmon.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dead_reckoning_trace import tensor_from_output, tensor_stats


DEFAULT_MODEL = "ibm-granite/granite-4.1-3b-llmon-preview"
DEFAULT_PROMPT = "Generate the first three letters of the alphabet."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / ".reckoning" / "ibm-granite_granite-4.1-3b-llmon-preview"),
                        help="Local model directory; this script never downloads models")
    parser.add_argument("--prompt-file", type=Path, help="File containing an already-rendered raw LLMON prompt")
    parser.add_argument("--output", type=Path, default=Path("target/dead-reckoning-hf/granite41-llmon.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()

    if args.prompt_file is not None:
        rendered_prompt = args.prompt_file.read_text(encoding="utf-8")
        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None and int(encoded["input_ids"][0, 0]) != bos_token_id:
            bos = torch.tensor([[bos_token_id]], dtype=encoded["input_ids"].dtype)
            encoded["input_ids"] = torch.cat((bos, encoded["input_ids"]), dim=1)
            if "attention_mask" in encoded:
                encoded["attention_mask"] = torch.cat(
                    (torch.ones((1, 1), dtype=encoded["attention_mask"].dtype), encoded["attention_mask"]), dim=1
                )
    else:
        messages = [{"role": "user", "content": DEFAULT_PROMPT}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        rendered_prompt = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)

    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = [{
        "kind": "RUN_START",
        "model": args.model,
        "prompt": rendered_prompt,
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "input_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "generator_parameters": {"temperature": 0.0, "max_tokens": args.max_tokens},
    }]
    state = {"forward": -1, "phase": "PREFILL", "position": -1}

    def before_model(_module, _inputs):
        state["forward"] += 1
        state["phase"] = "PREFILL" if state["forward"] == 0 else "DECODE"
        state["position"] = -1 if state["forward"] == 0 else encoded["input_ids"].shape[-1] + state["forward"] - 1

    def record(name, tensor):
        value = tensor.detach().float().reshape(-1).cpu()
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        records.append({
            "kind": "CHECKPOINT",
            "phase": state["phase"],
            "position": state["position"],
            "name": name,
            **tensor_stats(tensor),
            "hash": hashlib.sha256(raw).hexdigest(),
            "first_values": value[:32].tolist(),
        })

    hooks = [model.register_forward_pre_hook(before_model)]

    def hook(name):
        def capture(_module, _inputs, output):
            tensor = tensor_from_output(output)
            if tensor is not None:
                record(name, tensor)
        return capture

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=args.max_tokens, do_sample=False)

    for handle in hooks:
        handle.remove()

    generated_tokens = generated[0, encoded["input_ids"].shape[-1]:].tolist()
    records.append({
        "kind": "RESPONSE_END",
        "text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
        "generated_tokens": generated_tokens,
        "finish_reason": "MAX_TOKENS",
    })
    with args.output.open("w", encoding="utf-8") as stream:
        for record_value in records:
            stream.write(json.dumps(record_value, allow_nan=True) + "\n")

    print(f"HF_DEAD_RECKONING_TRACE={args.output.resolve()}")
    print(f"HF_DEAD_RECKONING_RESPONSE={tokenizer.decode(generated_tokens, skip_special_tokens=False)!r}")
    print(f"HF_DEAD_RECKONING_TOKENS={generated_tokens}")


if __name__ == "__main__":
    main()
