"""Phase 6: QLoRA SFT of Liquid LFM2.5-VL-1.6B-Extract on prompt/completion invoice data.

TEXT-ONLY training (the -VL image path is unused this phase). Loads the base in 4-bit NF4, attaches
LoRA explicitly (prepare_model_for_kbit_training -> get_peft_model), and SFT-trains only the
completion loss (the chat template lacks {% generation %} markers, so assistant_only_loss is not
usable). Saves the adapter + a merged full-precision export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from doc_extract import config
from doc_extract.prompting import format_prompt_for_generation


def _load_base_4bit(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, device_map="auto", dtype=torch.bfloat16,
        quantization_config=bnb, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    return model, processor


def _format_training_example(example: dict[str, str], processor) -> dict[str, str]:
    return {"prompt": format_prompt_for_generation(processor, example["prompt"])}


def train(
    *, train_file: Path, base_id: str, revision: str, adapter_dir: Path,
    epochs: int, seed: int, max_length: int,
) -> str:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(seed, deterministic=True)
    model, processor = _load_base_4bit(base_id, revision)
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # Explicit LoRA attachment (pedagogically transparent; verified-safe default for the
    # hybrid conv/GQA backbone whose exact module names aren't published).
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset("json", data_files=str(train_file), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])
    ds = ds.map(lambda example: _format_training_example(example, processor))

    args = SFTConfig(
        output_dir=str(adapter_dir) + "-runs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=seed,
        data_seed=seed,
        max_length=max_length,
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,           # already PEFT-wrapped above; do NOT also pass peft_config=
        processing_class=processor,
        args=args,
        train_dataset=ds,
    )
    trainer.train()

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    return str(adapter_dir)


def merge(adapter_dir: Path, base_id: str, revision: str, merged_dir: Path) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Merge on a FULL-PRECISION base (4-bit merge is unreliable per research).
    base = AutoModelForImageTextToText.from_pretrained(
        base_id, revision=revision, device_map="auto", dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_id, revision=revision, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    return str(merged_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="QLoRA SFT train LFM2.5-VL-1.6B-Extract")
    p.add_argument("--train-file", type=Path, default=config.SFT_DIR / "train.jsonl")
    p.add_argument("--base", default=config.STUDENT_MODEL_ID)
    p.add_argument("--revision", default=config.STUDENT_REVISION)
    p.add_argument("--adapter-dir", type=Path, default=config.ADAPTER_DIR)
    p.add_argument("--merged-dir", type=Path, default=config.MERGED_DIR)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args(argv)
    config.ensure_dirs()
    adapter = train(
        train_file=args.train_file, base_id=args.base, revision=args.revision,
        adapter_dir=args.adapter_dir, epochs=args.epochs, seed=args.seed,
        max_length=args.max_length,
    )
    print(f"adapter -> {adapter}")
    if not args.skip_merge:
        merged = merge(args.adapter_dir, args.base, args.revision, args.merged_dir)
        print(f"merged  -> {merged}")


if __name__ == "__main__":
    main()
