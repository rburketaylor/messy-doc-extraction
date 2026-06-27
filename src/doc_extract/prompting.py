"""Shared prompt formatting for student training and inference."""

from __future__ import annotations


def format_prompt_for_generation(processor, prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
