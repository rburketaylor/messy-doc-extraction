"""Shared train/eval chat prompt formatting."""

from __future__ import annotations

from doc_extract import evaluate, train
from doc_extract.prompting import format_prompt_for_generation


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        self.calls.append({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        })
        return "CHAT:" + messages[0]["content"]


def test_train_prompt_helper_applies_chat_template_once():
    processor = FakeProcessor()

    out = train._format_training_example({"prompt": "extract this", "completion": "{}"}, processor)

    assert out == {"prompt": "CHAT:extract this"}
    assert processor.calls == [{
        "messages": [{"role": "user", "content": "extract this"}],
        "add_generation_prompt": True,
        "tokenize": False,
    }]


def test_train_and_evaluate_use_same_prompt_helper():
    assert train.format_prompt_for_generation is format_prompt_for_generation
    assert evaluate.format_prompt_for_generation is format_prompt_for_generation
