"""doc_extract: dirty-to-clean synthetic-data SFT/QLoRA extraction pipeline.

Four stages: generate dirty synthetic docs -> teacher-label to clean JSON -> QLoRA SFT on
LFM2.5-VL-1.6B-Extract -> field-level evaluation vs teacher gold. "The process is the product."
"""

__version__ = "0.1.0"
