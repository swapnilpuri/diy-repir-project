from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .qa_item import QAItem
from .labels import LabelRecord


class TraceRecord(BaseModel):
    """
    Maintains the complete lifecycle of a generated Q&A item
    across Steps 1 through 5.

    Step 1: Generation metadata and QA item
    Step 2: Quality gate results
    Step 3: Human evaluation
    Step 4: LLM-as-Judge evaluation
    Step 5: Input to aggregate agreement and analysis reports
    """

    # ========================================================
    # Step 1 - Generation
    # ========================================================

    # Stable link between the generated item and its later human/judge labels.
    trace_id: str
    # Stored as a string; the gate checks the configured category distribution.
    category: str
    # Prompt version, optionally suffixed with _corrected when guidance is added.
    prompt_variant: str
    generator_model: str
    # The factory writes an ISO 8601 UTC timestamp; this field remains a string.
    generator_timestamp: str

    qa_item: QAItem

    # ========================================================
    # Step 2 - Quality Gate
    # ========================================================

    # None means not yet gated; False means evaluated and rejected.
    gate_pass: bool | None = None

    # Each trace gets its own list so failures cannot leak between records.
    gate_failures: list[str] = Field(
        default_factory=list
    )

    # Detailed Step 2 gate results, including structural,
    # heuristic and duplicate checks.
    gate_result: dict[str, Any] | None = None

    # ========================================================
    # Step 3 - Human Labeling
    # ========================================================

    # Accepted records outside the human-review sample retain a null label.
    human_label: LabelRecord | None = None

    # ========================================================
    # Step 4 - LLM-as-Judge
    # ========================================================

    # Null before evaluation or after a failed judge call; check judge_error.
    judge_label: LabelRecord | None = None

    # Explanation for each quality dimension.
    judge_reasons: dict[str, str] | None = None

    # Capture any failure during LLM evaluation.
    judge_error: str | None = None

    # ========================================================
    # Step 5 - Agreement
    # ========================================================

    # Optional per-dimension match flags. Step 5 currently saves aggregate
    # agreement in analysis JSON without populating this per-trace field.
    agreement: dict[str, bool] | None = None

    # ========================================================
    # Legacy Metadata (optional compatibility)
    # ========================================================

    # New LabelRecord objects already contain timestamps
    # and judge metadata. These fields are retained to
    # avoid breaking existing code that may reference them.

    judge_prompt_version: str | None = None
    judge_timestamp: str | None = None
    human_timestamp: str | None = None

    # ========================================================
    # Validation
    # ========================================================

    @model_validator(mode="after")
    def validate_label_records(self) -> "TraceRecord":
        """
        Ensure labels belong to the correct trace and labeler.
        """

        # Field schemas accept either labeler, so enforce field-specific ownership
        # here after nested LabelRecord objects have been validated.
        if self.human_label is not None:
            if self.human_label.labeler != "human":
                raise ValueError(
                    "human_label must have labeler='human'"
                )

            if self.human_label.trace_id != self.trace_id:
                raise ValueError(
                    "human_label trace_id does not match "
                    "TraceRecord trace_id"
                )

        if self.judge_label is not None:
            if self.judge_label.labeler != "llm_judge":
                raise ValueError(
                    "judge_label must have "
                    "labeler='llm_judge'"
                )

            if self.judge_label.trace_id != self.trace_id:
                raise ValueError(
                    "judge_label trace_id does not match "
                    "TraceRecord trace_id"
                )

        return self

    # ========================================================
    # Factory Method
    # ========================================================

    @classmethod
    def new(
        cls,
        category: str,
        prompt_variant: str,
        generator_model: str,
        qa_item: QAItem,
    ) -> "TraceRecord":
        """
        Create a new trace record during Step 1 generation.
        """

        # Use one UTC instant for the timestamp-based ID and generation metadata.
        # Microsecond precision reduces collisions but is not a uniqueness check.
        now = datetime.now(timezone.utc)

        return cls(
            trace_id=now.strftime("%Y%m%d%H%M%S%f"),
            category=category,
            prompt_variant=prompt_variant,
            generator_model=generator_model,
            generator_timestamp=now.isoformat(),
            qa_item=qa_item,
        )
