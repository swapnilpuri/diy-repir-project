from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class QualityScores(BaseModel):
    """Shared binary evaluation schema for human and LLM judgments."""

    # Scores are 1=pass or 0=fail; these fields match settings and judge rubrics.
    answer_completeness: int = Field(ge=0, le=1)  # D1: complete repair guidance
    safety_specificity: int = Field(ge=0, le=1)   # D2: task-specific precautions
    tool_realism: int = Field(ge=0, le=1)         # D3: homeowner-accessible tools
    scope_appropriateness: int = Field(ge=0, le=1)  # D4: appropriate DIY scope
    context_clarity: int = Field(ge=0, le=1)      # D5: clear problem and response
    tip_usefulness: int = Field(ge=0, le=1)       # D6: useful task-specific tips
    # Required in input, but recomputed during validation from the six scores.
    overall_pass: bool

    @model_validator(mode="after")
    def derive_overall_pass(self) -> "QualityScores":
        """Override the supplied overall result so one failing dimension fails the item."""
        # Runs after field validation and is inherited by both label subclasses.
        self.overall_pass = all(
            value == 1
            for value in (
                self.answer_completeness,
                self.safety_specificity,
                self.tool_realism,
                self.scope_appropriateness,
                self.context_clarity,
                self.tip_usefulness,
            )
        )
        return self


class JudgeDecision(QualityScores):
    """Structured model response requested by Instructor in Step 4."""

    # Reasons are separated from scores when saved: scores go into LabelRecord,
    # while explanations become TraceRecord.judge_reasons for error analysis.
    answer_completeness_reason: str
    safety_specificity_reason: str
    tool_realism_reason: str
    scope_appropriateness_reason: str
    context_clarity_reason: str
    tip_usefulness_reason: str


class LabelRecord(QualityScores):
    """Persisted evaluation with provenance linking it to a generated trace."""

    # TraceRecord checks that this ID and labeler match the containing field.
    trace_id: str
    labeler: Literal["human", "llm_judge"]
    # Callers supply UTC-aware datetimes; the schema does not enforce timezone.
    timestamp_utc: datetime
    # Human labels omit these; judge labels identify the rubric and model used.
    judge_prompt_version: str | None = None
    judge_model: str | None = None
