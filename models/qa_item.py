from pydantic import BaseModel, Field, field_validator


class QAItem(BaseModel):
    """Structured repair content generated in Step 1 and validated in Step 2."""

    # These length constraints enforce structure; detailed quality and safety
    # checks belong to the gate, human review, and judge rather than this schema.
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    equipment_problem: str = Field(min_length=1)
    # Collection min_length counts entries; the validator below checks their text.
    tools_required: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=3)
    safety_info: str = Field(min_length=1)
    tips: list[str] = Field(min_length=1)

    @field_validator("tools_required", "steps", "tips")
    @classmethod
    def no_blank_items(cls, value):
        """Reject empty or whitespace-only entries without rewriting accepted text."""
        if any(not str(v).strip() for v in value):
            raise ValueError("List items must not be blank")
        return value
