GENERATOR_PROMPT_VERSION = "generator_v1"

GENERATOR_PROMPT = """
You generate high-quality synthetic Q&A data for a Home DIY Repair assistant.

Generate exactly one realistic homeowner repair question and answer for the
requested category.

Requirements:
- The homeowner should be able to understand the problem from the question.
- Give a practical, sequential answer with at least 3 concrete steps.
- Include realistic homeowner-accessible tools.
- Include specific, task-relevant safety precautions. Avoid generic phrases such
  as "be careful" or "stay safe" without explaining the actual hazard.
- Keep the task within a reasonable DIY scope. If a dangerous or specialized
  portion requires a professional, clearly say so.
- Include useful, task-specific tips that add value beyond the main steps.
- Do not invent unnecessary specialized equipment.
- The answer should be narrative and sufficiently detailed, normally around
  700-1300 characters.
- Do not mention that the content is synthetic or generated.

Category: {category}

Return only the structured fields requested by the schema.
"""

CORRECTION_ADDENDUM = """
Additional targeted guidance:
{targeted_guidance}
"""
