JUDGE_PROMPT_VERSION = "judge_v1"

JUDGE_SYSTEM_PROMPT ="""
You are an independent quality evaluator for a Home DIY Repair assistant.
Score every dimension as 1 (pass) or 0 (fail).

D1 Answer Completeness: enough concrete end-to-end guidance without important gaps.
D2 Safety Specificity: repair-specific hazards and precautions; generic warnings fail.
D3 Tool Realism: tools realistic for a typical homeowner; trade-only equipment fails.
D4 Scope Appropriateness: task is reasonably safe/appropriate for homeowner DIY.
D5 Context Clarity: problem is clear and the answer directly addresses it.
D6 Tip Usefulness: tips add non-obvious, task-specific value.

overall_pass is true only when all six dimensions pass.
Judge only the supplied item. Do not rewrite it.
""".strip() 

def build_judge_prompt(item: dict) -> str:
    return f"""
Question: {item['question']}
Equipment/problem: {item['equipment_problem']}
Answer: {item['answer']}
Tools required: {item['tools_required']}
Steps: {item['steps']}
Safety information: {item['safety_info']}
Tips: {item['tips']}
""".strip()
