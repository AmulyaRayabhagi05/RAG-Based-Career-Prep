"""
[LLM-JUDGE] LLM-as-a-judge for outputs that have no single correct answer.

Why a judge?
    Resume bullets, interview feedback and "is this retrieved chunk relevant?" can't be
    checked with string equality. A strong model grading against an explicit rubric
    is a scalable proxy for human review.

How we keep the judge honest
    1. Explicit rubric with anchored 1-5 scale per criterion (no vague "rate quality").
    2. Reasoning field comes BEFORE the score in the schema, so the model commits to
       evidence first and the score follows from it (reduces arbitrary numbers).
    3. Structured output (Pydantic): verdicts are always parseable; no regex on prose.
    4. temperature=0 for repeatability across runs.
    5. The judge never sees which system produced the output (no self-preference cue),
       and it can be swapped for a different model family via --judge-model.
    6. Binary labels for retrieval relevance: easier for the judge to get right than
       a graded score, and they map directly onto precision@k / MRR.
"""

from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ── Rubrics ────────────────────────────────────────────────────────────────
# Each criterion: what a 1 looks like vs what a 5 looks like. Anchors matter more than wording.

RESUME_BULLET_RUBRIC: Dict[str, str] = {
    "groundedness": "5 = every skill/claim is supported by the student profile; "
                    "1 = invents skills, tools or experience the student does not have.",
    "role_relevance": "5 = clearly targets the stated role's real responsibilities; "
                      "1 = generic or about a different field.",
    "ats_quality": "5 = strong action verb, concrete task, measurable or specific outcome; "
                   "1 = vague, repetitive, or grammatically broken.",
}

INTERVIEW_FEEDBACK_RUBRIC: Dict[str, str] = {
    "accuracy": "5 = the assessment matches the actual quality of the candidate's answer; "
                "1 = praises a weak answer or criticises a strong one.",
    "actionability": "5 = the improvement is specific and the candidate could apply it next time; "
                     "1 = generic advice such as 'be more detailed'.",
    "specificity": "5 = references concrete content from the answer; "
                   "1 = could have been written without reading the answer.",
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator for a career-coaching AI system. "
    "Grade ONLY against the rubric provided. For each criterion, first write a short "
    "reasoning that cites evidence from the output, then give an integer score 1-5. "
    "Do not reward length. Do not assume facts that are not in the provided context."
)


# ── Structured verdicts ────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    criterion: str
    reasoning: str = Field(description="Evidence-based justification, written before the score.")
    score: int = Field(ge=1, le=5, description="1 = fails the criterion, 5 = fully meets it.")


class RubricVerdict(BaseModel):
    scores: List[CriterionScore]
    summary: str = Field(description="One sentence overall assessment.")

    @property
    def mean_score(self) -> float:
        return round(sum(s.score for s in self.scores) / len(self.scores), 2) if self.scores else 0.0

    def by_criterion(self) -> Dict[str, int]:
        return {s.criterion: s.score for s in self.scores}


class RelevanceVerdict(BaseModel):
    reasoning: str = Field(description="Why the passage does or does not help answer the query.")
    relevant: bool


# ── Judge ──────────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(self, model: str = "gpt-4o", llm: Optional[Any] = None):
        # `llm` lets tests inject a fake model; production builds a real chat model.
        self.llm = llm or init_chat_model(model=model, temperature=0)

    def _invoke(self, schema, user_prompt: str):
        structured = self.llm.with_structured_output(schema)
        return structured.invoke([SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user_prompt)])

    def grade(self, task: str, context: str, output: str, rubric: Dict[str, str]) -> RubricVerdict:
        """Score `output` on every criterion in `rubric` (1-5 each)."""
        rubric_text = "\n".join(f"- {name}: {anchor}" for name, anchor in rubric.items())
        prompt = (
            f"## Task the system was given\n{task}\n\n"
            f"## Context available to the system (ground truth)\n{context}\n\n"
            f"## System output to evaluate\n{output}\n\n"
            f"## Rubric (score each criterion 1-5)\n{rubric_text}\n\n"
            f"Return one score per criterion, using exactly these criterion names: "
            f"{', '.join(rubric)}."
        )
        return self._invoke(RubricVerdict, prompt)

    def is_relevant(self, query: str, passage: str) -> RelevanceVerdict:
        """Binary relevance label for one retrieved chunk (feeds precision@k / MRR)."""
        prompt = (
            f"## Retrieval query\n{query}\n\n"
            f"## Retrieved passage\n{passage[:2000]}\n\n"
            "Is this passage useful context for helping a student with this target role? "
            "It is relevant if it describes the same occupation or one whose skills and tasks "
            "substantially overlap. Answer with reasoning, then relevant=true/false."
        )
        return self._invoke(RelevanceVerdict, prompt)
