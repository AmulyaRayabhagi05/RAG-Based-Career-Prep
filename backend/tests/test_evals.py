"""
Offline tests for the eval harness, chunking and orchestrator contracts.
No network, no API key, no database: the LLM judge is replaced with a fake model.
"""

import pytest
from pydantic import ValidationError

from evals.judge import (
    RESUME_BULLET_RUBRIC,
    CriterionScore,
    LLMJudge,
    RelevanceVerdict,
    RubricVerdict,
)
from evals.metrics import agreement_rate, hit_at_k, precision_at_k, reciprocal_rank, set_precision_recall_f1
from rag.chunking import build_career_rec_chunk, build_onet_chunk, parse_list_field


# ── Metrics ────────────────────────────────────────────────────────────────

def test_routing_metrics_exact_match():
    scores = set_precision_recall_f1(["skill_gap", "resume_builder"], ["resume_builder", "skill_gap"])
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": True}


def test_routing_metrics_partial_and_extra_module():
    # Planner found one of two expected modules and added an unnecessary one.
    scores = set_precision_recall_f1(["skill_gap", "resume_builder"], ["skill_gap", "job_search"])
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["exact_match"] is False


def test_retrieval_metrics():
    labels = [False, True, True, False, False]
    assert precision_at_k(labels, 5) == 0.4
    assert precision_at_k(labels, 1) == 0.0
    assert hit_at_k(labels, 2) is True
    assert reciprocal_rank(labels) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_judge_human_agreement():
    assert agreement_rate([True, False, True, True], [True, True, True, False]) == 0.5
    assert agreement_rate([], []) == 0.0


# ── LLM judge (with a fake model) ──────────────────────────────────────────

class FakeLLM:
    """Mimics LangChain's `.with_structured_output(schema).invoke(messages)`."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def with_structured_output(self, schema):
        fake = self

        class _Runner:
            def invoke(self, messages):
                fake.prompts.append(messages[-1].content)
                return fake.response

        return _Runner()


def test_judge_grade_builds_rubric_prompt_and_scores():
    verdict = RubricVerdict(
        scores=[CriterionScore(criterion=c, reasoning="ok", score=s)
                for c, s in zip(RESUME_BULLET_RUBRIC, [5, 4, 3])],
        summary="Decent bullets.",
    )
    fake = FakeLLM(verdict)
    result = LLMJudge(llm=fake).grade("task", "profile: Python", "- Built X", RESUME_BULLET_RUBRIC)

    assert result.mean_score == 4.0
    assert result.by_criterion() == {"groundedness": 5, "role_relevance": 4, "ats_quality": 3}
    # Every rubric criterion and the ground-truth context must reach the judge.
    for criterion in RESUME_BULLET_RUBRIC:
        assert criterion in fake.prompts[0]
    assert "profile: Python" in fake.prompts[0]


def test_judge_relevance_returns_binary_label():
    fake = FakeLLM(RelevanceVerdict(reasoning="same occupation", relevant=True))
    assert LLMJudge(llm=fake).is_relevant("Data Scientists", "Job Title: Data Scientists").relevant is True


def test_judge_scores_are_bounded():
    # The schema itself rejects out-of-range scores from a misbehaving judge.
    with pytest.raises(ValidationError):
        CriterionScore(criterion="groundedness", reasoning="x", score=9)


# ── Chunking ([RAG-2 CHUNK]) ───────────────────────────────────────────────

def test_onet_chunk_is_title_anchored_with_metadata():
    doc = {
        "_id": "abc123",
        "occupation_title": "Data Scientists",
        "occupation_description": "Develop and implement methods to analyze data.",
        "skills": "[{'name': 'Programming', 'importance': 4.0}, {'name': 'Mathematics', 'importance': 4.2}]",
        "knowledge": [{"name": "Computers and Electronics"}],
        "task_statements": "['Clean data.', 'Build models.']",
    }
    text, meta = build_onet_chunk(doc)

    assert text.startswith("Job Title: Data Scientists\n")
    assert "Skills: Programming, Mathematics" in text
    assert "Knowledge: Computers and Electronics" in text
    assert "Tasks: Clean data. Build models." in text
    assert meta == {"title": "Data Scientists", "source": "onet", "mongo_id": "abc123"}


def test_career_rec_chunk_drops_pii():
    doc = {"_id": 1, "Name": "Jane Doe", "Age": 22, "Recommended_Career": "Web Developer",
           "Skills": "HTML, CSS", "Interests": "Design", "Education": "BSc"}
    text, meta = build_career_rec_chunk(doc)

    assert "Jane Doe" not in text and "22" not in text
    assert meta["title"] == "Web Developer" and meta["source"] == "career_recs"


def test_parse_list_field_handles_list_of_strings():
    # Regression: career-rec Skills are a list of strings; these used to be dropped entirely.
    assert parse_list_field(["Research", "Statistical Analysis", "Python"]) == "Research, Statistical Analysis, Python"
    assert parse_list_field("['SQL', 'Excel']") == "SQL, Excel"


def test_parse_list_field_handles_plain_strings():
    assert parse_list_field("HTML, CSS") == "HTML, CSS"
    assert parse_list_field(None) == ""


# ── Orchestrator contracts (the real schema, not a copy) ──────────────────

def test_real_decomposition_schema_parses_planner_output():
    from orchestrator import Decomposition

    parsed = Decomposition.model_validate({"sub_tasks": [
        {"module": "skill_gap", "inputs": {"target_role": "Data Scientist", "skills": ["Python"]}},
        {"module": "interview", "inputs": {}},
    ]})
    assert [t.module for t in parsed.sub_tasks] == ["skill_gap", "interview"]


def test_ui_payload_puts_triggered_modules_first():
    from orchestrator import build_ui_payload

    ui = build_ui_payload(
        "find my gaps",
        ["skill_gap"],
        {"skill_gap": {"target_occupation": "Data Scientist",
                       "gaps": [{"skill": "SQL", "gap_score": 1, "description": "needed"}],
                       "recommendations": [{"course": "SQL 101", "platform": "Coursera", "url": "https://x"}]}},
    )
    assert ui["tool_order"][0] == "skill_gap"
    card = ui["cards"]["skill_gap"]
    assert card["active"] and card["items"][0]["course_label"] == "SQL 101"
    assert ui["cards"]["resume_builder"]["active"] is False


# ── Reranker ([RAG-6 RERANK]) ──────────────────────────────────────────────

class FakeCrossEncoder:
    """Scores a (query, passage) pair by how many query words appear in the passage."""

    def predict(self, pairs):
        return [sum(w in passage.lower() for w in query.lower().split()) for query, passage in pairs]


def test_rerank_reorders_by_cross_encoder_score_and_truncates():
    from rag.retriever import rerank_chunks

    stage1 = [  # stage-1 (vector) order: the best match is last
        {"text": "Job Title: Digital Marketer", "title": "Digital Marketer"},
        {"text": "Job Title: Data Analyst", "title": "Data Analyst"},
        {"text": "Job Title: Software Engineer, java", "title": "Software Engineer"},
    ]
    top = rerank_chunks("software engineer java", stage1, top_k=2, model=FakeCrossEncoder())

    assert top[0]["title"] == "Software Engineer"
    assert len(top) == 2
    assert all("rerank_score" in c for c in top)


def test_rerank_handles_empty_candidates():
    from rag.retriever import rerank_chunks

    assert rerank_chunks("anything", [], top_k=5, model=FakeCrossEncoder()) == []


def test_role_abbreviations_are_expanded():
    from rag.retriever import expand_role

    assert expand_role("SWE") == "software engineer"
    assert expand_role(" ml engineer ") == "machine learning engineer"
    assert expand_role("Data Scientists") == "Data Scientists"  # unknown titles pass through
