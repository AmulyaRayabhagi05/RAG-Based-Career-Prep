"""
[EVAL] Offline evaluation harness for Elevatr.

    python -m evals.run_evals                      # all suites   (run from backend/)
    python -m evals.run_evals --suite routing      # one suite
    python -m evals.run_evals --suite retrieval --limit 2 --judge-model gemma4:e4b

Suites
    routing    Does the orchestrator pick the right modules?   deterministic: P / R / F1 / exact match
    retrieval  Are the chunks we retrieve relevant?             LLM judge labels -> precision@k, hit@k, MRR
    resume     Are generated bullets grounded + on-role?        LLM judge rubric (1-5 per criterion)
    interview  Does the feedback scorer rank answers correctly? deterministic ordering check + LLM judge rubric

Every run writes a JSON report to evals/results/ so runs can be diffed over time.
Uses the provider in llm.py (LLM_PROVIDER=ollama by default; openai needs OPENAI_API_KEY).
Retrieval/resume also require an ingested Chroma index.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

from dotenv import find_dotenv, load_dotenv

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

load_dotenv(find_dotenv())

from llm import LLM_PROVIDER, model_name
from evals.judge import INTERVIEW_FEEDBACK_RUBRIC, RESUME_BULLET_RUBRIC, LLMJudge
from evals.metrics import hit_at_k, precision_at_k, reciprocal_rank, safe_mean, set_precision_recall_f1

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(EVAL_DIR, "golden_set.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RETRIEVAL_K = 5


def load_golden() -> Dict[str, List[dict]]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Suite 1: routing (no judge needed; ground truth is a set of module names) ──

def eval_routing(cases: List[dict], judge: LLMJudge) -> dict:
    from orchestrator import decompose

    rows = []
    for case in cases:
        decomposition = asyncio.run(decompose(case["query"], case["profile"], history=[]))
        predicted = list(dict.fromkeys(t.module for t in decomposition.sub_tasks))
        scores = set_precision_recall_f1(case["expected_modules"], predicted)
        rows.append({"id": case["id"], "expected": case["expected_modules"], "predicted": predicted, **scores})
        print(f"  {'PASS' if scores['exact_match'] else 'FAIL'}  {case['id']:<26} expected={case['expected_modules']} got={predicted}")

    return {
        "summary": {
            "exact_match_rate": safe_mean(1.0 if r["exact_match"] else 0.0 for r in rows),
            "mean_precision": safe_mean(r["precision"] for r in rows),
            "mean_recall": safe_mean(r["recall"] for r in rows),
            "mean_f1": safe_mean(r["f1"] for r in rows),
        },
        "cases": rows,
    }


# ── Suite 2: retrieval (judge labels each retrieved chunk relevant / not) ──

def eval_retrieval(cases: List[dict], judge: LLMJudge) -> dict:
    from rag.retriever import jobs_collection, retrieve_for_resume

    if jobs_collection.count() == 0:
        return {"skipped": "Chroma 'jobs' collection is empty. Run: python rag/ingestion/ingest_mongo.py"}

    rows = []
    for case in cases:
        chunks = retrieve_for_resume(case["profile"], target_title=case["target_role"])["job_chunks"][:RETRIEVAL_K]
        query = f"Target role: {case['target_role']}. Student profile: {case['profile']}"
        labels = [judge.is_relevant(query, c["text"]).relevant for c in chunks]
        row = {
            "id": case["id"],
            "target_role": case["target_role"],
            "retrieved_titles": [c["title"] for c in chunks],
            "relevance": labels,
            f"precision@{RETRIEVAL_K}": precision_at_k(labels, RETRIEVAL_K),
            f"hit@{RETRIEVAL_K}": hit_at_k(labels, RETRIEVAL_K),
            "reciprocal_rank": reciprocal_rank(labels),
        }
        rows.append(row)
        print(f"  {case['id']:<22} P@{RETRIEVAL_K}={row[f'precision@{RETRIEVAL_K}']:.2f} RR={row['reciprocal_rank']:.2f} titles={row['retrieved_titles'][:3]}")

    return {
        "summary": {
            f"mean_precision@{RETRIEVAL_K}": safe_mean(r[f"precision@{RETRIEVAL_K}"] for r in rows),
            f"hit_rate@{RETRIEVAL_K}": safe_mean(1.0 if r[f"hit@{RETRIEVAL_K}"] else 0.0 for r in rows),
            "MRR": safe_mean(r["reciprocal_rank"] for r in rows),
        },
        "cases": rows,
    }


# ── Suite 3: resume bullets (judge grades against RESUME_BULLET_RUBRIC) ──

def eval_resume(cases: List[dict], judge: LLMJudge) -> dict:
    from modules.resume_builder import generate_resume
    from rag.retriever import jobs_collection

    if jobs_collection.count() == 0:
        return {"skipped": "Chroma 'jobs' collection is empty. Run: python rag/ingestion/ingest_mongo.py"}

    rows = []
    for case in cases:
        bullets = generate_resume(dict(case["profile"]), [case["target_role"]])
        verdict = judge.grade(
            task=f"Write ATS-optimised resume bullets for a student targeting '{case['target_role']}'.",
            context=json.dumps(case["profile"], indent=2),
            output="\n".join(f"- {b}" for b in bullets) or "(no bullets returned)",
            rubric=RESUME_BULLET_RUBRIC,
        )
        rows.append({"id": case["id"], "bullets": bullets, "scores": verdict.by_criterion(),
                     "mean": verdict.mean_score, "judge_summary": verdict.summary})
        print(f"  {case['id']:<16} mean={verdict.mean_score:.2f} {verdict.by_criterion()}")

    return {
        "summary": {
            "mean_score": safe_mean(r["mean"] for r in rows),
            **{f"mean_{c}": safe_mean(r["scores"].get(c, 0) for r in rows) for c in RESUME_BULLET_RUBRIC},
        },
        "cases": rows,
    }


# ── Suite 4: interview feedback (ordering check + judge rubric) ──

def eval_interview(cases: List[dict], judge: LLMJudge) -> dict:
    from modules.interview import _openai_client, generate_feedback

    client = _openai_client()
    rows = []
    for case in cases:
        strong = json.loads(generate_feedback(case["question"], case["strong_answer"], client))["feedback"]
        weak = json.loads(generate_feedback(case["question"], case["weak_answer"], client))["feedback"]

        # Deterministic sanity check: a scorer that can't rank a strong answer above a
        # one-liner is broken, regardless of what any judge thinks.
        ranks_correctly = strong["score"] > weak["score"]

        verdicts = {}
        for label, answer, fb in (("strong", case["strong_answer"], strong), ("weak", case["weak_answer"], weak)):
            verdicts[label] = judge.grade(
                task="Give a candidate feedback on their interview answer.",
                context=f"Question: {case['question']}\nCandidate answer: {answer}",
                output=json.dumps(fb, indent=2),
                rubric=INTERVIEW_FEEDBACK_RUBRIC,
            )

        rows.append({
            "id": case["id"],
            "strong_score": round(strong["score"], 3),
            "weak_score": round(weak["score"], 3),
            "ranks_correctly": ranks_correctly,
            "judge_strong": verdicts["strong"].by_criterion(),
            "judge_weak": verdicts["weak"].by_criterion(),
            "judge_mean": safe_mean([verdicts["strong"].mean_score, verdicts["weak"].mean_score]),
        })
        print(f"  {case['id']:<14} strong={strong['score']:.2f} weak={weak['score']:.2f} "
              f"{'RANKS OK' if ranks_correctly else 'MIS-RANKED'} judge_mean={rows[-1]['judge_mean']:.2f}")

    return {
        "summary": {
            "ranking_accuracy": safe_mean(1.0 if r["ranks_correctly"] else 0.0 for r in rows),
            "mean_feedback_quality": safe_mean(r["judge_mean"] for r in rows),
        },
        "cases": rows,
    }


SUITES: Dict[str, Callable[[List[dict], LLMJudge], dict]] = {
    "routing": eval_routing,
    "retrieval": eval_retrieval,
    "resume": eval_resume,
    "interview": eval_interview,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Elevatr offline evals")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model override (default: the provider's large tier). "
                             "Prefer a different model family than the generator.")
    parser.add_argument("--limit", type=int, default=None, help="Max cases per suite (cheap smoke runs)")
    args = parser.parse_args()

    golden = load_golden()
    judge = LLMJudge(model=args.judge_model)
    selected = list(SUITES) if args.suite == "all" else [args.suite]

    report: Dict[str, Any] = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                              "provider": LLM_PROVIDER,
                              "judge_model": args.judge_model or model_name("large"), "suites": {}}
    for name in selected:
        cases = golden.get(name, [])[: args.limit]
        print(f"\n▶ {name} ({len(cases)} cases)")
        started = time.perf_counter()
        try:
            result = SUITES[name](cases, judge)
        except Exception as exc:  # keep other suites running; record the failure
            result = {"error": f"{type(exc).__name__}: {exc}"}
        result["seconds"] = round(time.perf_counter() - started, 1)
        report["suites"][name] = result

    print("\n══ Summary ══")
    for name, result in report["suites"].items():
        status = result.get("summary") or result.get("skipped") or result.get("error")
        print(f"  {name:<10} {status}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"eval_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
