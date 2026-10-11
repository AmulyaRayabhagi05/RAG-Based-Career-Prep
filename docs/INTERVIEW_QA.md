# Interview Q&A cheat sheet

Likely questions, mapped to the README walkthrough stops. Each answer runs 30–60 seconds out loud.
Pattern for every technical answer: **what it is → why → a number or finding → what's next**.

---

## Say this, not that (accuracy traps)

| ❌ Don't say | ✅ Say |
|---|---|
| "I chunked and embedded the resumes" | "I embedded **occupation data** (O\*NET + career recs). The student's profile is the **query**." |
| "Chunked by section with overlap" | "**Record-level** chunks: one occupation = one chunk, no overlap needed." |
| "Hybrid chunking" | "**Hybrid retrieval**: metadata filter on title + vector similarity." |
| "The LLM rewrites the module output nicely" | "Responses are shaped **deterministically**, with no second LLM call." |
| "We prevent prompt injection" | "It's a **known gap**: the profile goes in a system message. Fix: delimited untrusted user message." |
| "An LLM judges the routing" | "Routing is checked **deterministically** against a golden set. The judge is used only where there's no single right answer." |
| "Comma DB" / "golden rule" | "**Chroma**DB" / "**golden set**" |

---

## Stop 1 · Overview
**Q: Give me a 2-minute overview, and what was your role?**
> "Elevatr is a career copilot for students. Instead of juggling a resume tool, a job board and interview prep, they ask one question, and an orchestrator decides which tools to run, in parallel. I owned the AI side: the RAG pipeline that grounds everything in O\*NET occupation data, the LangChain planner, and an eval harness with LLM-as-a-judge to measure retrieval and output quality."

## Stop 2 · Planning
**Q: How does the system decide what to do with a query?**
> "One LLM call with structured output. I defined a Pydantic schema, `Decomposition`, a list of sub-tasks each with a module name and typed inputs, and LangChain's `with_structured_output` forces the model to answer in that schema. I send the planner prompt, a trimmed profile and the last six chat turns, so prompt size is bounded no matter how much data the user has."

**Q: Why not a ReAct / tool-calling agent loop?**
> "Routing here is one-shot: I know all the tools up front and they don't depend on each other. A loop adds round-trips, latency and non-determinism I don't need. One structured call is predictable and easy to test. If modules started needing each other's outputs, I'd move to a graph, like LangGraph."

## Stop 3 · Execution
**Q: How do you keep multi-intent requests fast? What if one module fails?**
> "All sub-tasks run concurrently in an `asyncio.TaskGroup` with a 120-second budget, so latency is roughly the slowest module, not the sum. Each module is wrapped in `safe_execute_module`, which turns an exception into an error field on that module only, so a flaky job API can't take down resume output. The flip side: that isolation hid a bug where job search failed on every call. So isolated errors still need logging."

## Stop 4 · Chunking
**Q: Walk me through the RAG pipeline. How did you chunk, and why?**
> "The vector store holds 238 chunks: 38 O\*NET occupations and 200 career-rec records in ChromaDB. They're structured records, so I chunked at the record level, one occupation per chunk, formatted as Title, Description, Skills, Knowledge, Tasks. Fixed-size chunks would split an occupation, and you'd retrieve a fragment that mentions SQL but has lost which job it belongs to. The title heads every chunk and is stored as metadata so I can filter on it. I embed with all-MiniLM-L6-v2, which only reads 256 tokens, so fields go most-important-first. If task-level recall suffered, I'd move to parent-document chunking."

**Q: Why MiniLM and not OpenAI embeddings?**
> "It's free, local, fast on CPU, and good enough for short structured chunks. The swap point is one function, and the retrieval eval would tell me if a bigger model is worth it."

## Stop 5 · Retrieval
**Q: What's the query, and what if the role isn't in your data?**
> "The query is the student's profile (skills, coursework, projects, major, target role) embedded with the same model. First a metadata filter on `title == target role`. If that matches nothing, as with 'SWE', it falls back to pure cosine search. Measured: exact titles score precision@5 of 1.0, 'SWE' falls back to Backend and Front-end Developer at 0.6. A synonym map or query rewriting would close that gap."

## Stop 7 · Evaluation
**Q: How do you evaluate a system with no single right answer? Why trust an LLM judge?**
> "Three layers. Offline unit tests with the judge mocked. Then four eval suites on a hand-labelled golden set. Routing is deterministic: precision, recall and F1 against the expected modules. For retrieval, a judge labels each chunk relevant or not, which gives precision@5 and MRR. For resume bullets and interview feedback, a judge scores 1–5 against a rubric.
> To trust the judge: anchored rubrics, a reasoning field before the score, structured output bounded to 1–5, temperature 0, and a swappable judge model so it can be a different family than the generator. The gap is human calibration. Next I'd label about 30 examples and check agreement.
> What it found: routing recall 1.0 but precision 0.76, so the small model over-triggers. Retrieval is zero on nursing because the data doesn't exist. And the interview scorer can't tell a strong answer from a weak one."

## Behavioral-technical
**Q: Tell me about a bug. How did you find it, what was the root cause, how do you prevent it?**
> "While building the evals I inspected the stored chunks and saw `Skills:` was empty for all 200 career-rec records, 84% of the index. Root cause: O\*NET stores skills as a list of dicts and the career recs as a plain list of strings; the parser only handled dicts and silently returned an empty string. No crash, just worse retrieval. I fixed the parser, added a regression test with the exact format, re-ingested and verified the chunks. Lesson: RAG bugs are silent, so inspect what's actually stored. Next I'd add an ingest-time check that fails if too many chunks have empty skills, and run the retrieval eval before and after fixes."

**Q: Two more weeks: what do you prioritise?**
> "In order of impact per effort: (1) routing precision, by tightening the planner prompt and proving it with the routing eval; (2) the interview scorer, since it can't rank answers, using a stronger model and calibrating the judge against human labels; (3) data coverage plus a reranker for retrieval precision; (4) prompt-injection hardening by moving user data out of the system message. I'd pick by what the evals say is weakest, not by what's most fun to build."

## Curveballs
**Q: How would this scale to 10k users?**
> "The LLM call dominates latency and cost. I'd cache plans for repeated queries, move ingestion to a scheduled job, use a managed vector DB, and put Redis in front for session memory; the `RedisChatMemory` class already exists. The parallel module design already scales horizontally behind FastAPI workers."

**Q: How do you stop the model inventing skills?**
> "The resume builder is retrieval plus templates, so it can't invent skills; it only uses the student's own keywords. The planner prompt says never invent fields, but in testing the 3B model still added 'JavaScript' to a profile that didn't have it. Groundedness is exactly what the resume judge scores, so it's measured, not assumed."
