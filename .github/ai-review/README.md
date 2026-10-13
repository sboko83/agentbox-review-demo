# AI review

Every pull request is reviewed by `.github/workflows/ai-review.yml`:
diff → RAG over README, docs/ and backend code of the base branch → LLM → one PR comment
(bugs, architecture, recommendations), updated in place on each push.

- `review.py` — the steps: `diff`, `index`, `review`, `post`
- `corpus.py` — docs, SQL schema and Python rendered to Markdown for the indexer
- `document_indexer/` — the day-21 indexer (multilingual-e5-small)
- secret `OPENCODE_API_KEY`, optional variable `AI_REVIEW_MODEL`

Source: week-7-task-2 of the AIAC9 course repository.
