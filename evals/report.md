# ControlsGuard evaluation report

Cases: 47  |  LLM: mock

## Retrieval ablation (k=5, 20 queries)

| retriever | hit@1 | hit@3 | MRR |
|---|---|---|---|
| bm25 | 0.85 | 1.0 | 0.917 |
| vector | 0.85 | 0.95 | 0.91 |
| hybrid | 0.85 | 0.95 | 0.912 |

## Quality gates

| metric | value | gate | status |
|---|---|---|---|
| retrieval_hybrid_hit@3 | 0.95 | >= 0.95 | PASS |
| retrieval_hybrid_mrr | 0.912 | >= 0.8 | PASS |
| answer_pass_rate | 1.0 | >= 1.0 | PASS |
| graph_pass_rate | 1.0 | >= 1.0 | PASS |
| coverage_pass_rate | 1.0 | >= 1.0 | PASS |
| groundedness_rate | 1.0 | >= 1.0 | PASS |
| refusal_rate | 1.0 | >= 1.0 | PASS |
| clarification_rate | 1.0 | >= 1.0 | PASS |
| injection_block_rate | 1.0 | >= 1.0 | PASS |
| pii_leak_free_rate | 1.0 | >= 1.0 | PASS |
| indirect_injection_neutralised_rate | 1.0 | >= 1.0 | PASS |

## Failing cases

- `ret-19` 'who can use root rights on production servers' {'expected': ['C-AC-02'], 'got': ['C-CM-03', 'C-CM-01', 'C-TP-03']}
