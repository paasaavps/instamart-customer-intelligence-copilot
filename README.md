# Instamart Customer Intelligence & Support Copilot

## Overview

This project implements an offline customer intelligence and support copilot for a fictional Instamart-style grocery delivery business.

It combines data auditing, cleaning, customer segmentation, lapse prediction, support-ticket triage, policy retrieval, and a refund-support agent.

All records are synthetic. No hosted LLM, API key, or network connection is required.

**Project modules:**
- Part A: Data foundation and SQLite analytics.
- Part B: Customer segmentation and lapse prediction.
- Part C: Support-ticket classification and privacy.
- Part D: Policy retrieval and refund-agent tools.

## Reproduce

### Requirements
- Python 3.11 or later.
- Dependencies from `requirements.txt`.

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Generate data and run the main pipeline

```powershell
python generate_data.py
python run_project.py
```

### Run the seven refund-agent scenarios

```powershell
python run_agent_scenarios.py
```

Run the scenario command after the main pipeline has generated the database and policy data. It reads `data/raw/agent_scenarios.csv` and writes `outputs/agent_runs.jsonl`.

Refund and escalation actions in this runner are simulated; it does not append new transactions to the real refund ledger.

### Important project files

| Path | Purpose |
|---|---|
| `generate_data.py` | Generates synthetic data |
| `run_project.py` | Runs the main pipeline |
| `run_agent_scenarios.py` | Runs seven agent scenarios |
| `src/` | Project modules |
| `sql/queries.sql` | SQL examples |
| `data/` | Raw, clean, policy, and database files |
| `outputs/` | Generated results |
| `state/` | Agent checkpoints |
| `notebooks/` | Interactive notebook |

## Part A

The raw order dataset contains **13,084 rows**.

### Audit results

| Audit check | Count |
|---|---:|
| Exact duplicate rows | 119 |
| Pending orders | 656 |
| Distinct order IDs | 12,309 |
| Nonnumeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Nonblank gender spellings | 10 |
| Blank genders | 30 |

### Cleaning and storage

The pipeline standardizes city and gender values, handles invalid ages, resolves duplicate orders, repairs order amounts, and handles missing or negative delivery durations.

Cleaned data is saved as CSV files and loaded into SQLite tables for customers, products, and orders.

SQL learning examples are provided in `sql/queries.sql`.

## Part B 

### RFM segmentation

RFM features are calculated from completed orders using a snapshot date of 1 October 2025. K-means clustering produces four segments.

| Segment | Customers |
|---|---:|
| Regular Low-Value | 436 |
| Frequent Loyal | 419 |
| At Risk | 182 |
| Premium | 138 |
| **Total** | **1,175** |

### PCA results

| Component | Explained variance |
|---|---:|
| PC1 | 63.43% |
| PC2 | 21.80% |
| Combined | 85.23% |

### Lapse prediction

The lapse model uses orders before 1 September 2025 for historical features and September completed-order activity for labels.

The lapse cohort contains **1,156 customers**, of whom **443 were labelled lapsed**.

The 19-customer difference from RFM segmentation is due to the different date windows: RFM uses completed orders across the full dataset, while the lapse cohort requires completed-order history before 1 September 2025.

### Model results

The selected balanced Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Precision | 0.5891 |
| Recall | 0.8539 |
| F1-score | 0.6972 |

Confusion matrix:

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 90 | 53 |
| Actual lapsed | 13 | 76 |

Preprocessing is fitted on training data. The saved model was reloaded and its predictions were verified.

The Premium retention list is illustrative because Premium membership is derived from the full RFM window; it is not a leakage-free evaluation.

## Part C 

The pipeline processed **164 tickets**.

### Input and privacy checks

| Check | Result |
|---|---:|
| Empty tickets | 4 |
| Overlong tickets | 4 |
| Eligible tickets | 156 |
| Email redactions | 55 |
| Phone redactions | 91 |
| Order ID redactions | 132 |
| **Total redactions** | **278** |

### Classification results

| Status | Tickets |
|---|---:|
| Classified JSON | 135 |
| Classified by keyword fallback | 12 |
| Invalid model output | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |

There were 19 simulated transient errors, all recovered through retries.

### Zero-shot versus few-shot

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified | 147 | 147 |
| Correct | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Tokens | 7,372 | 12,520 |
| Tokens per classified ticket | 50.15 | 85.17 |
### Routing results
| Destination | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
The 6,000-token budget demonstration reached 6,009 tokens because the final response crossed the threshold. It produced 29 `budget_exhausted` cases. The full-budget triage output was restored afterward.
## Part D 

### Policy retrieval

The policy module uses TF-IDF and cosine similarity to retrieve relevant passages.

| Setting | Value |
|---|---:|
| Chunks | 15 |
| Chunk size | 40 |
| Overlap | 10 |

With overlap, evaluation retrieved 12/12 expected documents and 12/12 expected facts. With zero overlap, it retrieved 12/12 documents and 10/12 facts. Both configurations passed 2/2 fallback cases.

The two questions that failed to retrieve the expected facts with zero overlap were:

1. "What delivery time makes a completed order eligible for a full refund?"
2. "How soon is money returned when a payment fails but is deducted?"

These results demonstrate how chunk overlap affected retrieval of the expected policy facts.

### Refund-agent tools

The agent uses five tools:
1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

Completed orders delivered in more than 45 minutes qualify for a full refund. Orders above INR 1,000 are escalated instead of automatically refunded.

### Seven scenario results

| Scenario | Status | Refund (INR) | Calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 857 |
| S2 | REFUND | 135.00 | 5 | 857 |
| S3 | ESCALATE | 1,999.00 | 5 | 864 |
| S4 | DENIED | 0.00 | 4 | 581 |
| S5 | DENIED | 0.00 | 3 | 325 |
| S6 | NOT_FOUND | 0.00 | 2 | 72 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,006 |

All seven scenarios passed expected-outcome validation. Detailed tool steps are stored in `outputs/agent_runs.jsonl`.

### Stop-safe and recovery tests

| Test | Result |
|---|---|
| S1 with token limit 200 | `FALLBACK_TOKEN_LIMIT` |
| S1 with time limit 0 seconds | `FALLBACK_TIMEOUT` |

The S1 crash-resume test completed `get_order` and `search_policy` before interruption. Resume executed `calculate_refund` and `issue_refund`, without repeating tools.

Final result: **REFUND, INR 128.25**, with exactly four tool executions.

### ReAct versus Plan-and-Execute

The project uses a bounded ReAct loop, where the model selects a tool, receives its observation, and decides the next action. The seven scenarios required 2–6 model calls and 72–3,006 tokens.

Plan-and-Execute could make the initial sequence more explicit, but may introduce planning overhead. It was not benchmarked here, so no numerical performance advantage is claimed. ReAct was retained for incremental decisions, checkpointing, and resume support.

## Limitations and Next Steps

- Data is synthetic and the mock LLM is deterministic, not a hosted language model.
- Refund and escalation actions are local simulations, not real transactions.
- Model results reflect the generated dataset, not production performance.
- The Premium retention list is illustrative.
- Plan-and-Execute was discussed but not benchmarked.
- Production deployment would require access controls, secure secret management, monitoring, data governance, model-drift checks, and human approval workflows.