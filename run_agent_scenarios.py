
import csv
import json
from pathlib import Path

from src import refund_agent as agent


ROOT = Path(__file__).resolve().parent
SCENARIO_FILE = ROOT / "data/raw/agent_scenarios.csv"
OUTPUT_FILE = ROOT / "outputs/agent_runs.jsonl"


# Simulated actions prevent duplicate entries in the real refund ledger.
def simulate_refund(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "simulated": True
    }


def simulate_escalation(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "simulated": True
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    original_refund = agent.TOOL_MAP["issue_refund"]
    original_escalation = agent.TOOL_MAP["escalate_to_human"]

    results = []

    try:
        agent.TOOL_MAP["issue_refund"] = simulate_refund
        agent.TOOL_MAP["escalate_to_human"] = simulate_escalation

        with SCENARIO_FILE.open(
            "r", encoding="utf-8", newline=""
        ) as f:
            scenarios = list(csv.DictReader(f))

        for scenario in scenarios:
            result = agent.run_agent(
                user_message=scenario["message"],
                scenario_id=scenario["scenario_id"]
            )

            record = {
                "scenario_id": scenario["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "expected_outcome": scenario["expected_outcome"],
                "expected_refund_inr": float(
                    scenario["expected_refund_inr"]
                ),
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"]
            }

            results.append(record)

    finally:
        agent.TOOL_MAP["issue_refund"] = original_refund
        agent.TOOL_MAP["escalate_to_human"] = original_escalation

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print("Scenarios executed:", len(results))

    for record in results:
        print(
            record["scenario_id"],
            "| Expected:", record["expected_outcome"],
            "| Actual:", record["status"],
            "| Calls:", record["llm_calls"],
            "| Tokens:", record["tokens_used"]
        )


if __name__ == "__main__":
    main()
