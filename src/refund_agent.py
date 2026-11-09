
"""Part D: database-backed refund tools and bounded ReAct loop."""

from pathlib import Path
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"

rag = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id=?",
            (order_id,)
        ).fetchone()

    return (
        {"error": "order not found"}
        if not row
        else dict(zip(
            ["order_id", "status", "order_amount", "delivery_minutes"],
            row
        ))
    )


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = (
        float(order_amount)
        if delivery_minutes is not None and float(delivery_minutes) > 45
        else 0.0
    )

    return {"refund_inr": round(amount, 2)}


def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(exist_ok=True)
    path = OUT / "refund_ledger.jsonl"

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "order_id": order_id,
            "refund_inr": refund_inr
        }) + "\n")

    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using the order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search refund policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an approved refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund request to a human for approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


def run_agent(
    user_message,
    scenario_id="demo",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None
):
    run_id = run_id or str(uuid.uuid4())
    STATE.mkdir(exist_ok=True)

    messages = [
        {
            "role": "system",
            "content": "AGENT: Use tools to verify orders and follow refund policy."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    tokens = 0
    steps = []
    executed = []
    start = time.monotonic()
    llm_calls = 0

    def checkpoint(step):
        obj = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens,
            "step_number": step,
            "tools_executed": executed,
            "steps": steps,
            "llm_calls": llm_calls
        }

        (STATE / f"{run_id}.json").write_text(
            json.dumps(obj, default=str),
            encoding="utf-8"
        )

    for iteration in range(max_iterations):

        if time.monotonic() - start >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: hand off to a human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens += response.usage.total_tokens

        if tokens > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""

            return {
                "status": content.split(":", 1)[0],
                "final_answer": content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        tc = msg.tool_calls[0]
        args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            }]
        })

        observation = TOOL_MAP[tc.name](**args)
        executed.append(tc.name)

        steps.append({
            "tool": tc.name,
            "arguments": args,
            "observation": observation
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        checkpoint(iteration + 1)

        if crash_after is not None and len(executed) == crash_after:
            raise RuntimeError("Simulated crash after checkpoint")

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: hand off to a human agent.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):

    checkpoint_path = STATE / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    messages = saved["messages"]
    tokens_used = saved["tokens_used"]
    steps = saved["steps"]
    tools_executed = saved["tools_executed"]
    llm_calls = saved["llm_calls"]
    scenario_id = saved["scenario_id"]

    print("Resuming run:", run_id)
    print("Previously completed tools:", tools_executed)

    # Continue from the saved conversation.
    remaining_iterations = max_iterations - llm_calls

    for _ in range(max(0, remaining_iterations)):

        msg = mock_llm.chat(messages, tools=TOOLS_SPEC)
        response = msg.choices[0].message

        llm_calls += 1
        tokens_used += msg.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        if not response.tool_calls:
            answer = response.content or ""
            return {
                "status": answer.split(":")[0],
                "final_answer": answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        tc = response.tool_calls[0]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments
            }]
        })

        observation = TOOL_MAP[tc.name](**tc.arguments)

        tools_executed.append(tc.name)

        steps.append({
            "tool": tc.name,
            "arguments": tc.arguments,
            "observation": observation
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        saved.update({
            "messages": messages,
            "tokens_used": tokens_used,
            "step_number": len(steps),
            "tools_executed": tools_executed,
            "steps": steps,
            "llm_calls": llm_calls
        })

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2)

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: hand off to a human agent.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }