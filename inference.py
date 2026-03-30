"""
Baseline LLM agent for the CSV Clean Environment.

Uses a HuggingFace-hosted model via the OpenAI-compatible API
to issue cleaning operations step-by-step.
"""

import os
import json

from openai import OpenAI
from openenv.core.env_client import EnvClient
from models import CsvCleanAction


# ── Configuration ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── LLM client (HuggingFace Inference API, OpenAI-compatible) ────────────
llm_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN,
)


SYSTEM_PROMPT = """You are a data-cleaning agent. You will receive:
- A task description
- A preview of the current CSV data
- The null counts per column
- The data types per column

You must respond with ONLY a valid JSON object (no markdown, no explanation) with these fields:
{
  "operation": "<one of: drop_nulls, fill_nulls, fix_type, rename_column, drop_column, deduplicate, strip_whitespace, standardize_case, done>",
  "column": "<column name or null>",
  "value": "<value string or null>"
}

When you believe the dataset is fully clean, use operation "done".
"""


def run_task(task_name: str) -> float:
    """Run a single cleaning task and return the final score."""
    try:
        env = EnvClient(base_url=API_BASE_URL)
        obs = env.reset(task=task_name)

        for step in range(20):
            user_prompt = (
                f"Task: {obs.task_description}\n\n"
                f"Current CSV (first 20 rows):\n{obs.current_csv}\n\n"
                f"Null counts: {json.dumps(obs.null_counts)}\n"
                f"Column dtypes: {json.dumps(obs.dtypes)}\n"
                f"Steps taken: {obs.steps_taken}\n\n"
                f"What is the next cleaning operation?"
            )

            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            parsed = json.loads(raw)
            action = CsvCleanAction(
                operation=parsed["operation"],
                column=parsed.get("column"),
                value=parsed.get("value"),
            )

            print(f"  Step {step + 1}: {action.operation}"
                  f"{' -> ' + action.column if action.column else ''}"
                  f"{' = ' + action.value if action.value else ''}")

            obs = env.step(action)

            if obs.last_operation_result.startswith("Episode complete"):
                score_str = obs.last_operation_result.split(":")[-1].strip()
                score = float(score_str)
                print(f"  ✓ Done! Score: {score}")
                return score

            if obs.errors:
                print(f"  ⚠ Errors: {obs.errors}")

        # Loop ended without agent calling "done" — force it
        print("  Max steps reached, forcing done...")
        action = CsvCleanAction(operation="done")
        obs = env.step(action)
        if obs.last_operation_result.startswith("Episode complete"):
            score_str = obs.last_operation_result.split(":")[-1].strip()
            score = float(score_str)
            print(f"  ✓ Final score: {score}")
            return score

        return 0.0

    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return 0.0


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running task: {task}")
        print(f"{'='*50}")
        scores[task] = run_task(task)

    print(f"\n{'='*50}")
    print("Results:")
    print(f"{'='*50}")
    for task, score in scores.items():
        print(f"  {task:8s}: {score:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':8s}: {avg:.4f}")
