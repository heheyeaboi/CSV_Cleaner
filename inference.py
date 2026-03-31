"""
Baseline LLM agent for the CSV Clean Environment.

Uses HuggingFace Inference Router with Llama-3.3-70B to issue cleaning operations step-by-step.
"""

import os
import json
import requests

from openai import OpenAI


# ── Configuration ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
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

Important rules:
- Check "Last operation result" to know what was already done successfully.
- Check "Actions already taken" list — never repeat any operation from that list.
- Never repeat the same operation on the same column twice.
- After each successful operation, move on to the next problem.
- Only call "done" when ALL issues are fixed.
"""

# ── LLM client (Google Gemini) ────────────────────────────────────────────
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


def run_task(task_name: str) -> float:
    """Run a single cleaning task and return the final score."""
    try:
        # Reset the environment
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task": task_name},
        ).json()
        obs = reset_resp["observation"]
        action_history = []

        for step in range(20):
            user_prompt = (
                f"Task: {obs['task_description']}\n\n"
                f"Current CSV (first 20 rows):\n{obs['current_csv']}\n\n"
                f"Null counts: {json.dumps(obs['null_counts'])}\n"
                f"Column dtypes: {json.dumps(obs['dtypes'])}\n"
                f"Steps taken: {obs['steps_taken']}\n"
                f"Last operation result: {obs['last_operation_result']}\n"
                f"Actions already taken:\n" + ("\n".join(action_history[:-1]) if action_history else "None") + "\n\n"
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
            action = {
                "operation": parsed["operation"],
                "column": parsed.get("column"),
                "value": parsed.get("value"),
            }

            action_history.append(f"Step {step+1}: {action['operation']}" + (f" on {action['column']}" if action.get('column') else "") + (f" = {action['value']}" if action.get('value') else ""))

            print(f"  Step {step + 1}: {action['operation']}"
                  f"{' -> ' + action['column'] if action['column'] else ''}"
                  f"{' = ' + action['value'] if action['value'] else ''}")

            # Step the environment
            step_resp = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action},
            ).json()

            obs = step_resp["observation"]
            done = step_resp.get("done", False)
            reward = step_resp.get("reward")

            if done:
                score = reward if reward is not None else 0.0
                print(f"  DONE! Score: {score}")
                return float(score)

            if obs.get("errors"):
                print(f"  ⚠ Errors: {obs['errors']}")

        # Loop ended without agent calling "done" — force it
        print("  Max steps reached, forcing done...")
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"action": {"operation": "done"}},
        ).json()

        done = step_resp.get("done", False)
        reward = step_resp.get("reward")
        if done and reward is not None:
            print(f"  DONE Final score: {reward}")
            return float(reward)

        return 0.0

    except Exception as e:
        print(f"  X Exception: {e}")
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
