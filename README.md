# CSV Data Cleaning Agent (`csv_clean_env`)

An OpenEnv reinforcement learning environment where an LLM agent cleans messy CSV datasets by issuing one structured operation per step. The agent inspects the current state of the data — column types, null counts, and a CSV preview — then selects a cleaning operation. After each step the environment returns an updated observation. When the agent signals completion, the environment scores the result against a hidden ground-truth dataset using four equally-weighted metrics.

**Live Environment:** https://riyansh33-csv-clean-env-v2.hf.space

---

## Why This Domain?

- **Real-world utility.** Data cleaning is one of the most time-consuming tasks in any data pipeline. An agent that can reliably handle missing values, inconsistent casing, duplicate rows, and wrong column types has immediate practical value.
- **Deterministic grading.** Every task has a fixed ground-truth DataFrame. The scoring function compares the agent's output to it cell-by-cell, making evaluation fully reproducible with no human judgment required.
- **LLM-friendly action space.** The nine operations map naturally to plain-English instructions. The agent only needs to return a small JSON object (`operation`, `column`, `value`), making it straightforward for any instruction-tuned LLM.

---

## Action Space

Each step, the agent submits a `CsvCleanAction` with three fields:

| Field | Type | Description |
|---|---|---|
| `operation` | `str` (required) | One of the 9 operations listed below |
| `column` | `str \| null` | Target column name (required by most operations) |
| `value` | `str \| null` | Auxiliary value (fill value, new name, target type) |

### Supported Operations

| Operation | Required Fields | Description |
|---|---|---|
| `drop_nulls` | `column` | Drop all rows where the specified column is null |
| `fill_nulls` | `column`, `value` | Fill null values in the column with the given value (auto-casts to match column dtype) |
| `fix_type` | `column`, `value` | Cast column to a new type: `int`, `float`, `datetime`, or `str` |
| `rename_column` | `column`, `value` | Rename column from `column` to `value` |
| `drop_column` | `column` | Remove the entire column from the dataset |
| `deduplicate` | (optional `column`) | Remove duplicate rows; optionally deduplicate on a single column |
| `strip_whitespace` | `column` | Strip leading/trailing whitespace from all string values in the column |
| `standardize_case` | `column` | Lowercase all string values in the column |
| `done` | — | Signal that cleaning is complete; triggers final scoring |

---

## Observation Space

After each step, the environment returns a `CsvCleanObservation`:

| Field | Type | Description |
|---|---|---|
| `current_csv` | `str` | CSV string of the first 20 rows of the current dataset |
| `num_rows` | `int` | Total number of rows |
| `num_cols` | `int` | Total number of columns |
| `null_counts` | `dict[str, int]` | Number of null values per column |
| `dtypes` | `dict[str, str]` | Data type string for each column |
| `last_operation_result` | `str` | Human-readable result message from the last operation |
| `errors` | `list[str]` | Error messages if the last operation failed |
| `task_name` | `str` | Current task identifier (`easy`, `medium`, `hard`) |
| `task_description` | `str` | Plain-English description of what needs to be cleaned |
| `steps_taken` | `int` | Number of steps used so far |

---

## Tasks

### Easy — Product Sales Dataset (20 rows, 4 columns)

| Aspect | Details |
|---|---|
| **Columns** | `id`, `product`, `price`, `quantity` |
| **Problems** | Product names have extra whitespace; `price` is a string column with `"N/A"` and `None` values; `quantity` has 3 null values |
| **Goal** | Strip product names, drop rows with bad prices, cast price to float, fill missing quantities with 0 |
| **Target Score** | 0.9 |

### Medium — Employee Dataset (30 rows, 5 columns)

| Aspect | Details |
|---|---|
| **Columns** | `emp_id`, `name`, `Department`, `salary`, `join_date` |
| **Problems** | 4 exact duplicate rows; `Department` has mixed casing (`Engineering`, `ENGINEERING`, `engineering`); column named `Department` instead of `department`; `join_date` is a string not datetime; 4 null salaries |
| **Goal** | Deduplicate, standardize and rename department, cast join_date to datetime, fill missing salaries with 0 |
| **Target Score** | 0.7 |

### Hard — Medical Patient Dataset (40 rows, 10 columns)

| Aspect | Details |
|---|---|
| **Columns** | `patient_id`, `name`, `age`, `gender`, `blood_pressure`, `diagnosis`, `notes`, `admission_date`, `discharge_date`, `insurance_code` |
| **Problems** | 5 duplicate rows; `notes` column is entirely null; names and blood_pressure have whitespace; `age` contains `"unknown"` strings; `gender` has mixed casing; dates are strings; 7 null insurance codes |
| **Goal** | Drop notes column, deduplicate, strip whitespace from name and blood_pressure, fix age type (coerce unknowns to NaN), standardize gender casing, cast dates to datetime, fill missing insurance codes with `UNKNOWN` |
| **Target Score** | 0.5 |

---

## Reward / Scoring

When the agent calls `operation: "done"`, the environment computes a final score by comparing the agent's DataFrame to the ground-truth DataFrame. The score is the average of four equally-weighted sub-scores (each 25%):

| Sub-Score | Weight | How It's Computed |
|---|---|---|
| **dtype_score** | 25% | Fraction of shared columns whose dtype strings match the ground truth |
| **null_score** | 25% | Fraction of shared columns whose null counts match the ground truth |
| **shape_score** | 25% | Continuous score based on how close row and column counts are to ground truth; `1.0` for exact match, scales down proportionally |
| **value_score** | 25% | If shapes match, cell-by-cell equality ratio across shared columns; `0.0` if shapes differ |

The final score is clipped to `[0.0, 1.0]` and rounded to 4 decimal places.

---

## Quick Start

### 1. Install dependencies

```bash
cd csv_clean_env
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

### 2. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server exposes HTTP + WebSocket endpoints at `http://localhost:8000`.

### 3. Run the baseline agent

```bash
export GROQ_API_KEY="your_groq_key_here"
export API_BASE_URL="https://riyansh33-csv-clean-env-v2.hf.space"
python inference.py
```

This will run the agent on all three tasks and print the scores.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | URL of the running environment server |
| `GROQ_API_KEY` | — | Groq API key for inference |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model ID on Groq |

---

## Project Structure

```
csv_clean_env/
├── models.py                           # CsvCleanAction & CsvCleanObservation
├── client.py                           # EnvClient wrapper
├── inference.py                        # Baseline LLM agent
├── openenv.yaml                        # Environment metadata & task definitions
├── pyproject.toml                      # Package config & dependencies
├── README.md
└── server/
    ├── app.py                          # FastAPI app entry point
    ├── csv_clean_env_environment.py    # Core environment logic
    └── requirements.txt               # Server-side dependencies
```

---

## License

BSD-style license. See [LICENSE](LICENSE) for details.
