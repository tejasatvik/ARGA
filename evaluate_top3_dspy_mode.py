#!/usr/bin/env python3
"""
evaluate_top3_dspy_mode.py

LLM + ARGA evaluation pipeline (DSPy). Supports:
 - llm_mode on  : call LLM for each image 
 - llm_mode off : use precomputed predictions CSV/JSON and only run solver verification

Usage examples:
  # Full pipeline (LLM -> solver)
  python evaluate_top3_dspy_mode.py --tasks_csv tasks.csv --prompt_file prompt.txt --out results.csv --llm_mode on --run_solver true

  # Two-stage: generate predictions only (save them)
  python evaluate_top3_dspy_mode.py --tasks_csv tasks.csv --prompt_file prompt.txt --out predictions_only.csv --llm_mode on --run_solver false --save_predictions llm_preds.csv

  # Offline verification using saved predictions
  python evaluate_top3_dspy_mode.py --tasks_csv tasks.csv --predictions_file llm_preds.csv --out final_results.csv --llm_mode off --run_solver true
"""

import os
import sys
import json
import argparse
import time
import re
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm

# DSPy imports
import dspy

# Adjust this import if Task is in a submodule or package
from task import Task

# ---------------------
# Defaults & config
# ---------------------
DEFAULT_MODEL = os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini")
DATASET_TRAIN_DIR = "dataset/training/"
DEFAULT_TIME_LIMIT = 300  # solver time limit per task (seconds)
DSPY_TEMPERATURE = 0.0

# Short instruction that gets added to each prompt call (keeps LLM output stable)
ABSTRACTION_SHORT_GUIDE = (
    "Abstraction options (choose 3): "
    "ccg, nbccg, nbvcg, nbhcg, mcccg, lrg, ccgbr, ccgbr2, na. "
    "Return EXACTLY a JSON array with 3 objects. Each object must be "
    " {\"abstraction\": \"<name>\", \"confidence\": <0.0-1.0>, \"rationale\": \"<=20 words\"}. "
    "Output ONLY the JSON array and nothing else."
)

# regex for finding first JSON array
JSON_ARRAY_RE = re.compile(r"(\[[\s\S]*?\])", re.MULTILINE)

# ---------------------
# Helpers
# ---------------------
def image_to_data_uri(image_path):
    """Convert an image file to a data URI (PNG base64)."""
    im = Image.open(image_path).convert("RGBA")
    bio = BytesIO()
    im.save(bio, format="PNG")
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def extract_json_array(text):
    """
    Find and parse the first JSON array in text.
    Returns (parsed_array_or_None, warning_or_None)
    """
    m = JSON_ARRAY_RE.search(text)
    if not m:
        return None, "no_json_array_found"
    candidate = m.group(1)
    try:
        parsed = json.loads(candidate)
        return parsed, None
    except Exception:
        # tolerant fixes: single quotes and trailing commas in arrays
        fixed = candidate.replace("'", "\"")
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            parsed = json.loads(fixed)
            return parsed, "fixed_single_quotes_or_trailing_commas"
        except Exception as e2:
            return None, f"json_parse_failed:{e2}"

def parse_abstractions_from_text(text):
    """
    Parse LLM output and return (entries_list, parse_warn)
    - entries_list: list of up to 3 dicts {"abstraction","confidence","rationale"}
    - parse_warn: string with parse warnings or None
    """
    parsed_array, warn = extract_json_array(text)
    if parsed_array is None:
        # fallback: simple line-based search for known abstraction names
        known = ["ccg","nbccg","nbvcg","nbhcg","mcccg","lrg","ccgbr","ccgbr2","na"]
        found = []
        for k in known:
            if re.search(rf"\b{k}\b", text, re.IGNORECASE):
                found.append(k)
        uniq = []
        for n in found:
            if n not in uniq:
                uniq.append(n)
            if len(uniq) >= 3:
                break
        if not uniq:
            return [], (warn or "no_abstractions_found_fallback_failed")
        entries = [{"abstraction": n, "confidence": None, "rationale": None} for n in uniq]
        return entries, (warn + ";fallback_line_search") if warn else "fallback_line_search"

    # Normalize parsed array items
    entries = []
    for item in parsed_array[:3]:
        if isinstance(item, dict) and "abstraction" in item:
            entries.append({
                "abstraction": item.get("abstraction"),
                "confidence": float(item.get("confidence")) if item.get("confidence") is not None else None,
                "rationale": item.get("rationale")
            })
        else:
            # try parsing simple string entries like "ccg" or "1. ccg"
            if isinstance(item, str):
                m = re.search(r"(ccgbr2|ccgbr|ccg|nbccg|nbvcg|nbhcg|mcccg|lrg|na)", item, re.IGNORECASE)
                if m:
                    entries.append({"abstraction": m.group(1).lower(), "confidence": None, "rationale": item})
                else:
                    warn = (warn + ";unrecognized_array_item") if warn else "unrecognized_array_item"
            else:
                warn = (warn + ";bad_array_item_type") if warn else "bad_array_item_type"

    if not entries:
        return [], (warn or "json_parsed_but_no_valid_entries")
    # pad to length 3
    while len(entries) < 3:
        entries.append({"abstraction": None, "confidence": None, "rationale": None})
    return entries, warn

# ---------------------
# Main evaluation routine
# ---------------------
def run_evaluation(tasks_csv, prompt_file=None, predictions_file=None, out_csv="llm_eval_full.csv",
                   model_name=DEFAULT_MODEL, llm_mode="on", run_solver=True,
                   save_predictions=None, time_limit=DEFAULT_TIME_LIMIT):
    # Load tasks list
    df_tasks = pd.read_csv(tasks_csv)
    if "task_file" not in df_tasks.columns:
        raise ValueError("tasks_csv must contain a 'task_file' column with filenames (e.g., ddf7fa4f.json).")

    # Load user prompt (long prompt) if provided
    user_prompt = ""
    if prompt_file:
        with open(prompt_file, "r", encoding="utf8") as f:
            user_prompt = f.read().strip()

    # Configure DSPy if needed
    if llm_mode.lower() == "on":
        dspy_llm = dspy.LM(model_name, temperature=DSPY_TEMPERATURE)
        dspy.configure(lm=dspy_llm)
        predictor = dspy.Predict("image_md, prompt -> output")

    # Load precomputed predictions map if llm_mode is off
    preds_map = {}
    if llm_mode.lower() == "off":
        if not predictions_file:
            raise ValueError("When llm_mode is 'off' you must supply --predictions_file")
        if predictions_file.lower().endswith(".csv"):
            dfp = pd.read_csv(predictions_file, dtype=str).fillna("")
            for _, r in dfp.iterrows():
                task = r.get("task_file")
                if not task:
                    continue
                def make_entry(i):
                    return {
                        "abstraction": r.get(f"top{i}", "") or None,
                        "confidence": float(r.get(f"conf{i}", "")) if r.get(f"conf{i}", "") else None,
                        "rationale": r.get(f"rationale{i}", "") or None
                    }
                preds_map[task] = [make_entry(1), make_entry(2), make_entry(3)]
        else:
            with open(predictions_file, "r", encoding="utf8") as f:
                j = json.load(f)
            if isinstance(j, dict):
                preds_map = j
            elif isinstance(j, list):
                for entry in j:
                    t = entry.get("task_file")
                    p = entry.get("predictions", [])
                    preds_map[t] = p

    results = []
    saved_predictions_rows = []

    for _, row in tqdm(df_tasks.iterrows(), total=len(df_tasks), desc="tasks"):
        task_file = row["task_file"]

        # Locate training input image 
        train_img = None
        cand1 = os.path.join(DATASET_TRAIN_DIR, task_file, "0.png")
        cand2 = os.path.join(DATASET_TRAIN_DIR, task_file, "input.png")
        cand3 = os.path.join(DATASET_TRAIN_DIR, task_file)
        if os.path.isfile(cand1):
            train_img = cand1
        elif os.path.isfile(cand2):
            train_img = cand2
        elif os.path.isfile(cand3) and task_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_img = cand3
        else:
            # Could not find train image
            results.append({
                "task_file": task_file,
                "top1": None, "conf1": None, "rationale1": None,
                "top2": None, "conf2": None, "rationale2": None,
                "top3": None, "conf3": None, "rationale3": None,
                "llm_raw": None, "llm_parse_warn": "no_image_found",
                "solver_abstraction": None, "solver_train_success": None, "solver_test_success": None,
                "time_sec": None, "nodes_explored": None
            })
            continue

        parsed_entries = []
        llm_raw = None
        parse_warn = None

        if llm_mode.lower() == "on":
            # Build prompt: include the short guide and the user's long prompt, then embed the image
            image_data_uri = image_to_data_uri(train_img)
            full_prompt = (
                ABSTRACTION_SHORT_GUIDE + "\n\n"
                + (user_prompt + "\n\n" if user_prompt else "")
                + "Embedded image (data URI):\n\n"
                + image_data_uri + "\n\n"
                "IMPORTANT: Output ONLY the JSON array and nothing else."
            )
            try:
                rollout_id = int(time.time() * 1000)
                pred = predictor(image_md=image_data_uri, prompt=full_prompt, config={"rollout_id": rollout_id})
                llm_raw = str(pred)
            except Exception as e:
                llm_raw = f"[dspy_call_failed] {e}"
                parse_warn = "dspy_call_failed"

            if llm_raw:
                parsed_entries, parse_warn2 = parse_abstractions_from_text(llm_raw)
                if parse_warn2:
                    parse_warn = (parse_warn + ";" if parse_warn else "") + parse_warn2

        else:
            # offline mode: read predictions_map
            pred_list = preds_map.get(task_file)
            if pred_list:
                parsed_entries = pred_list[:3]
                llm_raw = json.dumps(pred_list)
            else:
                parse_warn = "no_prediction_for_task_in_predictions_file"

        # Ensure parsed_entries has 3 items
        while len(parsed_entries) < 3:
            parsed_entries.append({"abstraction": None, "confidence": None, "rationale": None})

        # Save predictions to CSV for later offline runs
        if save_predictions and llm_mode.lower() == "on":
            saved_predictions_rows.append({
                "task_file": task_file,
                "top1": parsed_entries[0].get("abstraction"),
                "conf1": parsed_entries[0].get("confidence"),
                "rationale1": parsed_entries[0].get("rationale"),
                "top2": parsed_entries[1].get("abstraction"),
                "conf2": parsed_entries[1].get("confidence"),
                "rationale2": parsed_entries[1].get("rationale"),
                "top3": parsed_entries[2].get("abstraction"),
                "conf3": parsed_entries[2].get("confidence"),
                "rationale3": parsed_entries[2].get("rationale"),
            })

        # Run solver if requested and if we have at least one valid abstraction
        solver_abstr = None
        solver_train_success = None
        solver_test_success = None
        solve_time = None
        nodes_explored = None

        valid_abstractions = [e["abstraction"] for e in parsed_entries if e and e.get("abstraction")]
        if run_solver and valid_abstractions:
            try:
                task_inst = Task(os.path.join(DATASET_TRAIN_DIR, task_file))
                task_inst.allowed_abstractions = valid_abstractions
                abs_name, apply_call, error, train_error, solving_time, nodes_explored = task_inst.solve(
                    shared_frontier=True, time_limit=time_limit, do_constraint_acquisition=True, save_images=False)
                solver_abstr = abs_name
                solver_train_success = (train_error == 0)
                solver_test_success = (error == 0)
                solve_time = solving_time
            except Exception as e:
                parse_warn = (parse_warn + ";" if parse_warn else "") + f"solver_failed:{e}"

        # Append result row
        results.append({
            "task_file": task_file,
            "top1": parsed_entries[0].get("abstraction"),
            "conf1": parsed_entries[0].get("confidence"),
            "rationale1": parsed_entries[0].get("rationale"),
            "top2": parsed_entries[1].get("abstraction"),
            "conf2": parsed_entries[1].get("confidence"),
            "rationale2": parsed_entries[1].get("rationale"),
            "top3": parsed_entries[2].get("abstraction"),
            "conf3": parsed_entries[2].get("confidence"),
            "rationale3": parsed_entries[2].get("rationale"),
            "llm_raw": llm_raw,
            "llm_parse_warn": parse_warn,
            "solver_abstraction": solver_abstr,
            "solver_train_success": solver_train_success,
            "solver_test_success": solver_test_success,
            "time_sec": solve_time,
            "nodes_explored": nodes_explored
        })

    # Save predictions CSV
    if save_predictions and saved_predictions_rows:
        pd.DataFrame(saved_predictions_rows).to_csv(save_predictions, index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    return df_out

# ---------------------
# CLI
# ---------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks_csv", required=True, help="CSV listing tasks (column 'task_file')")
    p.add_argument("--prompt_file", required=False, help="Prompt file to send to LLM (text)")
    p.add_argument("--predictions_file", required=False, help="Predictions CSV/JSON (used when --llm_mode off)")
    p.add_argument("--out", default="llm_eval_full.csv", help="Output CSV filepath")
    p.add_argument("--model", default=DEFAULT_MODEL, help="DSPy model string")
    p.add_argument("--llm_mode", choices=["on", "off"], default="on", help="Call the LLM or use existing predictions")
    p.add_argument("--run_solver", choices=["true", "false"], default="true", help="Whether to run solver after predictions")
    p.add_argument("--save_predictions", default=None, help="If provided and llm_mode=on, save predictions to this CSV")
    p.add_argument("--time_limit", type=int, default=DEFAULT_TIME_LIMIT, help="Solver time limit per task (seconds)")
    args = p.parse_args()

    df = run_evaluation(args.tasks_csv, prompt_file=args.prompt_file, predictions_file=args.predictions_file,
                        out_csv=args.out, model_name=args.model, llm_mode=args.llm_mode,
                        run_solver=(args.run_solver == "true"), save_predictions=args.save_predictions,
                        time_limit=args.time_limit)
    print(df.head())
    print("Wrote:", args.out)
