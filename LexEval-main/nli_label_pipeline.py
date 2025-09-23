import os
import pandas as pd
import numpy as np
import re
import torch
import gc
from contextlib import contextmanager
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional, Callable, Tuple

# ========== Utility Functions ==========
def clear_cache() -> None:
    """Free CUDA & Python memory."""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

@contextmanager
def get_mnli():
    """Context-managed MNLI pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        return_all_scores=True
    )
    try:
        yield pipe
    finally:
        del pipe
        clear_cache()

# ========== Core Components ==========

def load_previous_labels(
    prev_results_path: str,
    join_keys: List[str]
) -> pd.DataFrame:
    """
    Load and validate previous MNLI results from CSV.
    """
    if not isinstance(prev_results_path, str):
        raise TypeError("prev_results_path must be a string")
    if not os.path.isfile(prev_results_path):
        raise FileNotFoundError(f"File not found: {prev_results_path}")
    prev_df = pd.read_csv(prev_results_path)
    if prev_df.duplicated(subset=join_keys).any():
        dupes = prev_df[prev_df.duplicated(subset=join_keys, keep=False)]
        raise ValueError(
            "Duplicate MNLI entries found in previous CSV for keys:\n"
            + str(dupes[join_keys])
        )
    missing_keys = [k for k in join_keys if k not in prev_df.columns]
    if missing_keys:
        raise KeyError(f"Missing join keys in previous CSV: {missing_keys}")
    return prev_df.dropna(subset=join_keys).copy()


def merge_previous_labels(
    df: pd.DataFrame,
    prev_df: pd.DataFrame,
    join_keys: List[str],
    label_col: str
) -> pd.DataFrame:
    """
    Merge label_col from prev_df into df on join_keys.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(prev_df, pd.DataFrame):
        raise TypeError("df and prev_df must be pandas DataFrames")
    if not isinstance(join_keys, list) or not all(isinstance(k, str) for k in join_keys):
        raise TypeError("join_keys must be a list of strings")
    if not isinstance(label_col, str):
        raise TypeError("label_col must be a string")
    for key in join_keys:
        if key not in df.columns or key not in prev_df.columns:
            raise KeyError(f"Join key '{key}' not in DataFrame columns")
    if label_col not in prev_df.columns:
        raise KeyError(f"Label column '{label_col}' not in previous DataFrame")

    merged = df.merge(
        prev_df[join_keys + [label_col]],
        on=join_keys,
        how="left",
        suffixes=("", "_prev")
    )
    merged[label_col] = merged[label_col].combine_first(
        merged[f"{label_col}_prev"]
    )
    merged.drop(columns=[f"{label_col}_prev"], inplace=True)
    return merged


def run_mnli_on_missing(
    df: pd.DataFrame,
    declarative_generator: Callable[[List[str]], List[dict]],
    source_col: str
) -> pd.DataFrame:
    """
    Run MNLI only on rows where label is missing. Adds label and probs columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not callable(declarative_generator):
        raise TypeError("declarative_generator must be callable")
    if not isinstance(source_col, str):
        raise TypeError("source_col must be a string")

    with get_mnli() as mnli_pipe:
        inputs = []
        for _, row in df.iterrows():
            sent = row.get(f"best_{source_col}_sentence", "") or ""
            prompt = row.get("root_prompt", "") or ""
            inputs.append(f"{sent.strip()} \n {prompt}" if sent.strip() else "")

        generations = declarative_generator(inputs)
        declar_col = []
        for gen, inp in zip(generations, inputs):
            text = gen.get("generated_text", "").strip()
            if re.search(r"\\w", text):
                declar_col.append(text)
            else:
                fallback = inp.split("\n", 1)[0].strip()
                declar_col.append(fallback if re.search(r"\\w", fallback) else None)

        df[f"{source_col}_declarative"] = declar_col

        examples = [
            {
                "text": str(row.get(f"{source_col}_declarative", "") or ""),
                "text_pair": str(row.get("reference_ans_declarative", "") or "")
            }
            for _, row in df.iterrows()
        ]

        results = mnli_pipe(examples)
        labels, probs = [], []
        for out in results:
            best = max(out, key=lambda d: d.get("score", 0))
            labels.append(1 if best.get("label") == "entailment" else 0)
            score_map = {d.get("label"): d.get("score", 0.0) for d in out}
            probs.append([
                score_map.get("entailment", 0.0),
                score_map.get("contradiction", 0.0),
                score_map.get("neutral", 0.0)
            ])

        df[f"mnli_label_{source_col}_to_ref"] = labels
        df[f"mnli_probs_{source_col}_to_ref"] = probs
    return df


def aggregate_results(
    df: pd.DataFrame,
    join_keys: List[str],
    source_col: str
) -> pd.DataFrame:
    """
    Group by join_keys, aggregating MNLI labels (max) and taking first for other columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(join_keys, list) or not all(isinstance(k, str) for k in join_keys):
        raise TypeError("join_keys must be a list of strings")
    if not isinstance(source_col, str):
        raise TypeError("source_col must be a string")

    missing = [k for k in join_keys if k not in df.columns]
    if missing:
        raise KeyError(f"Missing group keys in DataFrame: {missing}")

    agg = {}
    for col in df.columns:
        if col in join_keys:
            continue
        agg[col] = "max" if col == f"mnli_label_{source_col}_to_ref" else "first"

    grouped = df.groupby(join_keys).agg(agg).reset_index()
    for col in grouped.columns:
        if col in join_keys or col == f"mnli_label_{source_col}_to_ref":
            continue
        grouped[col] = grouped[col].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray, pd.Series)) and len(x) == 1 else x
        )
    return grouped

# ========== Master Pipeline Function ==========
def evaluate_pipeline_incremental(
    df_list: List[pd.DataFrame],
    declarative_generator: Callable[[List[str]], List[dict]],
    source_col: str = "base_rag",
    prev_results_path: Optional[str] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Orchestrates incremental MNLI labeling with backups.
    Returns grouped results and ungrouped backups.
    """
    # Input validation
    if not isinstance(df_list, list) or not all(isinstance(d, pd.DataFrame) for d in df_list):
        raise TypeError("df_list must be a list of pandas DataFrames")
    if not callable(declarative_generator):
        raise TypeError("declarative_generator must be callable")
    if not isinstance(source_col, str):
        raise TypeError("source_col must be a string")
    if prev_results_path is not None and not isinstance(prev_results_path, str):
        raise TypeError("prev_results_path must be a string or None")

    updated_dfs: List[pd.DataFrame] = []
    ungrouped_backups: List[pd.DataFrame] = []
    join_keys = [
        "question_id", "gen_modelId", "layer",
        "id", "type", "para_type", "prompt"
    ]

    prev_df = (
        load_previous_labels(prev_results_path, join_keys)
        if prev_results_path else None
    )

    for original_df in df_list:
        df = original_df.copy()

        # Merge existing labels if available
        if prev_df is not None:
            df = merge_previous_labels(
                df, prev_df, join_keys,
                f"mnli_label_{source_col}_to_ref"
            )

        # Identify rows needing processing
        mask = df[f"mnli_label_{source_col}_to_ref"].isna()
        df_to_process = df[mask].copy()

        # Run MNLI on missing rows
        if not df_to_process.empty:
            df_to_process = run_mnli_on_missing(
                df_to_process, declarative_generator, source_col
            )
            df.update(df_to_process)

        # Backup ungrouped results
        ungrouped_backups.append(df.copy())

        # Aggregate and append
        grouped = aggregate_results(df, join_keys, source_col)
        updated_dfs.append(grouped)

        clear_cache()

    return updated_dfs, ungrouped_backups
