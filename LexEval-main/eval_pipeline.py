import os
import gc
import re
import ast
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from spacy.lang.en import English

def clear_cache():
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
def get_qnli():
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/roberta-base-QNLI", padding="max_length",
        truncation=True, max_length=512
    )
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/roberta-base-QNLI"
    )
    pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        batch_size=1024, return_all_scores=True
    )
    try:
        yield pipe
    finally:
        del pipe
        torch.cuda.empty_cache()

@contextmanager
def get_mnli():
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-mnli", padding="max_length",
        truncation=True, max_length=512
    )
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    )
    pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        batch_size=16, return_all_scores=True
    )
    try:
        yield pipe
    finally:
        del pipe
        torch.cuda.empty_cache()

# Step 0: get root prompt
def derive_root_prompt(df: pd.DataFrame) -> pd.DataFrame:
    # Build map: question_id -> prompt
    root_map = (
        df.loc[df['type'] == 'RootNode', ['question_id', 'prompt']]
          .drop_duplicates()
          .set_index('question_id')['prompt']
          .to_dict()
    )
    df = df.copy()
    df['root_prompt'] = df['question_id'].map(root_map)
    if df['root_prompt'].isnull().any():
        missing = df.loc[df['root_prompt'].isnull(), 'question_id'].unique()
        raise ValueError(f"Missing RootNode prompt for question_ids: {missing}")
    return df

# Step 1: reference answer declarative generation
def prepare_reference_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe & explode possible_answers."""
    first = df.drop_duplicates(subset=['root_prompt']).copy()
    def parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                lst = ast.literal_eval(x)
            except Exception:
                return [x]
            return lst if isinstance(lst, list) else [lst]
        return [x]
    first['possible_answers'] = first['possible_answers'].apply(parse)
    def flatten_once(cell):
        if isinstance(cell, list) and cell and all(isinstance(el, list) for el in cell):
            return cell[0]
        return cell
    first['possible_answers'] = first['possible_answers'].apply(flatten_once)
    return first.explode('possible_answers').reset_index(drop=True)


def format_inputs(example: dict) -> dict:
    example['input_text'] = f"{example['possible_answers']} \n {example['root_prompt']}"
    return example


def generate_reference_declaratives(
    df: pd.DataFrame,
    declarative_generator
) -> pd.DataFrame:
    ds = Dataset.from_pandas(
        df[['question_id', 'root_prompt', 'possible_answers']],
        preserve_index=False
    )
    ds = ds.map(format_inputs)
    gens = declarative_generator(ds['input_text'])
    declaratives = []
    for g, inp in zip(gens, ds['input_text']):
        txt = g['generated_text'].strip()
        is_valid = bool(re.search(r"\w", txt)) and txt.lower() != 'nan'
        declaratives.append(
            txt if is_valid
            else inp.split("\n", 1)[0].strip()
        )
    df['reference_ans_declarative'] = declaratives
    return df


# Step 2: QNLI annotation
def extract_sentences(text: str, nlp) -> list:
    sentences = []
    for segment in text.split("\n"):
        doc = nlp(segment.strip())
        sentences.extend([
            s.text.strip() for s in doc.sents if re.search(r"\w", s.text)
        ])
    return sentences


def annotate_qnli(df: pd.DataFrame, source_col: str, qnli_pipe) -> pd.DataFrame:
    df = df.copy()
    nlp = English()
    nlp.add_pipe('sentencizer')
    all_ex, idx_map = [], []
    sent_list = []
    for i, row in df.iterrows():
        sents = extract_sentences(row.get(source_col, ''), nlp)
        sent_list.append(sents)
        for s in sents:
            # Prepare inputs for the pipeline
            all_ex.append({'text': row['root_prompt'], 'text_pair': s})
            idx_map.append(i)

    res = qnli_pipe(all_ex, padding='longest', truncation=True)

    probs, labels, counts = (
        {i: [] for i in df.index},
        {i: [] for i in df.index},
        {i: 0 for i in df.index}
    )
    for out, i in zip(res, idx_map):
        choice = max(out, key=lambda d: d['score'])
        # The LABEL_0 is entailment for this specific QNLI model
        lbl = 'entailment' if choice['label'] == 'LABEL_0' else 'not entailment'
        sc = {d['label']: d['score'] for d in out}
        labels[i].append(lbl)
        probs[i].append([sc.get('LABEL_1', 0), sc.get('LABEL_0', 0)])
        if lbl == 'entailment':
            counts[i] += 1

    df[f'{source_col}_sentences'] = sent_list
    df[f'{source_col}_entailment_labels'] = pd.Series(
        [labels[i] for i in range(len(df))]
    )
    df[f'{source_col}_entailment_probs'] = pd.Series(
        [probs[i] for i in range(len(df))]
    )
    df[f'num_entailments_{source_col}'] = [counts[i] for i in range(len(df))]
    df[f'num_{source_col}_sentences'] = df[f'{source_col}_sentences'].apply(len)
    df[f'best_{source_col}_sentence'] = df.apply(
        lambda r: next(
            (s for s, l in zip(
                r[f'{source_col}_sentences'],
                r[f'{source_col}_entailment_labels']
            ) if l == 'entailment'), None
        ),
        axis=1
    )
    return df

# Step 3: MNLI annotation
def annotate_mnli(
    df: pd.DataFrame,
    source_col: str,
    declarative_generator,
    mnli_pipe
) -> pd.DataFrame:
    df = df.copy()
    inputs = []
    for _, r in df.iterrows():
        best = r.get(f'best_{source_col}_sentence') or ''
        prompt = r['root_prompt'] or ''
        inputs.append(f"{best.strip()} \n {prompt.strip()}" if best or prompt else '')
    gens = declarative_generator(inputs)
    decls = []
    for g, inp in zip(gens, inputs):
        txt = g['generated_text'].strip()
        decls.append(
            txt if re.search(r"\w", txt)
            else inp.split("\n", 1)[0].strip() if "\n" in inp else inp.strip()
        )
    df[f'{source_col}_declarative'] = decls
    valid_idx, mnli_ex = [], []
    for i, (p, r) in enumerate(
        zip(df[f'{source_col}_declarative'], df['reference_ans_declarative'])
    ):
        if isinstance(p, str) and p and isinstance(r, str) and r:
            mnli_ex.append({'text': p, 'text_pair': r})
            valid_idx.append(i)
    if not mnli_ex:
        df[f'mnli_label_{source_col}_to_ref'] = [0] * len(df)
        df[f'mnli_probs_{source_col}_to_ref'] = [[0.0, 0.0, 0.0]] * len(df)
        return df
    res = mnli_pipe(mnli_ex)
    full_lbls = [0] * len(df)
    full_pr = [[0.0, 0.0, 0.0]] * len(df)
    res_i = 0
    for idx in valid_idx:
        out = res[res_i]
        best = max(out, key=lambda d: d['score'])
        full_lbls[idx] = 1 if best['label'] == 'entailment' else 0
        sm = {d['label']: d['score'] for d in out}
        full_pr[idx] = [
            sm.get('entailment', 0),
            sm.get('contradiction', 0),
            sm.get('neutral', 0)
        ]
        res_i += 1
    df[f'mnli_label_{source_col}_to_ref'] = full_lbls
    df[f'mnli_probs_{source_col}_to_ref'] = full_pr
    return df


# Step 4: Aggregation
def group_and_finalize(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    question_keys = ['question_id', 'gen_modelId', 'layer', 'id', 'type', 'para_type', 'prompt']
    agg_ops = {
        c: ('max' if c == f'mnli_label_{source_col}_to_ref' else 'first')
        for c in df.columns if c not in question_keys
    }
    grouped = df.groupby(question_keys).agg(agg_ops).reset_index()
    for c in grouped.columns:
        if c not in question_keys and c != f'mnli_label_{source_col}_to_ref':
            grouped[c] = grouped[c].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray, pd.Series)) and len(x) == 1 else x
            )
    return grouped


def evaluate_pipeline(
    df_list: list,
    declarative_generator,
    source_cols: list = ['base_rag', 'base'],
    previous_output_paths: dict = None,
    output_paths: dict = None
) -> dict:
    results = {}
    backup_dir = 'backup'
    os.makedirs(backup_dir, exist_ok=True)
    question_keys = ['question_id', 'gen_modelId', 'layer', 'id', 'type', 'para_type', 'prompt']

    prev_data = {}
    if previous_output_paths:
        for col, path in previous_output_paths.items():
            prev_df = pd.read_csv(path)
            if prev_df.duplicated(subset=question_keys).any():
                raise ValueError(f"Duplicate keys in previous output for {col}")
            prev_data[col] = prev_df

    with get_qnli() as qnli_pipe, get_mnli() as mnli_pipe:
        for col in source_cols:
            prev_df = prev_data.get(col, pd.DataFrame())
            processed_keys = set(
                tuple(x) for x in prev_df[question_keys].values
            ) if not prev_df.empty else set()

            new_groups = []
            saved_backup = False
            for df in df_list:
                mask = df[question_keys].apply(
                    lambda row: tuple(row) not in processed_keys, axis=1
                )
                df_filtered = df[mask]

                # Derive root_prompt for all remaining rows
                df0 = derive_root_prompt(df_filtered)

                if df0.empty:
                    continue

                # Steps 1-3
                ref_df = prepare_reference_dataset(df0)
                ref_df = generate_reference_declaratives(ref_df, declarative_generator)
                merged = df0.merge(
                    ref_df[['question_id',
                             'reference_ans_declarative']],
                    on='question_id', how='left'
                )
                clear_cache()
                qnli_df = annotate_qnli(merged, col, qnli_pipe)
                clear_cache()
                mnli_df = annotate_mnli(
                    qnli_df, col, declarative_generator, mnli_pipe
                )
                clear_cache()

                if not saved_backup:
                    backup_path = os.path.join(
                        backup_dir, f'ungrouped_backup_{col}.csv'
                    )
                    mnli_df.to_csv(backup_path, index=False)
                    saved_backup = True

                grp = group_and_finalize(mnli_df, col)
                new_groups.append(grp)

            combined = (
                pd.concat([prev_df] + new_groups, ignore_index=True)
                if not prev_df.empty else
                pd.concat(new_groups, ignore_index=True)
                if new_groups else pd.DataFrame()
            )

            if output_paths and col in output_paths:
                combined.to_csv(output_paths[col], index=False)

            results[col] = combined
    return results

declarative_generator = pipeline(
    "text2text-generation",
    model='khhuang/zerofec-qa2claim-t5-base',
    tokenizer='khhuang/zerofec-qa2claim-t5-base',
    batch_size=64,
    return_tensors=False
)


# 1. Load your DataFrame from CSV
df = pd.read_csv("/content/drive/MyDrive/fyp/dataset/POPQA-prefix-trees.csv")

# 3. Call the pipeline on a list of DataFrames (here, a single df)

results = evaluate_pipeline(
    [df],
    declarative_generator,
    source_cols=['base_rag', 'base'],
    previous_output_paths=None,
    output_paths=None  # individual outputs not used now
)

# Merge base_rag and base results on question-level keys
question_keys = ['question_id', 'gen_modelId', 'layer', 'id', 'type', 'para_type', 'prompt']
df_br = results.get('base_rag', pd.DataFrame())
df_b = results.get('base', pd.DataFrame())
if not df_br.empty and not df_b.empty:
    merged_df = df_br.merge(df_b, on=question_keys, suffixes=('_base_rag', '_base'))
else:
    merged_df = df_br if not df_br.empty else df_b

# Save the merged DataFrame
merged_output_path = '/content/drive/MyDrive/fyp/dataset/POPQA-prefix-trees-merged_last.csv'
merged_df.to_csv(merged_output_path, index=False)
print(f"Saved merged output to {merged_output_path}")
print(merged_df.head())