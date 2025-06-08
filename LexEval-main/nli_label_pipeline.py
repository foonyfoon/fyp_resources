import csv
import pandas as pd
import transformers
from contextlib import contextmanager
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
import re, ast, os
from collections import defaultdict
from spacy.lang.en import English

# (Include all the function definitions from your original code here: clear_cache, get_qnli, get_mnli, format_inputs, evaluate_pipeline)

# Add back the function definitions that were in the original code block
def clear_cache():
    """ Free CUDA & Python memory. """
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
        "textattack/roberta-base-QNLI",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-QNLI")
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=1024,
        return_all_scores=True
    )
    try:
        yield pipe
    finally:
        del pipe
        torch.cuda.empty_cache()


@contextmanager
def get_mnli():
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-mnli",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokenizer.padding_side = "right"
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
        torch.cuda.empty_cache()


def format_inputs(example):
    prompt_text = example["root_prompt"]
    ans_text = example["possible_answers"]
    example["input_text"] =  f"{ans_text} \n {prompt_text}"
    return example


def evaluate_pipeline(
    df_list,
    declarative_generator,
    source_col: str = "base_rag"
):
    """
    df_list: list of pandas.DataFrame, each containing at least:
             ['question_id', 'prompt', 'type', 'metadata', source_col, 'possible_answers', ...]
    declarative_generator: a HF text‐generation pipeline that takes a list of strings and returns
                           [{'generated_text': ...}, ...].
    source_col: name of the column you want to treat as your “input sentences” for QNLI/MNLI.
                Default is "base_rag". If you want to run exactly the same steps on column "base",
                call evaluate_pipeline(..., source_col="base").
    """
    nlp = English()
    nlp.add_pipe("sentencizer")

    def extract_sentences(text: str):
        sentences = []
        for segment in text.split("\n"):
            doc = nlp(segment.strip())
            for sent in doc.sents:
                cleaned = sent.text.strip()
                if re.search(r"\w", cleaned):
                    sentences.append(cleaned)
        return sentences

    updated_dfs = []

    for original_df in df_list:
        # Inspect the initial DataFrame

        # —— Step 0: Compute 'root_prompt' exactly as before ——
        df = original_df.copy()
        df.loc[:, "root_prompt"] = df.apply(
            lambda row: row["prompt"]
                        if row["type"] == "RootNode"
                        else row["metadata"].get("root_prompt")
                        if isinstance(row["metadata"], dict)
                        else None,
            axis=1
        )
        df = df.dropna(subset=["root_prompt"]).copy()

        # Inspect DataFrame after Step 0
        print("DataFrame shape after Step 0 (dropna on root_prompt):", df.shape)
        if df.empty:
             print("DataFrame is empty after Step 0. Check data and root_prompt logic.")
             updated_dfs.append(pd.DataFrame()) # Append empty df or skip to next original_df
             continue # Skip the rest of the processing for this empty df

        # —— Step 1: Declarative Generation of reference answers ——
        # 1a) Keep only the first occurrence of each root_prompt
        first_occ = df.drop_duplicates(subset="root_prompt").copy()

        # Inspect first_occ after drop_duplicates
        print("DataFrame shape after drop_duplicates on root_prompt:", first_occ.shape)
        if first_occ.empty:
             print("first_occ DataFrame is empty after drop_duplicates. Check data and drop_duplicates logic.")
             updated_dfs.append(pd.DataFrame()) # Append empty df or skip to next original_df
             continue # Skip the rest of the processing for this empty df


        # 1b) Convert any string‐like‐list into a real python list
        def parse_to_list(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, str):
                try:
                    parsed = ast.literal_eval(x)
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    return [x]
            else:
                return [x]

        first_occ["possible_answers"] = first_occ["possible_answers"].apply(parse_to_list)

        # 1c) Flatten one level if nested lists
        def flatten_once(cell):
            if isinstance(cell, list) and len(cell) > 0 and all(isinstance(el, list) for el in cell):
                return cell[0]
            return cell

        first_occ["possible_answers"] = first_occ["possible_answers"].apply(flatten_once)

        # 1d) Explode so each row has exactly one possible_answers
        first_occ = first_occ.explode("possible_answers").reset_index(drop=True)
        # Print number of rows after explode
        print("Number of rows in first_occ after explode:", len(first_occ))
        # Print the number of rows before creating the dataset (already added)
        print("Number of rows in first_occ before creating Dataset:", len(first_occ))


        # 2) Build a small HF Dataset to generate declaratives for the reference answers
        # Add a check here as well
        if first_occ.empty:
             print("first_occ is empty, cannot create Dataset.")
             updated_dfs.append(pd.DataFrame()) # Append empty df or skip to next original_df
             continue # Skip to next original_df

        ds = Dataset.from_pandas(
            first_occ[["question_id", "root_prompt", "possible_answers"]],
            preserve_index=False
        )

        # Keep the print statement after map as well
        print("Columns after map:", ds.column_names)

        ds = ds.map(format_inputs)

        # Extract the list of strings from the 'input_text' column AFTER adding it with map
        input_texts_for_generator = ds["input_text"] # This extracts the list of values


        # Pass the list of strings to the declarative generator
        gens = declarative_generator(input_texts_for_generator)

        # 3) Apply fallback logic
        declaratives = []
        # Need to use the original inputs from the ds object here for fallback logic
        original_inputs_for_fallback = ds["input_text"]
        for g, orig in zip(gens, original_inputs_for_fallback):
            gen_text = g["generated_text"].strip()
            if re.search(r"\w", gen_text):
                declaratives.append(gen_text)
            else:
                answer_part = orig.split("\n", 1)[0].strip()
                declaratives.append(answer_part if answer_part else None)
        first_occ.loc[:, "reference_ans_declarative"] = declaratives

        # 4) Merge back to the full df on question_id
        df = original_df.merge(
            first_occ[["question_id", "reference_ans_declarative", 'root_prompt']],
            on="question_id",
            how="left"
        )
        clear_cache()

        # —— Step 2: QNLI over sentences extracted from `source_col` ——
        with get_qnli() as qnli_pipe:
            # We'll collect for each row: list of sentences, entailment probs, labels, etc.
            source_sentences_col = []
            entail_probs_col = []
            entail_labels_col = []
            num_entail_col = []
            num_sent_col = []

            all_examples = []
            example_index = []

            for idx, row in df.iterrows():
                text_blob = row.get(source_col, "")
                sentences = extract_sentences(text_blob)
                source_sentences_col.append(sentences)
                num_sent_col.append(len(sentences))

                # Build a QNLI example for each sentence
                for sent in sentences:
                    all_examples.append(
                        {"text": str(row["root_prompt"]), "text_pair": str(sent)}
                    )
                    example_index.append(idx)

            # Run QNLI in a single batch
            qnli_results = qnli_pipe(all_examples)

            # Initialize placeholders
            entail_probs_by_row = {i: [] for i in df.index}
            labels_by_row = {i: [] for i in df.index}
            count_by_row = {i: 0 for i in df.index}

            for i, out_list in enumerate(qnli_results):
                idx = example_index[i]
                # Choose the label with highest score
                chosen = max(out_list, key=lambda d: d["score"])["label"]
                scores = {d["label"]: d["score"] for d in out_list}
                # In QNLI: LABEL_0 = entailment, LABEL_1 = not entailment
                probs = [scores["LABEL_1"], scores["LABEL_0"]]
                labels_by_row[idx].append("entailment" if chosen == "LABEL_0" else "not entailment")
                entail_probs_by_row[idx].append(probs)
                if chosen == "LABEL_0":
                    count_by_row[idx] += 1

            for idx in df.index:
                entail_probs_col.append(entail_probs_by_row[idx])
                entail_labels_col.append(labels_by_row[idx])
                num_entail_col.append(count_by_row[idx])

            df.loc[:, f"{source_col}_sentences"] = source_sentences_col
            df.loc[:, f"{source_col}_entailment_probs"] = entail_probs_col
            df.loc[:, f"{source_col}_entailment_labels"] = entail_labels_col
            df.loc[:, f"num_entailments_{source_col}"] = num_entail_col
            df.loc[:, f"num_{source_col}_sentences"] = num_sent_col

            # Build the “best” sentence per row (first one labeled “entailment”)
            def best_entailing(row):
                for sent, lab in zip(row[f"{source_col}_sentences"], row[f"{source_col}_entailment_labels"]):
                    if lab == "entailment":
                        return sent
                return None

            df.loc[:, f"best_{source_col}_sentence"] = df.apply(best_entailing, axis=1)

        clear_cache()

        # —— Step 3: MNLI — comparing declaratives derived from `best_{source_col}_sentence` to the reference ——
        df = df.copy()
        with get_mnli() as mnli_pipe:
            # Build inputs: “<best_sentence> \n <root_prompt>”
            inputs = []
            for _, row in df.iterrows():
                sent = row.get(f"best_{source_col}_sentence", "")
                prompt = row["root_prompt"]
                if isinstance(sent, str) and sent.strip():
                    inputs.append(f"{sent.strip()} \n {prompt}")
                else:
                    inputs.append("")

            # 1) Generate declaratives for those combined strings
            gens = declarative_generator(inputs)

            # 2) Fallback logic
            declar_col = []
            for gdict, inp in zip(gens, inputs):
                gen_text = gdict["generated_text"].strip()
                if re.search(r"\w", gen_text):
                    declar_col.append(gen_text)
                else:
                    first_part = inp.split("\n", 1)[0].strip()
                    declar_col.append(first_part if re.search(r"\w", first_part) else None)

            df.loc[:, f"{source_col}_declarative"] = declar_col

            # 3) Build MNLI examples: premise = <source_declarative>, hypothesis = reference_ans_declarative
            mnli_examples = []
            for _, row in df.iterrows():
                raw_premise = row.get(f"{source_col}_declarative", "")
                premise = " ".join(raw_premise) if isinstance(raw_premise, list) else (raw_premise or "")
                raw_hypothesis = row.get("reference_ans_declarative", "")
                hypothesis = (
                    " ".join(raw_hypothesis)
                    if isinstance(raw_hypothesis, list)
                    else (raw_hypothesis or "")
                )
                mnli_examples.append({
                    "text": str(premise),
                    "text_pair": str(hypothesis)
                })

            print(f"Running MNLI on {len(mnli_examples)} examples (source_col = {source_col})")
            print("Sample examples:", mnli_examples[:5])

            mnli_results = mnli_pipe(mnli_examples)

            best_labels = []
            all_probs = []
            for out in mnli_results:
                best = max(out, key=lambda d: d["score"])
                best_labels.append(best["label"])
                scores_dict = {d["label"]: d["score"] for d in out}
                all_probs.append([
                    scores_dict.get("entailment",   0.0),
                    scores_dict.get("contradiction", 0.0),
                    scores_dict.get("neutral",       0.0)
                ])

            df.loc[:, f"mnli_label_{source_col}_to_ref"] = best_labels
            df.loc[:, f"mnli_probs_{source_col}_to_ref"] = all_probs

        clear_cache()

        # Convert MNLI labels to binary (entailment=1, else=0)
        df[f"mnli_label_{source_col}_to_ref"] = df[f"mnli_label_{source_col}_to_ref"].apply(
            lambda x: 1 if x == "entailment" else 0
        )

        # Drop the helper column if you like:
        #  df = df.drop(columns=["reference_ans_declarative"])

        # —— Step 4: Group by multiple keys: question_id, gen_modelId, layer, id, type, para_type, prompt
        group_keys = [
            "question_id",
            "gen_modelId",
            "layer",
            "id",
            "type",
            "para_type",
            "prompt"
        ]

        # Build an aggregation dict: for each column not in group_keys
        # • if it’s the MNLI label, take max
        # • otherwise, take first
        df_columns = list(df.columns)
        aggregation = {}
        for col in df_columns:
            if col in group_keys:
                continue
            elif col == f"mnli_label_{source_col}_to_ref":
                aggregation[col] = "max"
            else:
                aggregation[col] = "first"

        df_grouped = df.groupby(group_keys).agg(aggregation).reset_index()

        # Flatten single‐element arrays back into scalars where it makes sense
        for col in df_grouped.columns:
            if col in group_keys or col == f"mnli_label_{source_col}_to_ref":
                continue
            df_grouped[col] = df_grouped[col].apply(
                lambda arr: arr[0] if isinstance(arr, (list, np.ndarray, pd.Series)) and len(arr) == 1 else arr
            )

        updated_dfs.append(df_grouped)

    clear_cache()
    return updated_dfs