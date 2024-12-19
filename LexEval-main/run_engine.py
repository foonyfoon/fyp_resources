from tree.tree import Tree
# from adapters.SemanticAdapter import SemanticAdapter
# from adapters.SyntacticPerturb import SyntacticPerturber
# from model.engine import ClaudeAdapter
from tree.node import RootNode, SyntacticNode, SemanticNode

import os
import json
import re
from typing import List
import logging
from collections import deque
import math

import boto3
import botocore
import pandas as pd
import pickle
import textstat
import numpy as np
import matplotlib.pyplot as plt

modelId = 'anthropic.claude-3-haiku-20240307-v1:0'

class Node:
    def __init__(self, value):
        self.value = value
        self.depth = 0
        self.children = []
    
    def __str__(self):
        if self.children is not None:
            return f"{{value: {self.value}, depth: {self.depth}, children:{[str(c) for c in self.children]}}}"
        else:
            return f"{{value: {self.value}, depth: {self.depth}, children:[]}}"
        
    def add_children(self, children):
        self.children = children
        self.update_depth(self, self.depth)
    
    def update_depth(self, root, root_depth):
        if root is None or root.children is None:
            return 
        else:
            for c in root.children:
                c.depth = root_depth + 1
                self.update_depth(c, root_depth + 1)
           
    def save_node(self, filepath):
        node = {
            "children": self.children,
            "value": self.value,
        }
        dir = os.path.dirname(filepath)
        if not os.path.exists(dir):
            os.makedirs(dir) 
        with open(filepath, "wb") as file:
            pickle.dump(node, file)
    
    @staticmethod
    def load_node(filepath):
        state = {}
        with open(filepath, "rb") as file:
            state = pickle.load(file)
        node = Node(state["value"])
        node.add_children(state["children"])
        node.update_depth(node, node.depth)
        return node
    
def test_store_node():
    filepath = "/vol/bitbucket/lst20/treenodes/node.pkl"
    node = Node(0)
    b1 = Node(1)
    b2 = Node(2)
    node.add_children([b1, b2])
    b1.add_children([Node(3),Node(4)])
    b1.add_children([Node(4)])
    b2.add_children([Node(5),Node(6)])
    print(str(node))
    node.save_node(filepath)
    new_node = Node.load_node(filepath)
    print(str(new_node))

    state = {}
    with open("/vol/bitbucket/lst20/treenodes/2_1_1/0.pkl", "rb") as file:
        state = pickle.load(file)
    node = state["root"].to_dict()
    print(node)


MAX_RETRY = 5
def sem_perturb_combined(model, user_prompt, num_semantic) -> List[str]:
        prompt = (
            f"Generate variations of the following user prompt "
            f"by perturbing its semantics while preserving its core intent. Aim to create the most diverse "
            f"question set.  Respond with {num_semantic} perturbation(s) in json format. For example, for the question"
            f"When was Mozart born, the response for 2 perterbation is as follows: "
        )

        prompt += '{"perturbations": ["Care to share the birthdate of Mozart?", "Can you tell me the date when Mozart was born?"]}'
        prompt = model.format_prompt(
            user_prompt,
            state=[{
                    "role": "system",
                    "content":prompt
                }]
        )
        for retry_count in range(1, MAX_RETRY + 1):
            response, _ = model.complete(prompt)
            try:
                # Regex to find the JSON block
                json_match = re.search(r'{(.*)?}', response, re.DOTALL)
                if not json_match:
                    logging.info(f"No JSON object found in the provided text. retry: {retry_count}")
                json_data = json.loads(json_match.group())
                if "perturbations" not in json_data:
                    logging.info(f"The JSON does not contain the key 'perturbations'. retry: {retry_count}")
                perturbations = json_data["perturbations"]
                if len(perturbations) != num_semantic:
                    logging.info(f"length of perterbations is {len(perturbations)} instead of {num_semantic}. retry: {retry_count}")
                else:
                    return perturbations
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error encountered: {e}. retry: {retry_count}")
        
        raise RuntimeError(f"sem_perturb_combined: could not generate perturbations after {MAX_RETRY} retries")

def test_preturbation_format():   
    modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
    model = ClaudeAdapter(modelId)
    user_prompt = "Who are the member in the band the Beetles?"
    preturb_list = sem_perturb_combined(model, user_prompt, 3)
    print(preturb_list)

        
def format_prompt(utterance: str, state= None, role='user', **kwargs):
        instr = ""
        if state is not None:
            instr += "<instruction> \n"
            instr += "\n".join(f"{i + 1}. {s['content']}" for i, s in enumerate(state))
            instr += "\n</instruction>\n"
        instr += f"\n<question>\n{utterance}\n</question>"
        return instr


def sem_perturb(user_prompt):
        prompt = (
            f"Generate variations of the following user prompt "
            f"by perturbing its semantics while preserving its core intent. Aim to create the most diverse "
            f"question set.  Respond with 3 perturbation(s) in json format. For example, for the question"
            f"When was Mozart born, the response for 2 perterbation is as follows: "
        )

        prompt += '{"perturbations": ["Care to share the birthdate of Mozart?", "Can you tell me the date when Mozart was born?"]}'
        prompt = format_prompt(
            user_prompt,
            state=[{
                    "role": "system",
                    "content":prompt
                }]
        )
        print(prompt)
        
def format_wiki_answer(prompt: str, title: str, extracts: str) -> str:
    extracts = extracts.replace("{", "")
    extracts = extracts.replace("}", "")
    text = " ".join(
        [f"You are a helpful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous,",
        f"or illegal content. You have retrieved the following extracts from the Wikipedia page {title}: {extracts}.\nYou are expected to give ",
        f"truthful and concise answers based on the previous extracts. If it doesn't include relevant information for the request just say so ",
        f"and don't make up false information. \n",
        f"Keep the answers as concise as possible, does not have to be full sentences. For example, for the question: What is Scooter Braun's occupation? Your response should be:",
        f"Talent manager, Entrepreneur, Record executive, Film and television producer.\n"])
    text = text.format(prompt=prompt, title=title, extracts=extracts)
    return text

def wiki_rag_completions(model_name, system_prompt, user_prompt):
    prompt=format_prompt(
        user_prompt,
        state=[{
                "role": "system",
                "content": system_prompt,
            }]
        )
    print(prompt)
    
def test_prompt():
    prompt = "What was the primary vocation of Edgar Allan Poe"
    input_text = format_wiki_answer(
        prompt="What was the primary vocation of Edgar Allan Poe?",
        title="Loss of Breath",
        extracts="Loss of Breath, also known as Loss of Breath: A Tale Neither in Nor Out of \'Blackwood\'\u2009",
    )
    print(input_text)
    wiki_rag_completions("model", input_text, prompt)
    sem_perturb(prompt)

def add_sem_complexity(node):
    if type(node) == RootNode or type(node) == SemanticNode:
        # Calculate Flesch-Kincaid Grade Level and Dale-Chall Readability Score
        fk_score = textstat.flesch_kincaid_grade(node.prompt)
        dc_score = textstat.dale_chall_readability_score(node.prompt)
        complexity_score = (fk_score + dc_score) / 2
        node.complexity_score = complexity_score
        node.fk_score = fk_score
        node.dc_score = dc_score
    for child in node.children:
        add_sem_complexity(child)
                

def count_sem_complexity(node):
    count = 0
    if type(node) == RootNode or type(node) == SemanticNode:
        if node.complexity_score == 0:
            count += 1
    for child in node.children:
        count += count_sem_complexity(child)
    return count
        
def count_no_complexity():
    print("start!")
    modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
    model = ClaudeAdapter(modelId)
    semantic_adapter = SemanticAdapter(model)
    syntactic_adapter = SyntacticPerturber()
    for i in range(30):
        treepath = f"/vol/bitbucket/lst20/treenodes/claude_2_2_1/complete/{i}_checked.pkl"
        test_tree = Tree.load_tree(treepath, semantic_adapter, syntactic_adapter)
        # recalculate complexity for all tree
        root = test_tree.root
        print(count_sem_complexity(root))
        print(f"tree {i} done")

def process_node(node, model_name, possible_answers):
        # This method contains the code to process a single node.
        response = node.answers[model_name]["base"]
        base_rag_response = node.answers[model_name]["base_rag"]
        bm25_rag_response = node.answers[model_name]["bm25_rag"]
        contriever_response = node.answers[model_name]["contriever_rag"]

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        base_true_positives = 0
        base_false_positives = 0
        base_false_negatives = 0

        bm25_true_positives = 0
        bm25_false_positives = 0
        bm25_false_negatives = 0

        cont_true_positives = 0
        cont_false_positives = 0
        cont_false_negatives = 0

        found_match = False
        rag_found_match = False
        bm25_found_match = False
        contr_found_match = False

        for expected_answer in json.loads(possible_answers):
            if response.__contains__(expected_answer):
                found_match = True
                true_positives += 1
                break
        if not found_match:
            false_positives += 1

        for expected_answer in json.loads(possible_answers):
            if base_rag_response.__contains__(expected_answer):
                rag_found_match = True
                base_true_positives += 1
                break
        if not rag_found_match:
            base_false_positives += 1

        for expected_answer in json.loads(possible_answers):
            if bm25_rag_response.__contains__(expected_answer):
                bm25_found_match = True
                bm25_true_positives += 1
                break
        if not bm25_found_match:
            bm25_false_positives += 1

        for expected_answer in json.loads(possible_answers):
            if contriever_response.__contains__(expected_answer):
                contr_found_match = True
                cont_true_positives += 1
                break
        if not contr_found_match:
            cont_false_positives += 1

        if not found_match:
            false_negatives += 1
        if not rag_found_match:
            base_false_negatives += 1
        if not bm25_found_match:
            bm25_false_negatives += 1
        if not contr_found_match:
            cont_false_negatives += 1

        values = {
            "metric":{
                "base": {
                    "true_pos": true_positives,
                    "false_pos": false_positives,
                    "false_neg": false_negatives,
                },
                "base_rag": {
                    "true_pos": base_true_positives,
                    "false_pos": base_false_positives,
                    "false_neg": base_false_negatives,
                },
                "bm25_rag": {
                    "true_pos": bm25_true_positives,
                    "false_pos": bm25_false_positives,
                    "false_neg": bm25_false_negatives,
                },
                "contriever_rag": {
                    "true_pos": cont_true_positives,
                    "false_pos": cont_false_positives,
                    "false_neg": cont_false_negatives,
                },
            }
        }
        if isinstance(node, (RootNode, SemanticNode)):
            node_semantic_complexity = {
                "complexity_score": node.complexity_score,
                "fk_score": node.fk_score,
                "dc_score": node.dc_score,
            }
            values["semantic_scores"] = node_semantic_complexity

        return node.answers[model_name], values

def read_node_stat(n):
    fk_score_metrics = []
    dc_score_metrics = []
    complexity_score_metrics = []
    df_location = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/test.csv"
    df = pd.read_csv(df_location)
    for i, row in df.iterrows():
        if i > n:
            break
        possible_answers = row["possible_answers"]
        treepath = f"/vol/bitbucket/lst20/treenodes/claude_2_2_1/complete/{i}_checked.pkl"
        test_tree = Tree.load_tree(treepath, None, None)
        # recalculate complexity for all tree
        root = test_tree.root
        queue = deque([root])
        visited = {root}
        count = 0
        while queue:
            node = queue.popleft()
            print(f"new node! {count}")
            count += 1
            # process node
            ans, metric = process_node(node, modelId, possible_answers)
            if isinstance(node, (RootNode, SemanticNode)):
                fk, dc, comp = metric["semantic_scores"]["fk_score"], metric["semantic_scores"]["dc_score"], metric["semantic_scores"]["complexity_score"]
                fk_score_metrics.append({"score": fk, "metric": metric["metric"]})
                dc_score_metrics.append({"score": dc, "metric": metric["metric"]})
                complexity_score_metrics.append({"score": comp, "metric": metric["metric"]})
            for child in node.children:
                if child not in visited:
                    queue.append(child)
                    visited.add(child)
    return fk_score_metrics, dc_score_metrics, complexity_score_metrics

def get_plot_data(score_metric_list):
    # separate into bucket
    ret = []
    bucket_size = 0.5
    for score in score_metric_list:
        sorted_data = sorted(score, key=lambda x: x["score"])
        min_value = math.floor(sorted_data[0]["score"])
        max_value = math.ceil(sorted_data[-1]["score"])
        buckets = np.arange(min_value, max_value, bucket_size)
        bucket_count = buckets.shape[0]
        bucket_metric = []
        for i in range(bucket_count):
            bucket_metric.append({
                "score": min_value + i * bucket_size,
                "metric": {
                    "base": {
                        "true_pos": 0,
                        "false_pos": 0,
                        "false_neg": 0,
                    },
                    "base_rag": {
                        "true_pos": 0,
                        "false_pos": 0,
                        "false_neg": 0,
                    },
                    "bm25_rag": {
                        "true_pos": 0,
                        "false_pos": 0,
                        "false_neg": 0,
                    },
                    "contriever_rag": {
                        "true_pos": 0,
                        "false_pos": 0,
                        "false_neg": 0,
                    },
                }})
        for d in sorted_data:
            idx = int(math.floor((d["score"] - min_value) / bucket_size))
            bucket_metric[idx]["score"] = bucket_size * idx + min_value
            quadrant = ["true_pos", "false_pos", "false_neg"]
            method = ["base", "base_rag", "bm25_rag", "contriever_rag"]
            for q in quadrant:
                for m in method:
                    bucket_metric[idx]["metric"][m][q] += d["metric"][m][q]   
        ret.append(bucket_metric)
    return ret

def plot(data, name):
    # Extract scores and values for plotting
    scores = [entry['score'] for entry in data]
    categories = ['base', 'base_rag', 'bm25_rag']
    true_pos = [[entry["metric"][category]['true_pos'] for category in categories] for entry in data]
    false_pos = [[entry["metric"][category]['false_pos'] for category in categories] for entry in data]
    false_neg = [[entry["metric"][category]['false_neg'] for category in categories] for entry in data]

    # Set bar positions
    x = np.arange(len(scores))  # The x locations for the groups
    width = 0.2  # The width of each bar

    # Create the bar chart
    fig, ax = plt.subplots()
    plt.figure(figsize=(30, 6))

    # Create sub-groups of bars (3 categories per score)
    for i, score in enumerate(scores):
        ax.bar(x[i] - width, true_pos[i], width, label=f"True Pos {score}", color='green')
        ax.bar(x[i], false_pos[i], width, label=f"False Pos {score}", color='red')
        ax.bar(x[i] + width, false_neg[i], width, label=f"False Neg {score}", color='blue')

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Scores')
    ax.set_ylabel('Counts')
    ax.set_title('True Positives, False Positives, and False Negatives by Score and Category')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{score:.2f}' for score in scores])  # Format score as float with 2 decimal places
    ax.legend()

    # Save the plot
    save_path = os.path.join("/homes/lst20/fyp/fyp_resources/LexEval-main/output", f'{name}_score_analysis_plot.png')
    plt.savefig(save_path)
           
stat = read_node_stat(20)
data = get_plot_data(stat)
scores = ["fk", "dc", "complexity"]
for i, s in enumerate(scores):
    plot(data[i], s)

