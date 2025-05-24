import json
from collections import deque
import time
import os
import logging
from abc import ABC, abstractmethod
from typing import Tuple
import logging
import pickle
import io

import networkx as nx
import matplotlib.pyplot as plt
import textstat
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

from adapters.SemanticAdapter import SemanticAdapter
from adapters.SemanticPerturb import SemanticPerturber
from adapters.OAI_Embeddings import EmbedAdapter
from similarity.cosine_similarity import similarity
from utils.timer import Timers
from adapters.prompt_package import PromptPackage
from adapters.rag import RAGAgent
from tree.node import RootNode, SyntacticNode, SemanticNode


class AbstractTree(ABC):
    def print_tree(self, node=None, level=0, model_name=None, truncate_passage=True):
        def _tuncate_wiki_page(page: tuple, max_char=45) -> tuple:
            """
            used only in print_tree() to truncate long title, content and summary of wiki page
            input: (wikipage, sim_score): tuple
            """
            return (
                {
                    "context": (
                        page[0]["context"]
                        if len(page[0]["context"]) <= max_char
                        else page[0]["context"][:max_char] + "..."
                    ),
                    "title": (
                        page[0]["title"]
                        if len(page[0]["title"]) <= max_char
                        else page[0]["title"][:max_char] + "..."
                    ),
                },
                page[1],
            )

        if node is None:
            node = self.root

        indent = "  " * level
        print(f"{indent}{node.id} - {node.prompt} ({node.__class__.__name__})")

        if isinstance(node, (RootNode, SemanticNode)):
            closest_match = (
                list(map(_tuncate_wiki_page, node.rag_closest_match))
                if truncate_passage is True
                else node.rag_closest_match
            )
            print(f"{indent}{node.id} - {node.prompt}:")
            print(f"{indent}- RAG({closest_match})")
            print(f"{indent}- NER({node.rag_entities})")
            print(
                f"{indent}complexity_score: {node.complexity_score}, dc_score: {node.dc_score}, fk_score: {node.fk_score}"
            )
        else:
            print(f"{indent}{node.id} - {node.prompt} - NER({node.rag_entities})")
        for model, answer in node.answers.items():
            print(f"{indent}Answer ({model}): {answer}")

        for child in node.children:
            self.print_tree(child, level + 1, model_name, truncate_passage)


class ReadTree(AbstractTree):
    def __init__(self, root_prompt, prev_state=None):
        self.root_prompt = root_prompt
        self.root = RootNode(root_prompt) if prev_state is None else prev_state["root"]
        self.thresholds = [] if prev_state is None else prev_state["thresholds"]
        self.prompt_list = (
            [root_prompt] if prev_state is None else prev_state["prompt_list"]
        )
        self.time_semantic = 0 if prev_state is None else prev_state["time_semantic"]
        self.time_syntactic = 0 if prev_state is None else prev_state["time_syntactic"]
        self.time_check = {} if prev_state is None else prev_state["time_check"]
        self.metrics = {} if prev_state is None else prev_state["metrics"]
        self.possible_answers = (
            {} if prev_state is None else prev_state["possible_answers"]
        )
        self.rag_entities = [] if prev_state is None else prev_state["rag_entities"]
        self.ner_entities = [] if prev_state is None else prev_state["ner_entities"]
        self.rag_closest_match = (
            [] if prev_state is None else prev_state["rag_closest_match"]
        )
        self.gt_passage = [] if prev_state is None else prev_state["gt_passage"]

    @staticmethod
    def load_read_tree(file_path: str) -> "ReadTree":
        """
        Load a pickled ReadTree, forcing every tensor inside to CPU.
        Works even if the file was written on a CUDA box.
        """

        class _CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    # route all tensor blobs through torch.load(..., map_location='cpu')
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                return super().find_class(module, name)

        with open(file_path, "rb") as fh:
            prev_state = _CPUUnpickler(fh).load()

        return ReadTree(prev_state["root_prompt"], prev_state=prev_state)


class Tree(AbstractTree):
    def __init__(self, root_prompt, **kwargs):
        """
        Parameters expected in kwargs:
            - eval (bool): Whether running in evaluation mode.
            - generator (Optional[GeneratorModel]): Only if eval=True. Model used for answer generation.
            - ner_model (Optional[NERModel]): Named Entity Recognition model (used if eval=False).
            - embedder (EmbedAdapter): Embedding model, used in all modes.
            - sem_perturber (SemanticPerturber): Semantic perturbation model (used if eval=False).
            - syn_perturber (SyntacticPerturber): Syntactic perturbation model (optional, used if eval=False).
            - s_wiki_title (str): wiki_title used when retrieving ground truth passages (used if eval=False).
            - prev_state (Optional[dict]): Previous tree state for resuming/continuing (optional).
        """
        # answer model
        stage = kwargs.get("eval", "sem")
        if stage == "eval" or (isinstance(stage, bool) and stage):
            self.generator = kwargs.get("generator")
            self.rag = RAGAgent(eval=True, generator=self.generator)
            self.base = SemanticAdapter(self.generator)
        elif stage == "sem":
            self.rag = RAGAgent(
                eval=False,
                ner_model=kwargs.get("ner_model"),
                embedder=kwargs.get("embedder"),
            )
            # preturb model
            self.sem_perturber: SemanticPerturber = kwargs.get("sem_perturber")
            s_wiki_title = kwargs.get("s_wiki_title")
            if self.sem_perturber:
                self.sem_perturber.setup_perturber(root_prompt, s_wiki_title)
            self.syn_perturber = kwargs.get("syn_perturber")
            self.embed_model: EmbedAdapter = kwargs.get("embedder")
            self.root.embedding = self.embed_model.encode(root_prompt)
            
        self.root_prompt = root_prompt
        self.num_semantic = 0
        self.num_syntactic = 0

        prev_state = kwargs.get("prev_state", None)
        self.root = RootNode(root_prompt) if prev_state is None else prev_state["root"]

        if prev_state is not None:
            self.rag_entities = prev_state.get("rag_entities")
            self.ner_entities = prev_state.get("ner_entities")
            self.gt_passage = prev_state.get("gt_passage")
            self.root.rag_closest_match = prev_state.get("rag_closest_match")
            self.gt_passage = prev_state.get("gt_passage")
            self.root.wiki_title = prev_state.get("wiki_title")
        elif stage == "sem":
            # Otherwise, run retriever pipeline
            # root
            wiki_data = self.rag.retrieve_wiki_data_2(root_prompt)
            self.gt_passage = self.rag.find_gt_passage(s_wiki_title, root_prompt)
            closest_match = self.construct_retrieved_evidence(0, wiki_data, root_prompt, correct_docs=len(self.gt_passage))
            self.rag_entities = self.rag.search_entities_2(prompt=root_prompt)
            self.ner_entities = self.rag.search_entities_NER(prompt=root_prompt)
            self.root.rag_closest_match = closest_match
            self.root.wiki_title = [w["title"] for w in wiki_data]

        self.thresholds = [] if prev_state is None else prev_state["thresholds"]
        self.prompt_list = (
            [root_prompt] if prev_state is None else prev_state["prompt_list"]
        )
        self.time_semantic = 0 if prev_state is None else prev_state["time_semantic"]
        self.time_syntactic = 0 if prev_state is None else prev_state["time_syntactic"]
        self.time_check = {} if prev_state is None else prev_state["time_check"]
        self.metrics = {} if prev_state is None else prev_state["metrics"]
        self.possible_answers = (
            {} if prev_state is None else prev_state["possible_answers"]
        )

    def set_possible_answers(self, possible_answers):
        self.possible_answers = possible_answers

    def make_tree(self, depth, num_semantic, num_syntactic, index):
        start_time = time.time()
        self.num_semantic = num_semantic
        self.num_syntactic = num_syntactic
        self.thresholds = self.make_thresholds("linear", 0.97, 0.84, depth)

        # Calculate Flesch-Kincaid Grade Level and Dale-Chall Readability Score
        fk_score = textstat.flesch_kincaid_grade(self.root.prompt)
        dc_score = textstat.dale_chall_readability_score(self.root.prompt)
        complexity_score = (fk_score + dc_score) / 2
        self.root.complexity_score = complexity_score
        self.root.fk_score = fk_score
        self.root.dc_score = dc_score

        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()

            if level > depth - 1:
                continue

            upper_thresh = 0.96
            lower_thresh = 0.849

            # generate semantic children
            for _ in range(self.num_semantic):

                Timers.start(index, "create tree", "generate_semantic_node")
                semantic_node, is_valid = self.generate_semantic_node(
                    self.root.prompt, node, upper_thresh, lower_thresh, index
                )
                Timers.end(index, "create tree", "generate_semantic_node")
                # semantic node can have children
                if is_valid:
                    node.add_child(semantic_node)
                    queue.append((semantic_node, level + 1))
            sem_time = time.time()

            # generate syntactic children
            for _ in range(num_syntactic):
                syntactic_node = self.generate_syntactic_node(node)
                node.add_child(syntactic_node)
            syn_time = time.time()

        self.time_semantic = sem_time - start_time
        self.time_syntactic = syn_time - sem_time
        self.sem_perturber.post_process()
        print("Time to create semantic nodes: ", sem_time - start_time)
        print("Time to create syntactic nodes: ", syn_time - sem_time)
        print("Total time: ", syn_time - start_time)

    def construct_retrieved_evidence(
        self, index, wiki_data, perturbation, k_docs=4, correct_docs=1
    ):
        # list of tuple of {title: context} dict and sim_score of doc with prompt
        ground_truth_docs = self.gt_passage
        # Use however many ground truth docs are available.
        extraneous_needed = k_docs - correct_docs
        # filter out wiki data in ground_truth_docs from wiki_data before doing topK
        gt_set = {doc_tuple[0]["title"] for doc_tuple in ground_truth_docs}
        wiki_data = [doc for doc in wiki_data if doc.get("title") not in gt_set]
        Timers.start(index, "create tree", "find_topk_relevant_pages")
        # list of tuple of {title: context} dict and sim_score of doc with prompt
        extraneous_docs = self.rag.find_topk_relevant_pages(
            wiki_data, perturbation, top_k=extraneous_needed
        )
        Timers.start(index, "create tree", "find_topk_relevant_pages")
        combined_docs = ground_truth_docs + extraneous_docs
        return combined_docs

    def generate_syntactic_node(self, node, index):
        syn_perturb = self.syn_perturber.syn_perturb(
            text=node.prompt,
            butterfinger=self.syn_perturber.butterfinger,
        )
        wiki_data = self.rag.retrieve_wiki_data_2(syn_perturb)
        closest_match = self.construct_retrieved_evidence(0, wiki_data, syn_perturb, correct_docs=len(self.gt_passage))
        rag_entities = self.rag.search_entities_2(prompt=syn_perturb)
        ner_entities = self.rag.search_entities_NER(prompt=syn_perturb)
        rag_closest_match = closest_match
        syntactic_node = SyntacticNode(
            syn_perturb,
            0.0,
            "test_context",
            parent=node,
            rag_closest_match=rag_closest_match,
            rag_entities=rag_entities,
            ner_entities=ner_entities,
            wiki_title=[w["title"] for w in wiki_data],
        )
        return syntactic_node
        
    def generate_semantic_node(
        self, root_prompt, parent_node: SemanticNode, upper_thresh, lower_thresh, index
    ) -> Tuple[SemanticNode, bool]:
        """
        Generates a valid semantic perturbation and creates a SemanticNode.
        """

        #####################################################################
        Timers.start(index, "create tree", "prompt_perturbation")
        # if root -> construct new state
        new_state = {
            **parent_node.metadata,             # spreads metadata if any (or does nothing if {})
            "root_prompt": root_prompt,         # always set
            **({"base_prompt": root_prompt}     # only add when metadata is empty
            if not parent_node.metadata
            else {}),
        }
        perturb_pkg = PromptPackage(text=parent_node.prompt, state=new_state)
        new_perturb_pkg: PromptPackage = self.sem_perturber.sem_perturb(perturb_pkg)
        # unpack results
        perturbation = new_perturb_pkg.text  # the new prompt
        perturb_state = new_perturb_pkg.state  # metadata collected by perturbers

        perturb_embedding = self.embed_model.encode(perturbation)
        parent_embedding = parent_node.embedding
        root_embedding = self.root.embedding
        sem_sim = similarity(perturb_embedding, parent_embedding)
        root_sim = similarity(perturb_embedding, root_embedding)

        is_valid = (
            perturbation not in self.prompt_list  # duplicate check
            and perturb_state.get("is_valid", True)
        )  # perturber-level checks passed
        Timers.end(index, "create tree", "prompt_perturbation")
        if is_valid:
            self.prompt_list.append(perturbation)
            #####################################################################
            # Calculate Flesch-Kincaid Grade Level and Dale-Chall Readability Score
            fk_score = textstat.flesch_kincaid_grade(perturbation)
            dc_score = textstat.dale_chall_readability_score(perturbation)
            complexity_score = (fk_score + dc_score) / 2

            Timers.start(index, "create tree", "retrieve_wiki_data_post_perturb")
            wiki_data = self.rag.retrieve_wiki_data_2(perturbation)
            Timers.end(index, "create tree", "retrieve_wiki_data_post_perturb")

            # new: introducing extraneous wiki passage post retrieval from retrieval
            Timers.start(index, "create tree", "construct_retrieved_evidence")
            closest_match = self.construct_retrieved_evidence(
                index, wiki_data, perturbation, correct_docs=len(self.gt_passage)
            )
            Timers.end(index, "create tree", "construct_retrieved_evidence")

            Timers.start(index, "create tree", "search_entities_2")
            rag_entities = self.rag.search_entities_2(prompt=perturbation)
            Timers.end(index, "create tree", "search_entities_2")

            Timers.start(index, "create tree", "search_entities_NER")
            ner_entities = self.rag.search_entities_NER(perturbation)
            Timers.end(index, "create tree", "search_entities_NER")
            sem_node = SemanticNode(
                perturbation,
                sem_sim,
                root_sim,
                lower_thresh,
                self.embed_model.encode(perturbation),
                closest_match,
                rag_entities,
                ner_entities,
                wiki_title=[w["title"] for w in wiki_data],
                parent=parent_node,
                fk_score=fk_score,
                dc_score=dc_score,
                complexity_score=complexity_score,
            )
            # update metadata
            sem_node.metadata.update(perturb_state)
            return (sem_node, is_valid)
        else:
            # invalid semantic node
            logging.error(
                "Encountered invalid perturbation; state=%s, perturbation=%s",
                perturb_state,
                perturbation,
            )
            return (None, is_valid)

    def make_thresholds(self, distribution, upper_bound, lower_bound, depth):
        if distribution == "linear":
            thresholds = [
                upper_bound - i * (upper_bound - lower_bound) / (depth - 1)
                for i in range(depth)
            ]
            return thresholds
        else:
            raise ValueError("Unsupported distribution type")

    def level_weight(self, level):
        if level < len(self.thresholds):
            return self.thresholds[level]
        else:
            return 0.0

    def run_check(self, context, expected_answer):
        start_time = time.time()
        queue = [self.root]
        sum = 0
        num_nodes = 0
        while queue:
            node = queue.pop()
            num_nodes += 1
            response = self.base.sem_check(context, node.prompt)
            if response.__contains__(expected_answer):
                print(node.root_similarity_score if type(node) is SemanticNode else 1)
                sum += node.root_similarity_score if type(node) is SemanticNode else 1
            for child in node.children:
                queue.append(child)
        end_time = time.time()
        print("Time to run check: ", end_time - start_time)
        return sum / num_nodes

    def run_check_pop_qa(self, model_name):
        start_time = time.time()
        queue = [self.root]
        responses = []
        base_rag_responses = []

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        rag_true_positives = 0
        rag_false_positives = 0
        rag_false_negatives = 0

        while queue:
            node = queue.pop()
            response = self.base.sem_check(node.prompt, model_name)

            if node.rag_closest_match is not None:
                base_rag_response = self.rag.answer_using_wiki_2(
                    model_name, node.prompt, node.rag_closest_match
                )
            else:
                base_rag_response = "No answer"

            found_match = False
            rag_found_match = False

            node.answers[model_name] = {}
            node.answers[model_name]["base"] = response
            node.answers[model_name]["base_rag"] = base_rag_response

            responses.append(response)
            base_rag_responses.append(base_rag_response)

            for expected_answer in json.loads(self.possible_answers):
                if response.__contains__(expected_answer):
                    found_match = True
                    true_positives += 1
                    break
            if not found_match:
                false_positives += 1

            for expected_answer in json.loads(self.possible_answers):
                if base_rag_response.__contains__(expected_answer):
                    rag_found_match = True
                    rag_true_positives += 1
                    break
            if not rag_found_match:
                rag_false_positives += 1

            if not found_match:
                false_negatives += 1
            if not rag_found_match:
                rag_false_negatives += 1
            for child in node.children:
                queue.append(child)

        # Calculate accuracy and F1 score for the base model
        accuracy = (
            true_positives / (true_positives + false_positives + false_negatives)
            if (true_positives + false_positives + false_negatives) > 0
            else 0
        )
        try:
            f1_score = (
                2
                * (true_positives / (true_positives + false_positives))
                * (true_positives / (true_positives + false_negatives))
                / (
                    (true_positives / (true_positives + false_positives))
                    + (true_positives / (true_positives + false_negatives))
                )
                if (true_positives + false_positives) > 0
                and (true_positives + false_negatives) > 0
                else 0
            )
        except:
            f1_score = 0

        # Calculate accuracy and F1 score for RAG
        rag_accuracy = (
            rag_true_positives
            / (rag_true_positives + rag_false_positives + rag_false_negatives)
            if (rag_true_positives + rag_false_positives + rag_false_negatives) > 0
            else 0
        )
        try:
            rag_f1_score = (
                2
                * (rag_true_positives / (rag_true_positives + rag_false_positives))
                * (rag_true_positives / (rag_true_positives + rag_false_negatives))
                / (
                    (rag_true_positives / (rag_true_positives + rag_false_positives))
                    + (rag_true_positives / (rag_true_positives + rag_false_negatives))
                )
                if (rag_true_positives + rag_false_positives) > 0
                and (rag_true_positives + rag_false_negatives) > 0
                else 0
            )
        except:
            rag_f1_score = 0

        end_time = time.time()
        self.time_check[model_name] = end_time - start_time
        print("Time to run check: ", end_time - start_time)
        metrics = {
            "answers": {
                "base": responses,
                "base_rag": base_rag_responses,
            },
            "metrics": {
                "base": {
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                },
                "base_rag": {
                    "accuracy": rag_accuracy,
                    "f1_score": rag_f1_score,
                },
            },
        }
        self.metrics[model_name] = metrics
        return metrics

    def calculate_metrics(self, metrics):
        true_positives = metrics["true_pos"]
        false_positives = metrics["false_pos"]
        false_negatives = metrics["false_neg"]

        accuracy = (
            true_positives / (true_positives + false_positives + false_negatives)
            if (true_positives + false_positives + false_negatives) > 0
            else 0
        )

        try:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0

        return accuracy, f1_score

    def process_node(self, node, model_name):
        # call sem check if response is not in node
        if not node.answers or model_name not in node.answers:
            # base
            response = self.base.sem_check(node.prompt, model_name)
            base_rag_response = "No answer"
            # rag
            if node.rag_closest_match is not None:
                base_rag_response = self.rag.answer_using_wiki_2(
                    model_name, node.prompt, node.rag_closest_match
                )
            node.answers[model_name] = {}
            node.answers[model_name]["base"] = response
            node.answers[model_name]["base_rag"] = base_rag_response
        else:
            print(f"answer for {node.__class__.__name__} already exists")
            response = node.answers[model_name]["base"]
            base_rag_response = node.answers[model_name]["base_rag"]
            
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        base_true_positives = 0
        base_false_positives = 0
        base_false_negatives = 0

        found_match = False
        rag_found_match = False

        # TODO: fix dataset wrong format of possible answer column
        if isinstance(self.possible_answers, str):
            try:
                parsed = json.loads(self.possible_answers)
                if not isinstance(parsed, list):
                    self.possible_answers = json.dumps([self.possible_answers])
            except json.JSONDecodeError:
                self.possible_answers = json.dumps([self.possible_answers])
        elif isinstance(self.possible_answers, list):
            self.possible_answers = json.dumps(self.possible_answers)
               
        for expected_answer in json.loads(self.possible_answers):
            if response.__contains__(expected_answer):
                found_match = True
                true_positives += 1
                break
        if not found_match:
            false_positives += 1

        for expected_answer in json.loads(self.possible_answers):
            if base_rag_response.__contains__(expected_answer):
                rag_found_match = True
                base_true_positives += 1
                break
        if not rag_found_match:
            base_false_positives += 1

        if not found_match:
            false_negatives += 1
        if not rag_found_match:
            base_false_negatives += 1

        values = {
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
        }
        if isinstance(node, (RootNode, SemanticNode)):
            node_semantic_complexity = {
                "complexity_score": node.complexity_score,
                "fk_score": node.fk_score,
                "dc_score": node.dc_score,
            }
            values["semantic_scores"] = node_semantic_complexity

        return node.answers[model_name], values

    def run_check_pop_qa_batched(self, index, model_name, batch_size=5):
        start_time = time.time()
        # Populate the queue with all nodes in the tree
        queue = deque([self.root])
        visited = {self.root}
        while queue:
            node = queue.popleft()
            for child in node.children:
                if child not in visited:
                    queue.append(child)
                    visited.add(child)

        # Now queue contains all nodes in the tree
        # Reset the queue to the start of the list
        queue = deque(list(visited))

        responses = []
        base_rag_responses = []

        # Initialize counters for each model
        metrics = {
            "base": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
            "base_rag": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
        }
        while queue:
            batch = [queue.popleft() for _ in range(min(batch_size, len(queue)))]
            for node in batch:
                # Process the node sequentially
                Timers.start(index, model_name, "sem_check")
                answers, node_metrics = self.process_node(node, model_name)
                Timers.end(index, model_name, "sem_check")

                # Append results
                responses.append(answers["base"])
                base_rag_responses.append(answers["base_rag"])

                # Update the metrics
                for metric in metrics:
                    metrics[metric]["true_pos"] += node_metrics[metric]["true_pos"]
                    metrics[metric]["false_pos"] += node_metrics[metric]["false_pos"]
                    metrics[metric]["false_neg"] += node_metrics[metric]["false_neg"]

        # Calculate accuracy and F1 score for each model
        for metric in metrics:
            accuracy, f1_score = self.calculate_metrics(metrics[metric])
            metrics[metric]["accuracy"] = accuracy
            metrics[metric]["f1_score"] = f1_score

        end_time = time.time()

        answer = {
            "answers": {
                "base": responses,
                "base_rag": base_rag_responses,
            },
            "metrics": metrics,
        }

        self.metrics[model_name] = answer
        
        # update processing time
        elapsed_time = end_time - start_time
        self.time_check[model_name] = self.time_check.get(model_name, 0) + elapsed_time
        end_time = time.time()
        print("Time to run check: ", end_time - start_time)

        return answer

    def add_bleu_and_rouge(self, model_name):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        base_references = [
            [word for word in answer.split()]
            for answer in json.loads(self.possible_answers)
        ]

        if not hasattr(self, "metrics"):
            self.run_check_pop_qa(model_name)

        for method in ["base", "base_rag"]:
            predictions = self.metrics[model_name]["answers"][method]
            cands = [response.split() for response in predictions]
            refs = [base_references for _ in cands]
            if len(cands) != len(refs):
                raise ValueError("The number of responses and references must match.")
            bleu = corpus_bleu(refs, cands)

            rouge_scores = {"rouge1": [], "rougeL": []}

            for response in predictions:
                scores = scorer.score(" ".join(self.possible_answers), response)
                rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

            self.metrics[model_name]["metrics"][method].update(
                {
                    "bleu_score": bleu,
                    "rouge1_list": rouge_scores["rouge1"],
                    "rougeL_list": rouge_scores["rougeL"],
                }
            )

    def to_edges(self):
        edges = []
        node_types = {}
        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            node_types[node.id] = node.type
            for child in node.children:
                edges.append((node.id, child.id))
                queue.append(child)

        return edges, node_types

    def nx_print(self):
        edges, node_types = self.to_edges()
        G = nx.DiGraph(edges)

        color_map = []
        for node in G:
            if node_types[node] == "semantic":
                color_map.append("blue")
            elif node_types[node] == "syntactic":
                color_map.append("green")
            else:
                color_map.append("red")

        pos = nx.spring_layout(G, scale=500000, center=[0, 0])
        nx.draw(G, pos, with_labels=True, node_color=color_map)
        plt.show()

    def get_node_by_id(self, node_id):
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.id == node_id:
                return node
            for child in node.children:
                queue.append(child)
        return None

    def save_tree(self, file_path):
        self.root.move_to_cpu()
        node = {
            "root": self.root,
            "thresholds": self.thresholds,
            "prompt_list": self.prompt_list,
            "time_semantic": self.time_semantic,
            "time_syntactic": self.time_syntactic,
            "time_check": self.time_check,
            "metrics": self.metrics,
            "root_prompt": self.root_prompt,
            "possible_answers": self.possible_answers,
            "rag_entities": self.rag_entities,
            "ner_entities": self.ner_entities,
            "rag_closest_match": self.root.rag_closest_match,
            "gt_passage": self.gt_passage,
        }
        dir = os.path.dirname(file_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(file_path, "wb") as file:
            pickle.dump(node, file)

    @staticmethod
    def load_tree(device, file_path, eval=False, **kargs):
        prev_state = {}
        with open(file_path, "rb") as file:
            prev_state = pickle.load(file)
        root_prompt = prev_state["root_prompt"]
        tree = Tree(
            root_prompt,
            eval=eval,
            sem_perturber=kargs.get("sem_perturber", None),
            syn_perturber=kargs.get("syn_perturber", None),
            generator=kargs.get("generator", None),
            embedder=kargs.get("embedder", None),
            prev_state=prev_state,
        )
        tree.root.move_to_device(device)
        return tree
