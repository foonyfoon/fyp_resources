import json
from collections import deque
import time

import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures

from adapters.OAI_Embeddings import OAIEmbedAdapter
from similarity.cosine_similarity import similarity

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

from adapters.rag import (
    retrieve_wiki_data,
    find_most_relevant_page,
    search_entities,
    answer_using_wiki,
    search_entities_NER,
    find_closest_contriever_match,
    create_retriever,
    retrieve_bm25,
)
from tree.node import RootNode, SyntacticNode, SemanticNode


class Tree:
    def __init__(self, root_prompt, adapter, perturbor):
        self.embed_model = OAIEmbedAdapter()
        self.root_prompt = root_prompt
        self.adapter = adapter
        self.perturbor = perturbor
        self.num_semantic = 0
        self.num_syntactic = 0
        self.root = RootNode(root_prompt)
        self.root.embedding = self.embed_model.encode(root_prompt)
        wiki_data = retrieve_wiki_data(root_prompt)
        closest_match = find_most_relevant_page(
            wiki_data=wiki_data, prompt=root_prompt
        )
        contriever_closest_match = find_closest_contriever_match(
            wiki_data=wiki_data, prompt=root_prompt
        )
        bm25_retriever = create_retriever(wiki_data)
        self.rag_entities = search_entities(prompt=root_prompt).split(",")
        self.ner_entities = search_entities_NER(prompt=root_prompt)
        self.root.rag_closest_match = closest_match
        self.root.contriever_closest_match = contriever_closest_match
        self.root.bm25_closest_match = retrieve_bm25(
            bm25_retriever, root_prompt
        )
        self.thresholds = []
        self.prompt_list = [root_prompt]
        self.time_semantic = 0
        self.time_syntactic = 0
        self.time_check = {}
        self.metrics = {}

    def set_possible_answers(self, possible_answers):
        self.possible_answers = possible_answers

    def make_tree(self, depth, num_semantic, num_syntactic, model_name='gpt-3.5-turbo', rag_eval=False):
        start_time = time.time()
        self.num_semantic = num_semantic
        self.num_syntactic = num_syntactic
        self.thresholds = self.make_thresholds("linear", 1.0, 0.96, depth)

        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()

            if level > depth - 1:
                continue

            upper_thresh = 0.96
            low_thresh = 0.8

            perturbations = self.adapter.sem_perturb_combined(
                node.prompt, num_semantic
            )

            # Check perturbations concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.check_perturbation,
                        node,
                        perturbation,
                        self.root.embedding,
                        upper_thresh,
                        low_thresh,
                    )
                    for perturbation in perturbations
                ]
                for future in concurrent.futures.as_completed(futures):
                    semantic_node = future.result()
                    node.add_child(semantic_node)
                    queue.append((semantic_node, level + 1))
            sem_time = time.time()

            # Create syntactic nodes concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.perturbor.syn_perturb,
                        text=node.prompt,
                        butterfinger=self.perturbor.butterfinger,
                    )
                    for _ in range(num_syntactic)
                ]
                for future in concurrent.futures.as_completed(futures):
                    syn_perturb = future.result()
                    wiki_data = retrieve_wiki_data(syn_perturb)
                    closest_match = find_most_relevant_page(
                        wiki_data=wiki_data, prompt=syn_perturb
                    )
                    rag_entities = search_entities(prompt=syn_perturb).split(
                        ","
                    )
                    ner_entities = search_entities_NER(prompt=syn_perturb)
                    rag_closest_match = closest_match
                    contriever_closest_match = find_closest_contriever_match(
                        wiki_data=wiki_data, prompt=syn_perturb
                    )
                    bm25_retriever = create_retriever(wiki_data)
                    syntactic_node = SyntacticNode(
                        syn_perturb,
                        0.0,
                        "test_context",
                        parent=node,
                        rag_closest_match=rag_closest_match,
                        contriever_closest_match=contriever_closest_match,
                        bm25_closest_match=retrieve_bm25(
                            bm25_retriever, syn_perturb
                        ),
                        rag_entities=rag_entities,
                        ner_entities=ner_entities,
                    )
                    node.add_child(syntactic_node)
            syn_time = time.time()

        self.time_semantic = sem_time - start_time
        self.time_syntactic = syn_time - sem_time
        print("Time to create semantic nodes: ", sem_time - start_time)
        print("Time to create syntactic nodes: ", syn_time - sem_time)
        print("Total time: ", syn_time - start_time)

    def check_perturbation(
            self, node, perturbation, root_embedding, upper_thresh, lower_thresh
    ):
        sem_perturb_embedding = self.embed_model.encode(perturbation)
        parent_root_similarity = node.root_similarity_score

        root_similarity_score = similarity(
            root_embedding, sem_perturb_embedding
        )

        semantic_similarity_score = similarity(
            node.embedding, sem_perturb_embedding
        )

        retry_count = 0
        best_perturbation = None
        best_semantic_similarity = 0.0
        best_embedding = None
        best_root_similarity = float("inf")
        while (
                semantic_similarity_score > upper_thresh
                or semantic_similarity_score < lower_thresh
                or (parent_root_similarity < root_similarity_score)
                or perturbation in self.prompt_list
        ) and retry_count < 5:
            perturbation = self.adapter.sem_perturb(node.prompt)
            sem_perturb_embedding = self.embed_model.encode(perturbation)
            semantic_similarity_score = similarity(
                node.embedding, sem_perturb_embedding
            )
            root_similarity_score = similarity(
                root_embedding, sem_perturb_embedding
            )

            if root_similarity_score < best_root_similarity:
                best_perturbation = perturbation
                best_semantic_similarity = semantic_similarity_score
                best_root_similarity = root_similarity_score
                best_embedding = sem_perturb_embedding

            retry_count += 1

        if best_perturbation is not None:
            wiki_data = retrieve_wiki_data(best_perturbation)
            closest_match = find_most_relevant_page(
                wiki_data=wiki_data, prompt=best_perturbation
            )
            rag_entities = search_entities(prompt=best_perturbation).split(",")
            rag_closest_match = closest_match
            ner_entities = search_entities_NER(prompt=best_perturbation)
            contriever_closest_match = find_closest_contriever_match(
                wiki_data, best_perturbation
            )
            bm25_retriever = create_retriever(wiki_data)
            semantic_node = SemanticNode(
                best_perturbation,
                best_semantic_similarity,
                best_root_similarity,
                lower_thresh,
                best_embedding,
                rag_closest_match,
                contriever_closest_match,
                retrieve_bm25(bm25_retriever, best_perturbation),
                rag_entities,
                ner_entities,
                parent=node,
            )
        else:
            wiki_data = retrieve_wiki_data(perturbation)
            closest_match = find_most_relevant_page(
                wiki_data=wiki_data, prompt=perturbation
            )
            rag_entities = search_entities(prompt=perturbation).split(",")
            rag_closest_match = closest_match
            ner_entities = search_entities_NER(prompt=perturbation)
            contriever_closest_match = find_closest_contriever_match(
                wiki_data, perturbation
            )
            bm25_retriever = create_retriever(wiki_data)
            semantic_node = SemanticNode(
                perturbation,
                semantic_similarity_score,
                root_similarity_score,
                lower_thresh,
                sem_perturb_embedding,
                rag_closest_match,
                contriever_closest_match,
                retrieve_bm25(bm25_retriever, perturbation),
                rag_entities,
                ner_entities,
                parent=node,
            )

        self.prompt_list.append(semantic_node.prompt)
        return semantic_node

    def evaluate(self):
        queue = []
        sum = 0
        queue.append((self.root, 1))
        while len(queue) != 0:
            node, level = queue.pop()
            if node.type == "semantic":
                pass
            if node.type == "syntactic":
                pass
            for child in node.children:
                queue.append((child, level + 1))

        return sum

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
            response = self.adapter.sem_check(context, node.prompt)
            if response.__contains__(expected_answer):
                print(
                    node.root_similarity_score
                    if type(node) is SemanticNode
                    else 1
                )
                sum += (
                    node.root_similarity_score
                    if type(node) is SemanticNode
                    else 1
                )
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
        bm25_responses = []
        contriever_responses = []

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        rag_true_positives = 0
        rag_false_positives = 0
        rag_false_negatives = 0

        bm25_true_positives = 0
        bm25_false_positives = 0
        bm25_false_negatives = 0

        contr_true_positives = 0
        contr_false_positives = 0
        contr_false_negatives = 0

        while queue:
            node = queue.pop()
            response = self.adapter.sem_check(node.prompt, model_name)

            if node.rag_closest_match is not None:
                base_rag_response = answer_using_wiki(
                    model_name,
                    node.prompt,
                    node.rag_closest_match["content"],
                    node.rag_closest_match["title"],
                )
            else:
                base_rag_response = "No answer"

            if node.bm25_closest_match is not None:
                bm25_rag_response = answer_using_wiki(
                    model_name,
                    node.prompt,
                    node.bm25_closest_match[0].page_content,
                    "",
                )
            else:
                bm25_rag_response = "No answer"

            if node.contriever_closest_match is not None:
                contriever_response = answer_using_wiki(
                    model_name,
                    node.prompt,
                    node.contriever_closest_match["text"]["content"],
                    node.contriever_closest_match["text"]["title"],
                )
            else:
                contriever_response = "No answer"

            found_match = False
            rag_found_match = False
            bm25_found_match = False
            contr_found_match = False

            node.answers[model_name] = {}
            node.answers[model_name]["base"] = response
            node.answers[model_name]["base_rag"] = base_rag_response
            node.answers[model_name]["bm25_rag"] = bm25_rag_response
            node.answers[model_name]["contriever_rag"] = contriever_response

            responses.append(response)
            base_rag_responses.append(base_rag_response)
            bm25_responses.append(bm25_rag_response)
            contriever_responses.append(contriever_response)

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

            for expected_answer in json.loads(self.possible_answers):
                if bm25_rag_response.__contains__(expected_answer):
                    bm25_found_match = True
                    bm25_true_positives += 1
                    break
            if not bm25_found_match:
                bm25_false_positives += 1

            for expected_answer in json.loads(self.possible_answers):
                if contriever_response.__contains__(expected_answer):
                    contr_found_match = True
                    contr_true_positives += 1
                    break
            if not contr_found_match:
                contr_false_positives += 1

            if not found_match:
                false_negatives += 1
            if not rag_found_match:
                rag_false_negatives += 1
            if not bm25_found_match:
                bm25_false_negatives += 1
            if not contr_found_match:
                contr_false_negatives += 1
            for child in node.children:
                queue.append(child)

        # Calculate accuracy and F1 score for the base model
        accuracy = (
            true_positives
            / (true_positives + false_positives + false_negatives)
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
            if (rag_true_positives + rag_false_positives + rag_false_negatives)
               > 0
            else 0
        )
        try:
            rag_f1_score = (
                2
                * (
                        rag_true_positives
                        / (rag_true_positives + rag_false_positives)
                )
                * (
                        rag_true_positives
                        / (rag_true_positives + rag_false_negatives)
                )
                / (
                        (
                                rag_true_positives
                                / (rag_true_positives + rag_false_positives)
                        )
                        + (
                                rag_true_positives
                                / (rag_true_positives + rag_false_negatives)
                        )
                )
                if (rag_true_positives + rag_false_positives) > 0
                   and (rag_true_positives + rag_false_negatives) > 0
                else 0
            )
        except:
            rag_f1_score = 0

        # Calculate accuracy and F1 score for BM25
        bm25_accuracy = (
            bm25_true_positives
            / (
                    bm25_true_positives
                    + bm25_false_positives
                    + bm25_false_negatives
            )
            if (
                       bm25_true_positives
                       + bm25_false_positives
                       + bm25_false_negatives
               )
               > 0
            else 0
        )
        try:
            bm25_f1_score = (
                2
                * (
                        bm25_true_positives
                        / (bm25_true_positives + bm25_false_positives)
                )
                * (
                        bm25_true_positives
                        / (bm25_true_positives + bm25_false_negatives)
                )
                / (
                        (
                                bm25_true_positives
                                / (bm25_true_positives + bm25_false_positives)
                        )
                        + (
                                bm25_true_positives
                                / (bm25_true_positives + bm25_false_negatives)
                        )
                )
                if (bm25_true_positives + bm25_false_positives) > 0
                   and (bm25_true_positives + bm25_false_negatives) > 0
                else 0
            )
        except:
            bm25_f1_score = 0

        # Calculate accuracy and F1 score for Contriever
        contr_accuracy = (
            contr_true_positives
            / (
                    contr_true_positives
                    + contr_false_positives
                    + contr_false_negatives
            )
            if (
                       contr_true_positives
                       + contr_false_positives
                       + contr_false_negatives
               )
               > 0
            else 0
        )
        try:
            contr_f1_score = (
                2
                * (
                        contr_true_positives
                        / (contr_true_positives + contr_false_positives)
                )
                * (
                        contr_true_positives
                        / (contr_true_positives + contr_false_negatives)
                )
                / (
                        (
                                contr_true_positives
                                / (contr_true_positives + contr_false_positives)
                        )
                        + (
                                contr_true_positives
                                / (contr_true_positives + contr_false_negatives)
                        )
                )
                if (contr_true_positives + contr_false_positives) > 0
                   and contr_true_positives > 0
                   and (contr_true_positives + contr_false_negatives) > 0
                else 0
            )
        except:
            contr_f1_score = 0

        end_time = time.time()
        self.time_check[model_name] = end_time - start_time
        print("Time to run check: ", end_time - start_time)
        metrics = {
            "answers": {
                "base": responses,
                "base_rag": base_rag_responses,
                "bm25_rag": bm25_responses,
                "contriever_rag": contriever_responses,
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
                "bm25_rag": {
                    "accuracy": bm25_accuracy,
                    "f1_score": bm25_f1_score,
                },
                "contriever_rag": {
                    "accuracy": contr_accuracy,
                    "f1_score": contr_f1_score,
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
            true_positives
            / (true_positives + false_positives + false_negatives)
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
        # This method contains the code to process a single node.
        response = self.adapter.sem_check(node.prompt, model_name)
        base_rag_response = "No answer"
        bm25_rag_response = "No answer"
        contriever_response = "No answer"
        if node.rag_closest_match is not None:
            base_rag_response = answer_using_wiki(
                model_name,
                node.prompt,
                node.rag_closest_match["content"],
                node.rag_closest_match["title"],
            )
        if node.bm25_closest_match is not None:
            bm25_rag_response = answer_using_wiki(
                model_name,
                node.prompt,
                node.bm25_closest_match[0].page_content,
                "",
            )
        if node.contriever_closest_match is not None:
            contriever_response = answer_using_wiki(
                model_name,
                node.prompt,
                node.contriever_closest_match["text"]["content"],
                node.contriever_closest_match["text"]["title"],
            )
        node.answers[model_name] = {}
        node.answers[model_name]["base"] = response
        node.answers[model_name]["base_rag"] = base_rag_response
        node.answers[model_name]["bm25_rag"] = bm25_rag_response
        node.answers[model_name]["contriever_rag"] = contriever_response

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

        for expected_answer in json.loads(self.possible_answers):
            if bm25_rag_response.__contains__(expected_answer):
                bm25_found_match = True
                bm25_true_positives += 1
                break
        if not bm25_found_match:
            bm25_false_positives += 1

        for expected_answer in json.loads(self.possible_answers):
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

        return node.answers[model_name], values

    def run_check_pop_qa_batched(self, model_name, batch_size=5):
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
        bm25_responses = []
        contriever_responses = []

        # Initialize counters for each model
        metrics = {
            "base": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
            "base_rag": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
            "bm25_rag": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
            "contriever_rag": {"true_pos": 0, "false_pos": 0, "false_neg": 0},
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while queue:
                batch = [
                    queue.popleft() for _ in range(min(batch_size, len(queue)))
                ]
                futures = [
                    executor.submit(self.process_node, node, model_name)
                    for node in batch
                ]
                for future in concurrent.futures.as_completed(futures):
                    answers, node_metrics = future.result()
                    responses.append(answers["base"])
                    base_rag_responses.append(answers["base_rag"])
                    bm25_responses.append(answers["bm25_rag"])
                    contriever_responses.append(answers["contriever_rag"])
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
                "bm25_rag": bm25_responses,
                "contriever_rag": contriever_responses,
            },
            "metrics": metrics,
        }

        self.metrics[model_name] = answer
        self.time_check[model_name] = end_time - start_time
        end_time = time.time()
        print("Time to run check: ", end_time - start_time)

        return answer

    def add_bleu_and_rouge(self, model_name):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )
        base_references = [
            [word for word in answer.split()]
            for answer in json.loads(self.possible_answers)
        ]

        if not hasattr(self, "metrics"):
            self.run_check_pop_qa(model_name)

        for method in ["base", "base_rag", "bm25_rag", "contriever_rag"]:
            predictions = self.metrics[model_name]["answers"][method]
            cands = [response.split() for response in predictions]
            refs = [base_references for _ in cands]
            if len(cands) != len(refs):
                raise ValueError(
                    "The number of responses and references must match."
                )
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

    def print_tree(self, node=None, level=0, model_name=None):
        if node is None:
            node = self.root
        if type(node) == RootNode or type(node) == SyntacticNode:
            print("  " * level + f"{node.id} - {node.prompt}")
            for model_name in node.answers:
                print(
                    "  " * level
                    + f"Answer ({model_name}): {node.answers[model_name]}"
                )
        else:
            print(
                "  " * level
                + f"{node.id} - {node.prompt} - RAG({node.rag_closest_match})"
            )
            for model_name in node.answers:
                print(
                    "  " * level
                    + f"Answer ({model_name}): {node.answers[model_name]}"
                )
        for child in node.children:
            self.print_tree(child, level + 1, model_name)
