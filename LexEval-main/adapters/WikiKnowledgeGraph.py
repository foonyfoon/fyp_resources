import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import torch
import wikipedia

from collections import deque
from typing import List
import random
import time
import logging

from adapters.OAI_Embeddings import EmbedAdapter
from utils.wiki_helper import WikiHelper
class RootNode:
    def __init__(self, title: str):
        self.title = title
        self.children = []
        self.visited = set() # set of entities
        
    def add_child(self, node):
        self.children.append(node)
    
    def visualize(self):
        # Initialize a directed graph
        G = nx.DiGraph()

        # Recursive function to traverse nodes and add them to the graph
        def add_node_edges(node, parent_label=None):
            # Create a label based on the type of node
            if isinstance(node, RootNode):
                node_label = f"Root: {node.title}"
            else:
                node_label = f"Page: {node.title}"
            
            G.add_node(node_label)
            
            if parent_label:
                G.add_edge(parent_label, node_label)
            
            # Recursively process all children
            for child in node.children:
                add_node_edges(child, node_label)

        # Start building the graph from the root
        add_node_edges(self)

        # Draw the graph using a spring layout
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, node_size=800, font_size=5)
        plt.title("Graph Visualization Using NetworkX")
        plt.show()
   
class PageNode:
    def __init__(self, title: str, summary: str, content: str, url: str):
        self.title = title
        self.summary = summary
        self.content = content
        self.url = url
        self.children = []
        
        # passsage to insert: list of tuples if (doc, tfidf score)
        self.document_list = []
    
    def add_child(self, node):
        self.children.append(node)

class KnowledgeGraph: 
    '''
    each knowledge graph is created per root prompt and stored there
    '''
    def __init__(self, entity: str, **kwargs):
        prompt: str = kwargs.get("prompt")
        k_hop, top_k = 2, 3
        self.embedder = kwargs.get("embedder")
        self.wiki_helper = WikiHelper(self.embedder)
        self.graph = self.create_graph(entity, k_hop, top_k)
        self.ordered_docs = self.get_unordered_docs(prompt=prompt, embedder=self.embedder)
        # Initialize a boolean array for tracking visited documents:
        self.visited_doc_flags = [False] * len(self.ordered_docs)
        
            
    def link_importance(self, title: str, content: str) -> int:
            """Returns an importance score for a link based on its presence in the text"""
            score = 0
            title_lower = title.lower()
            content_lower = content.lower()

            score += content_lower.count(title_lower)
            if title_lower in content_lower[:500]:  # check if in summary-like start
                score += 3
            return score


    def rank_sentences_by_tfidf(self, article: str) -> List[str]:
        # Split on newline, tokenize each non-empty line, and flatten
        raw_lines = article.split('\n')
        sentences = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            for sent in sent_tokenize(line):
                # 2) Discard any sentence shorter than 5 characters
                if len(sent) >= 5:
                    sentences.append(sent)
        vectorizer = TfidfVectorizer()
        # Compute TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Compute the average TF-IDF score for each sentence https://aclanthology.org/P04-1049.pdf
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)  
        # Sort sentences by their scores in descending order
        ranked_sentences = [(score, sentence) for score, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
        
        return ranked_sentences


    def get_k_documents(self, corpus: List[str], top_k: int = 50) -> List[str]:
        # use rank_sentences_by_tfidf to get top_k doc
        sentences = self.rank_sentences_by_tfidf(corpus)
        top_k_ret = min(len(sentences), top_k)
        return sentences[:top_k_ret]


    def create_graph(self, start_entity: str, k_hop: int, top_k: int):
        '''
        Create a graph/tree using BFS from the start_entity using Wikipedia links.
        '''
        start_time = time.time()
        
        # tree_size = sum([pow(size[1], exponent) for exponent in range(size[0])]) - 1 # not counting root
        # num_kg_nodes = sum([pow(kg_size[1], exponent) for exponent in range(kg_size[0])]) - 1 # nor counting root and first layer
        # doc_per_tree_node = 10
        # num_preturb_tree_node = num_kg_nodes
        
        # 1. Initialize root node with the start entity
        root = RootNode(start_entity)
        root.visited = {start_entity}
        queue = deque()
        
        pages = wikipedia.search(start_entity, results=top_k)
        print(f"create_graph root pages: {pages}")
        
        # 2. Fetch relevant pages related to the start entity
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.wiki_helper.fetch_wiki_page_with_retry, title): title for title in pages}
            for future in concurrent.futures.as_completed(futures):
                wiki_page = future.result()
                if not wiki_page:
                    continue  # Skip pages that couldn't be fetched or are ambiguous.
                node = PageNode(wiki_page['title'],
                                wiki_page['summary'],
                                wiki_page['content'],
                                wiki_page['url'])
                node.document_list = self.get_k_documents(node.content)
                root.add_child(node)
                root.visited.add(wiki_page['title'])
                queue.append((node, wiki_page))
        
        # 3. BFS expansion
        for i in range(k_hop):
            to_process = deque()
            while queue:
                parent_node, parent_page = queue.popleft()
                try:
                    links = list(set(parent_page["links"]))
                except wikipedia.exceptions.DisambiguationError:
                    logging.info("create_graph: wikipedia.exceptions.DisambiguationError: no links?")
                link_weights = []
                valid_links = []
                for link in links:
                    if link not in root.visited:
                        try:
                            weight = self.link_importance(link, parent_page["content"])
                            link_weights.append(weight)
                            valid_links.append(link)
                        except Exception:
                            logging.info("create_graph: Exception weights")

                # Sort valid_links based on weights and select the top_k links.
                sorted_indices = sorted(range(len(link_weights)), key=lambda i: link_weights[i])
                top_indices = sorted_indices[:min(len(valid_links), top_k)]  # Use valid_links length here.
                top_links = [valid_links[i] for i in top_indices]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(self.wiki_helper.fetch_wiki_page_with_retry, link): link for link in top_links}
                    for future in concurrent.futures.as_completed(futures):
                        child_page = future.result()
                        if not child_page:
                            continue  # Skip pages that couldn't be fetched or are ambiguous.
                        child_node = PageNode(child_page["title"], child_page["summary"], child_page["content"], child_page["url"])
                        # Generate passages (passage, tfidf score)
                        child_node.document_list = self.get_k_documents(child_page["content"])
                        parent_node.add_child(child_node)
                        to_process.append((child_node, child_page))
                        
                for tl in top_links:
                    root.visited.add(tl)
                        
            queue = to_process

        print(f"--- create graph: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} ---")
        return root

        
    def get_unordered_docs(self, **kwargs):
        """
        Proposes next documents for the perturber to add to the prompt.
        Documents are collected via a breadth-first traversal of the document tree,
        and returned in the order they are encountered (no similarity-based sorting).

        Returns:
            List[Tuple[str, str, str]]:
                A list of tuples each containing:
                - Document text,
                - Document title, and
                - Document content.
        """
        import time
        from collections import deque

        start_time = time.time()
        doc_db = []
        title_db = []
        content_db = []
        
        # Get the set of titles from the root's direct children to avoid collecting their document lists.
        direct_children_titles = {child.title for child in self.graph.children}
        queue = deque([self.graph])
        
        while queue:
            node = queue.popleft()
            # If node is a PageNode and not a direct child of the root,
            # collect its document sentences.
            if isinstance(node, PageNode) and node.title not in direct_children_titles:
                for entry in node.document_list:
                    # Each entry is assumed to be a tuple (score, sentence); we use the sentence.
                    doc_db.append(entry[1])
                    title_db.append(node.title)
                    content_db.append(node.content)
            # Add child nodes to the queue for further traversal.
            for child in node.children:
                queue.append(child)
        
        unordered_docs = [(doc_db[i], title_db[i], content_db[i]) for i in range(len(doc_db))]
        print(f"--- get_unordered_docs: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} seconds ---")
        return unordered_docs

    def get_next_document(self):
        """
        Proposes the next document for the perturber to add to the prompt that has not been visited.
        """
        # Identify indices for documents that have not been visited.
        available_indices = [i for i, visited in enumerate(self.visited_doc_flags) if not visited]
        if not available_indices:
            raise RuntimeError(f"All documents are visited. doclist length = {len(self.ordered_docs)}")
        # Randomly select an available document's index.
        selected_index = random.choice(available_indices)
        doc, title, content = self.ordered_docs[selected_index]
        return (doc, title, content, selected_index)


    def update_visit_status(self, index: int):
        """
        Marks the document at the provided index as visited by setting the corresponding
        boolean flag to True
        """
        if self.visited_doc_flags[index]:
            raise RuntimeError(f"Document at index {index} has already been visited.")
        self.visited_doc_flags[index] = True
