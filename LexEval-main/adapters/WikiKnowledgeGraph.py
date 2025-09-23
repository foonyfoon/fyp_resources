import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import torch
import wikipedia
import textwrap

from collections import deque
from typing import List, Union
import random
import time
import logging

from utils.wiki_helper import WikiHelper, get_exact_page_from_entity
class RootNode:
    def __init__(self, titles: List[str]):
        # a tree can have 1..n roots entities
        self.title = ", ".join(titles)
        self.children = []
        self.visited = set() # set of entities
        
    def add_child(self, node):
        self.children.append(node)
    
    def visualize(self):
        G = nx.DiGraph()

        def add_node_edges(node, parent_label=None):
            if isinstance(node, RootNode):
                node_label = f"Root: {node.title}"
            else:
                node_label = f"Page: {node.title}"

            G.add_node(node_label)

            if parent_label:
                G.add_edge(parent_label, node_label)

            for child in node.children:
                add_node_edges(child, node_label)

        # Allow single root or list of roots
        if isinstance(self, list):
            for root in self:
                add_node_edges(root)
        else:
            add_node_edges(self)

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

        # Color root nodes yellow, others lightblue
        node_colors = ['yellow' if node.startswith("Root:") else 'lightblue' for node in G.nodes()]

        def wrap_label(label, width=20):
            return "\n".join(textwrap.wrap(label, width))
        labels = {node: wrap_label(node) for node in G.nodes()}

        plt.figure(figsize=(14, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=1500,
            font_size=14,
            edge_color='gray',
            arrows=True,
            arrowsize=10
        )
        
        # If multiple roots, make a collective title
        graph_title = ', '.join(r.title for r in self) if isinstance(self, list) else self.title
        plt.title(f"Knowledge Graph Visualization of {graph_title}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
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
    def __init__(self, entity: str, cr_model, skip_cr = False, cache_res=True, **kwargs):
        prompt: str = kwargs.get("prompt")
        k_hop, top_k = 2, 3
        if not skip_cr:
            self.cr = cr_model
        self.skip_cr = skip_cr
        self.cache_res = cache_res
        self.embedder = kwargs.get("embedder")
        self.wiki_helper = WikiHelper(self.embedder)
        self.graph = self.create_graph(entity, k_hop, top_k)
        self.ordered_docs = self.get_unordered_docs(prompt=prompt, embedder=self.embedder)
        # Initialize a boolean array for tracking visited documents:
        self.visited_doc_flags = [False] * len(self.ordered_docs)
        
            
    def link_importance(self, title: str, content: str,  summary: str) -> int:
        """Returns an importance score for a link based on its presence in the text"""
        score = 0
        title_lower = title.lower()
        content_lower = content.lower()
        summary_lower = content.lower()
        score += content_lower.count(title_lower)
        score += (summary_lower.count(title_lower) * 2)
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
                # 2) Discard any sentence shorter than 5 words
                if len(sent.split()) >= 5:
                    sentences.append(sent)
        vectorizer = TfidfVectorizer()
        # Compute TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Compute the average TF-IDF score for each sentence https://aclanthology.org/P04-1049.pdf
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)  
        # Sort sentences by their scores in descending order
        ranked_sentences = [(score, sentence) for score, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
        
        return ranked_sentences
        

    def get_k_documents(self, corpus: str, top_k: int = 50) -> List[str]:
        """
        Preprocess corpus: resolve corefs so each sentence has standalone meaning.
        Then return top_k sentences ranked by TF-IDF relevance.
        """
        # Step 1: Coreference resolution
        if not self.skip_cr:
            resolved_corpus = self.cr.apply_coref_resolution(corpus)
        else:
            resolved_corpus = corpus
        # Step 2: Sentence ranking via TF-IDF
        sentences = self.rank_sentences_by_tfidf(resolved_corpus)
        top_k_ret = min(len(sentences), top_k)
        return sentences[:top_k_ret]


    def create_graph(self, start_entity: Union[str, List[str]], k_hop: int, top_k: int):
        '''
        Create a graph/tree using BFS from the start_entity using Wikipedia links.
        '''
        start_time = time.time()

        # Ensure start_entity is a list of strings
        if isinstance(start_entity, str):
            start_entity = [start_entity]

        # 1. Initialize root node with the start entity
        root = RootNode(start_entity)
        root.visited = set(start_entity)
        queue = deque()
        
        # 2. Fetch the Wikipedia page for the start entities
        for start_ent in start_entity:
            wiki_page = get_exact_page_from_entity(start_ent)
    
            node = PageNode(wiki_page.title,
                            wiki_page.summary,
                            wiki_page.content,
                            wiki_page.url)
            node.document_list = self.get_k_documents(node.content)
            root.add_child(node)
            root.visited.add(wiki_page.title)
            page_dict = {
                "title":wiki_page.title,
                "summary":wiki_page.summary,
                "content":wiki_page.content,
                "url":wiki_page.url,
                'links': wiki_page.links,
                }
            queue.append((node, page_dict))
        
        # 3. BFS expansion
        for i in range(k_hop):
            to_process = deque()
            while queue:
                parent_node, parent_page = queue.popleft()
                try:
                    links = list(set(parent_page['links']))

                except wikipedia.exceptions.DisambiguationError:
                    logging.error("create_graph: wikipedia.exceptions.DisambiguationError: no links?")
                link_weights = []
                valid_links = []
                for link in links:
                    if link not in root.visited:
                        try:
                            weight = self.link_importance(link, parent_page['content'],  parent_page['summary'])
                            link_weights.append(weight)
                            valid_links.append(link)
                        except Exception as e:
                            logging.error("create_graph: Exception weights", e)
                
                # Sort valid_links based on weights and select the top_k links.
                sorted_indices = sorted(range(len(link_weights)), key=lambda i: link_weights[i], reverse=True)
                top_indices = sorted_indices[:min(len(valid_links), top_k)]  # Use valid_links length here
                top_links = [valid_links[i] for i in top_indices] # Use valid_links length here
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(self.wiki_helper.fetch_wiki_page_with_retry, link, from_db=False): link for link in top_links}
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

        logging.info(f"--- create graph: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} ---")
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
        logging.info(f"--- get_unordered_docs: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} seconds ---")
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
