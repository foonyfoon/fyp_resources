import wikipedia
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import torch

from collections import deque
from typing import List
import random
import time
import logging

from adapters.OAI_Embeddings import EmbedAdapter

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
        k_hop, top_k = kwargs.get("k_hop", 2), kwargs.get("top_k", 2)
        self.graph = self.create_graph(entity, k_hop, top_k)
        prompt: str = kwargs.get("prompt")
        embedder: EmbedAdapter = kwargs.get("embedder")
        self.ordered_docs = self.get_ordered_docs(prompt=prompt, embedder=embedder)
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
        sentences = sent_tokenize(article)
        vectorizer = TfidfVectorizer()
        # Compute TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Compute the average TF-IDF score for each sentence
        # https://aclanthology.org/P04-1049.pdf
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)  
        # Sort sentences by their scores in descending order
        ranked_sentences = [(score, sentence) for score, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
        
        return ranked_sentences


    def get_k_documents(self, corpus: str, top_k=8) -> str:
        # use rank_sentences_by_tfidf to get top_k doc
        sentences = self.rank_sentences_by_tfidf(corpus)
        return sentences[:top_k]


    def create_graph(self, start_entity: str, k_hop: int, top_k: int):
        '''
        Create a graph/tree using BFS from the start_entity using Wikipedia links.
        '''
        start_time = time.time()
        
        def fetch_wiki_page_with_retry(page_title: str, max_retries=3, base_delay=1, max_delay=8):
            """
            Attempts to fetch a Wikipedia page for the given title with retry logic.
            If a DisambiguationError or PageError occurs, returns None immediately.
            For other exceptions, retries with exponential backoff and jitter.
            """
            attempt = 0
            while True:
                if attempt >= max_retries:
                    logging.error(f"Max retries exceeded for '{page_title}'.")
                    return None
                try:
                    # Attempt to fetch the Wikipedia page
                    return wikipedia.page(page_title)
                except wikipedia.exceptions.DisambiguationError:
                    return None
                except wikipedia.exceptions.PageError:
                    return None
                except Exception as e:
                    attempt += 1
                    delay = min(base_delay * 2 ** attempt, max_delay)
                    delay += random.uniform(0, 1)
                    logging.info(f"Error fetching page '{page_title}': {e}. Retrying in {delay:.2f} seconds (attempt {attempt}/{max_retries}).")
                    time.sleep(delay)
        
        # 1. Initialize root node with the start entity
        root = RootNode(start_entity)
        root.visited = set([start_entity])
        queue = deque()
        pages = wikipedia.search(start_entity, results=top_k)
        
        # 2. Fetch relevant pages related to the start entity
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_wiki_page_with_retry, title): title for title in pages}
            for future in concurrent.futures.as_completed(futures):
                wiki_page = future.result()
                if not wiki_page:
                    continue  # Skip pages that couldn't be fetched or are ambiguous.
                node = PageNode(wiki_page.title, wiki_page.summary, wiki_page.content, wiki_page.url)
                node.document_list = self.get_k_documents(node.content)
                root.add_child(node)
                queue.append((node, wiki_page))
        
        # 3. BFS expansion
        for i in range(k_hop):
            to_process = deque()
            while queue:
                parent_node, parent_page = queue.popleft()
                try:
                    links = list(set(parent_page.links))
                except wikipedia.exceptions.DisambiguationError:
                    continue

                link_weights = []
                valid_links = []
                for link in links:
                    if link not in root.visited:
                        try:
                            weight = self.link_importance(link, parent_page.content)
                            link_weights.append(weight)
                            valid_links.append(link)
                        except Exception:
                            continue

                # Sort valid_links based on weights and select the top_k links.
                sorted_indices = sorted(range(len(link_weights)), key=lambda i: link_weights[i])
                top_indices = sorted_indices[:min(len(valid_links), top_k)]  # Use valid_links length here.
                top_links = [valid_links[i] for i in top_indices]
            
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(fetch_wiki_page_with_retry, link): link for link in top_links}
                    for future in concurrent.futures.as_completed(futures):
                        child_page = future.result()
                        if not child_page:
                            continue  # Skip pages that couldn't be fetched or are ambiguous.
                        
                        # Retrieve the corresponding link for this future.
                        link = futures[future]
                        
                        child_node = PageNode(child_page.title, child_page.summary, child_page.content, child_page.url)
                        # Generate passages (passage, tfidf score)
                        child_node.document_list = self.get_k_documents(child_page.content)
                        parent_node.add_child(child_node)
                        root.visited.add(link)
                        to_process.append((child_node, child_page))
                        
            queue = to_process

        print(f"--- create graph: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} seconds ---")
        return root


    
    def get_ordered_docs(self, strategy="similarity", **kargs):
        '''
        proposes next document for preturber to add to prompt, if prompt is accepted, mark sentence as visited
        '''
        # Collect all docs by traversing the tree in a breadth-first style.
        start_time = time.time()
        doc_db = []
        title_db = []
        content_db = []
        
        # Get the set of titles from the root's direct children.
        direct_children_titles = set()
        for child in self.graph.children:
            direct_children_titles.add(child.title)
        queue = deque([self.graph])
        while queue:
            node = queue.popleft()
            if isinstance(node, PageNode) and node.title not in direct_children_titles: #Skip node's document list if at hop 1
                title_db.append(node.title)
                for entry in node.document_list:
                    # entry is (score, sentence)
                    doc_db.append(entry[1])
                    content_db.append(node.content)      
            for child in node.children:
                    queue.append(child)
                                           
        if strategy == "similarity":
            prompt: str = kargs.get("prompt")
            embedder: EmbedAdapter = kargs.get("embedder") #(returns embbedding with embedder.embed(sentence))
            # Get the embedding for the prompt.
            prompt_embedding =  torch.tensor(embedder.encode(prompt) ) # Assume shape (d,)
            
            # Generate embeddings for each document.
            doc_embeddings = []
            for doc in doc_db:
                doc_embedding = embedder.encode(doc)
                doc_tensor = torch.tensor(doc_embedding)
                doc_embeddings.append(doc_tensor)
            
            print(f"len(doc_embeddings), len(doc_db): {len(doc_embeddings), len(doc_db)}")
            # Stack embeddings into a single tensor of shape (N, d).
            db_embeddings = torch.stack(doc_embeddings, dim=0)
            # Compute sot product similarity between each doc and the prompt.
            if prompt_embedding.dim() == 1:
                prompt_embedding = prompt_embedding.unsqueeze(0)
            similarities = torch.matmul(db_embeddings, prompt_embedding.mT).squeeze()  # shape (N,)
            
            # Sort documents by similarity in ascending order (least similar first).
            sorted_indices = torch.argsort(similarities, descending=False)
            ordered_document_list = [(doc_db[i], float(similarities[i].item()), title_db[i], content_db[i]) for i in sorted_indices]
            print(f"--- get_ordered_docs: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} seconds ---")
            return ordered_document_list
             
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented.")
        

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
        doc, sim, title, content = self.ordered_docs[selected_index]
        return (doc, title, content, selected_index)


    def update_visit_status(self, index: int):
        """
        Marks the document at the provided index as visited by setting the corresponding
        boolean flag to True
        """
        if self.visited_doc_flags[index]:
            raise RuntimeError(f"Document at index {index} has already been visited.")
        self.visited_doc_flags[index] = True
