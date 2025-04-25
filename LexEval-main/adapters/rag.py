import json
import concurrent.futures
from typing import List, Dict, Any, Tuple
import logging
import requests

import wikipedia
import textwrap
import spacy
import torch
import torch.nn.functional as F

from adapters.SemanticAdapter import SemanticAdapter
from adapters.OAI_Embeddings import EmbedAdapter
from utils.wiki_helper import WikiHelper
from similarity.cosine_similarity import similarity, similarities

class RAGAgent:
    def __init__(self, **kwargs):
        # If eval mode is enabled, set self.generator to the provided generator
        if kwargs.get("eval", False):
            self.generator = SemanticAdapter(kwargs.get("generator"))
            self.ner_model  = None
        else:
            self.ner_model  = SemanticAdapter(kwargs.get("ner_model"))
            self.generator = None
        self.query = None
        self.embedding_adapter: EmbedAdapter = kwargs.get("embedder")
        self.NER = spacy.load(
            "en_core_web_trf"
        )
        self.wiki_helper = WikiHelper(self.embedding_adapter)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA not available. Using CPU instead.")
        self.device = device


    def create_passage_db(
        self, wiki_data: List[Dict],
    ):
        if not wiki_data:
            return None
        # print("Creating pages retrieval database...")
        texts = [data["context"] for data in wiki_data]
        titles = [data["title"] for data in wiki_data]
        emb_tensor = torch.stack([
            torch.tensor(data["embedding"], dtype=torch.float32).to(self.device)
            for data in wiki_data
        ])  # Shape: (N, d)
        passages = [
            {"title": t, "context": c}
            for t, c in zip(titles, texts)
        ]
        return (emb_tensor, passages)
    
    
    def find_gt_passage(self, wiki_data_entity: str, prompt: str):
        # Step 1: Fetch entity data from Wikidata
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_data_entity}.json"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for entity {wiki_data_entity}")
        
        data = response.json()
        entity = data["entities"][wiki_data_entity]

        # Step 2: Get the English Wikipedia page title from sitelinks
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki")
        if not enwiki:
            raise Exception("No English Wikipedia link found for this entity.")
        
        title = enwiki["title"]
        wikipedia.set_lang("en")

        try:
            summary = wikipedia.summary(title)
            content = wikipedia.page(title).content
            url = wikipedia.page(title).url
        except wikipedia.exceptions.PageError:
            raise Exception("Wikipedia page not found.")
        except wikipedia.exceptions.DisambiguationError as e:
            raise Exception(f"Disambiguation error: {e.options}")

        # Step 4: Prepare the page data and combined text
        # page_data = {
        #     "title": title,
        #     "content": content[:4000],
        #     "summary": summary,
        #     "url": url
        # }
        page_text = f"Title: {title}\n{content[:4000]}"

        page_data = {
            "title": title,
            "context": page_text,
        }
        
        # Step 5: Embed and compute similarity
        page_embedding = self.embedding_adapter.encode(page_text)
        prompt_embedding = self.embedding_adapter.encode(prompt)
        sim_score = similarity(page_embedding, prompt_embedding)
        return [(page_data, sim_score)]
    
    
    def find_topk_relevant_pages(self, wiki_data, prompt, top_k=3) -> List[Tuple[str, float]]:
        """
        Return the top k pages (and their scores) with highest similarity to the prompt.
        """
        if not wiki_data:
            return []

        prompt_embedding = self.embedding_adapter.encode(prompt).to(self.device)
        if prompt_embedding.ndim == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)  # (1, hidden_size) tensor

        db_embeddings, title_text_dict = self.create_passage_db(wiki_data)  # (N, hidden_size) tensor, List[str]
        similarity_scores = similarities(db_embeddings, prompt_embedding)
        k = min(top_k, similarity_scores.shape[0])
        top_values, top_indices = torch.topk(similarity_scores.squeeze(), k)
        # returns tuple of {title: context} dict and sim_score of doc with prompt
        return [(title_text_dict[idx], float(top_values[i].item())) for i, idx in enumerate(top_indices)]


    def find_most_relevant_page(
        self, 
        wiki_data: List[Dict],
        prompt: str,
        count: int = 3
    ) -> Dict:
        if not wiki_data:
            return None

        # Get passage embeddings and the prompt embedding
        page_embeds = self.create_embeddings(wiki_data=wiki_data)  # (N, hidden_size)
        prompt_embed = self.embedding_adapter.encode(prompt)       # (hidden_size,) or (1, hidden_size)

        if prompt_embed.ndim == 1:
            prompt_embed = prompt_embed.unsqueeze(0)  # Make shape (1, hidden_size)

        # Normalize to unit vectors for cosine similarity
        page_embeds = F.normalize(page_embeds, p=2, dim=1)
        prompt_embed = F.normalize(prompt_embed, p=2, dim=1)

        # Compute cosine similarity
        similarities = torch.matmul(page_embeds, prompt_embed.T).squeeze()  # shape: (N,)

        # Find the index of the most similar page
        best_index = torch.argmax(similarities).item()
        return wiki_data[best_index]


    def search_query(self, prompt: str) -> str:
        text = 'You are a helpful assistant whose job it is to extract entities from the given string. Do not attempt to answer the question, your job is just to perform named entity recognition. As a point of reference, these are (proper) nouns in the string. For example: Who released the song "Smells Like Teen Spirit"? should return Smells Like Teen Spirit'
        text = text.format(prompt=prompt)
        return text

    
    def format_topk_wiki_answer(self, document_list: list) -> str:
        extracts = "\n".join([doc["context"] for doc in document_list])
        extracts = extracts.replace("{", "").replace("}", "")

        # Build the final text prompt
        text_template = (
            "You are a helpful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, "
            "or illegal content. You have retrieved the following extracts from the Wikipedia pages:\n{extracts}\n"
            "You are expected to give truthful and concise answers based on the previous extracts. If it doesn't include relevant information "
            "for the request just say so and don't make up false information.\n"
            "Keep the answers as concise as possible, does not have to be full sentences."
        )

        # Format the text with the combined extracts
        text = text_template.format(extracts=extracts)
        return text

    
    def search_entities_2(self, prompt: str):
        entity_list = self.NER(prompt)
        return list(set(ent.text for ent in entity_list.ents))
    
    
    def search_entities_NER(self, prompt: str):
        entity_list = self.NER(prompt)
        return [li.text for li in list(entity_list.ents)]


    def retrieve_wiki_data(self, prompt: str, **kwargs) -> List[Dict]:
        answer = self.search_entities_2(prompt)
        # get title, content (4000 characters), summary and page url of top 5 wiki pages
        wiki_data = self.wiki_helper.get_wiki_page(query=answer)[0] # first entity instance
        return wiki_data
    
    
    def retrieve_wiki_data_2(self, prompt: str, **kwargs) -> List[Dict]:
        # Get entities in query
        entities = self.search_entities_2(prompt)
        wiki_data = []
        
        # Create a ThreadPoolExecutor to fetch wiki data concurrently for each entity.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks concurrently
            future_to_entity = {
                executor.submit(self.wiki_helper.get_wiki_page, entity=entity): entity 
                for entity in entities
            }
            
            # As each future completes, extend the wiki_data list with its results.
            for future in concurrent.futures.as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    data = future.result()
                    wiki_data.extend(data)
                except Exception as e:
                    logging.error(f"Error processing entity '{entity}': {e}")
        
        seen = set()
        unique_wiki_data = []
        for entry in wiki_data:
            key = entry["title"]
            if key not in seen:
                seen.add(key)
                unique_wiki_data.append(entry)
        
        return unique_wiki_data


    def answer_using_wiki(
        self, model_name: str, prompt: str, extracts: str, title: str, **kwargs
    ) -> str:
        # input text is the prompt hydrated with documents
        input_text = self.format_wiki_answer(
            prompt=prompt, title=title, extracts=extracts
        )
        answer = self.generator.wiki_rag_completions(model_name, input_text, prompt)
        return answer
    
    def answer_using_wiki_2(
        self, model_name, root_prompt, rag_closest_matches):
        rag_closest_passage = [match[0] for match in rag_closest_matches] # textual content
        contriever_response = self.format_topk_wiki_answer(
            rag_closest_passage
        )
        answer = self.generator.wiki_rag_completions(model_name, contriever_response, root_prompt)
        return answer