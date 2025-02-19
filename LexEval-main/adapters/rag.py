import json
import ast
from typing import List, Dict, Any, Tuple

import wikipedia
import textwrap
import spacy
import torch

from adapters.SemanticAdapter import SemanticAdapter
from adapters.OAI_Embeddings import OAIEmbedAdapter

from similarity.cosine_similarity import similarity
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel


class RAGAgent:
    def __init__(
        self,
        llm_model,
        **kwargs,
    ):
        self.llm_adapter = SemanticAdapter(llm_model)
        self.query = None
        self.embedding_adapter = OAIEmbedAdapter()
        self.NER = spacy.load(
            "en_core_web_trf"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever")

    # def answer_using_wiki(
    #     self, prompt: str, extracts: str, title: str, **kwargs
    # ) -> str:
    #     input_text = self.format_wiki_answer(
    #         prompt=prompt, title=title, extracts=extracts
    #     )
    #     answer = self.llm_adapter.wiki_rag_completions(input_text, prompt)
    #     return answer


    def create_embeddings(
        self, 
        wiki_data: List[Dict],
    ):
        if not wiki_data:
            return None
        res = []
        for data in wiki_data:
            res.append(self.embedding_adapter.encode(data))
        embeddings = []
        for i in range(len(wiki_data)):
            embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))

        return embeddings


    def create_retriever(self, wiki_data: List[Dict]):
        if not wiki_data:
            return None
        retriever = BM25Retriever.from_documents(
            [Document(page_content=json.dumps(data["title"])) for data in wiki_data]
        )
        return retriever


    def retrieve_bm25(self, BM25_retriever, prompt: str):
        if not BM25_retriever:
            return None
        return BM25_retriever.invoke(prompt)


    def contriever_retriever(self, wiki_data):
        '''
        Computing the mean-pooled embedding across the non-padded tokens to get one vector per input.
        '''
        if not wiki_data:
            return None

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        inputs = self.tokenizer(
            [wiki_data],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embeddings = self.model(**inputs)
        return mean_pooling(embeddings[0], inputs["attention_mask"])


    def create_contriever_db(
        self, wiki_data: List[Dict],
    ):
        if not wiki_data:
            return None
        # print("Creating pages retrieval database...")
        texts = [data["title"] for data in wiki_data]
        res = []
        for text in texts:
            res.append(self.contriever_retriever(text))

        embeddings = []
        for i in range(len(wiki_data)):
            embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))
        return embeddings


    def find_closest_contriever_match(self, wiki_data: List[Dict], prompt: str):
        if not wiki_data:
            return None
        prompt_embedding = self.contriever_retriever(prompt)
        closest_match = None
        max_similarity = 0
        contriever_db = self.create_contriever_db(wiki_data)
        for i in range(len(contriever_db)):
            if contriever_db[i][1] is None:
                similarity_score = 0
            else:
                similarity_score = prompt_embedding @ contriever_db[i][1].T
            # print(similarity_score)
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                closest_match = contriever_db[i][2]
        return closest_match

    def find_top3_contriever_matches(self, wiki_data, prompt):
        """
        Return the top 3 pages (and their scores) with highest similarity to the prompt.
        """
        if not wiki_data:
            return []
        prompt_embedding = self.contriever_retriever(prompt)
        contriever_db = self.create_contriever_db(wiki_data) # (f"embedding_id_{i}", embedding, {"text": wiki_data[i]})
        
 
        db_embeddings = [entry[1] for entry in contriever_db]
        text_data = [entry[2]["text"] for entry in contriever_db]

        # Convert the list of embeddings to a single Tensor of shape (N_wiki_data, d).
        db_embeddings = torch.stack(db_embeddings, dim=0)
        similarities = torch.matmul(db_embeddings, prompt_embedding.T)

        k = (min(3,similarities.shape[0]))
        top_values, top_indices = torch.topk(similarities.squeeze(), k)
        top3_matches = [
            (text_data[idx], float(top_values[i].item()))
            for i, idx in enumerate(top_indices)
        ]
        return top3_matches


    def create_summary_db(
       self, wiki_data: List[Dict],
    ):
        if not wiki_data:
            return None
        # print("Creating pages retrieval database...")
        texts = [data["summary"] for data in wiki_data]
        res = []
        for text in texts:
            res.append(self.embedding_adapter.encode(text))

        embeddings = []
        for i in range(len(wiki_data)):
            embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))
        return embeddings


    def find_most_relevant_page(
        self, 
        wiki_data: List[Dict],
        prompt: str,
        count=3
    ) -> Dict:
        if not wiki_data:
            return None
        page_embeds = self.create_embeddings(wiki_data=wiki_data)
        prompt_embed = self.embedding_adapter.encode(prompt)

        max_similarity = 0

        page = None
        for i in range(len(page_embeds)):
            emb = page_embeds[i]
            # print(emb)
            dist = similarity(prompt_embed, emb[1])
            if dist > max_similarity:
                max_similarity = dist
                page = wiki_data[i]
        return page


    def find_most_relevant_passages(
        self, wiki_data: Dict, prompt: str, k: int = 3
    ) -> List[str]:
        if not wiki_data:
            return None
        page = wiki_data["title"]
        passages = page.split("\n\n")
        prompt_embed = self.embedding_adapter.encode(prompt)
        passage_embeds = self.create_embeddings(wiki_data=passages)
        distances = []
        for i in range(len(passage_embeds)):
            emb = passage_embeds[i][1]
            dist = similarity(prompt_embed, emb)
            distances.append(dist)

        sorted_passages = [
            passage
            for _, passage in sorted(
                zip(distances, passages), key=lambda pair: pair[0], reverse=True
            )
        ]
        return sorted_passages[:k]


    def get_wiki_data(self, query: str) -> List[Dict]:
        wiki_data = []
        # print(query)
        for search_result in wikipedia.search(query, results=5):
            try:
                page = wikipedia.page(title=search_result)
            except:
                continue
            wiki_data.append(
                {
                    "title": search_result,
                    "content": page.content[:4000],
                    "summary": page.summary,
                    "url": page.url,
                }
            )
        return wiki_data


    def search_query(self, prompt: str) -> str:
        text = f'You are a helpful assistant whose job it is to extract entities from the given string. Do not attempt to answer the question, your job is just to perform named entity recognition. As a point of reference, these are (proper) nouns in the string. For example: Who released the song "Smells Like Teen Spirit"? should return Smells Like Teen Spirit'
        text = text.format(prompt=prompt)
        return text
    
    def search_query_2(self) -> List[str]:
        text = (
            f'You are a helpful assistant whose job is to extract named entities from the given string. '
            f'Do not answer the question itself. Only return a comma-separated list of named entities. '
            f'Example: "Which American born Sinclair won the Nobel Prize for Literature in 1930?" should '
            'return a json of the following format: '
            '{"res" : [("Sinclair", "Person"), ("Nobel Prize for Literature", "Award"), ("1930", "Date")]}'
        )
        return text

    
    def format_topk_wiki_answer(self, prompt: str, document_list: list) -> str:
        excerpt = "\n".join(f'{doc[0]['title']}: {doc[0]['content'].replace("{", "").replace("}", "")}'for doc in document_list)
        text = " ".join(
            [f"You are a helpful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous,",
            f"or illegal content. You have retrieved the following extracts from the Wikipedia pages: \n{excerpt}.""\nYou are expected to give ",
            f"truthful and concise answers based on the previous extracts. If it doesn't include relevant information for the request just say so ",
            f"and don't make up false information. \n",
            f"Keep the answers as concise as possible, does not have to be full sentences. For example, for the question: What is Scooter Braun's occupation? Your response should be:",
            f"Talent manager, Entrepreneur, Record executive, Film and television producer."])
        text = text.format(prompt=prompt, excerpt=excerpt)
        return text


    def format_wiki_answer(self, prompt: str, title: str, extracts: str) -> str:
        
        extracts = extracts.replace("{", "")
        extracts = extracts.replace("}", "")
        text = " ".join(
            [f"You are a helpful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous,",
            f"or illegal content. You have retrieved the following extracts from the Wikipedia page {title}: {extracts}.\nYou are expected to give ",
            f"truthful and concise answers based on the previous extracts. If it doesn't include relevant information for the request just say so ",
            f"and don't make up false information. \n",
            f"Keep the answers as concise as possible, does not have to be full sentences. For example, for the question: What is Scooter Braun's occupation? Your response should be:",
            f"Talent manager, Entrepreneur, Record executive, Film and television producer."])
        text = text.format(prompt=prompt, title=title, extracts=extracts)
        return text


    def display_result(
        answer: str, metadata: Dict, extracts: List[str], max_row_length: int = 80
    ) -> None:
        out_string = ""
        for i in range(len(extracts)):
            out_string = out_string + f"Extract_{i}:{extracts[i]} \n"
        output = (
            textwrap.fill(answer, width=max_row_length)
            # + " \n\n"
            # + "RETRIEVED WIKIPEDIA PAGE: \n"
        )
        metadata_info = " \n".join(
            [
                key + ": " + val
                for key, val in metadata.items()
                if key not in ["content", "summary"]
            ]
        )
        # print(
        #     output
        #     # + metadata_info
        #     # + "\nRetrieved extracts: \n"
        #     # + textwrap.fill(out_string, width=max_row_length)
        # )


    def search_entities(self, prompt: str):
        input_text = self.search_query(prompt)
        answer = self.llm_adapter.wiki_rag_completions('gpt-3.5-turbo', input_text, prompt)
        return answer
    
    def search_entities_2(self, prompt: str):
        input_text = self.search_query_2()
        answer = self.llm_adapter.wiki_rag_completions('gpt-3.5-turbo', input_text, prompt)
        try:
            dict_data = ast.literal_eval(answer)
            if isinstance(dict_data, dict) and "res" in dict_data:
                parsed_answer = dict_data["res"]
                if isinstance(parsed_answer, list) and all(
                                isinstance(item, tuple) and len(item) == 2 and
                                isinstance(item[0], str) and isinstance(item[1], str)
                                for item in parsed_answer
                ):
                    return parsed_answer
            else:
                raise ValueError("Parsed answer is not in the expected format (list of (str, str) tuples)")
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse entity tuples from response: {answer}") from e

    def search_entities_NER(self, prompt: str):
        entity_list = self.NER(prompt)
        return [li.text for li in list(entity_list.ents)]


    def retrieve_wiki_data(self, prompt: str, **kwargs) -> List[Dict]:
        # get entity in query
        answer = self.search_entities(prompt)
        # get title, content (4000 characters), summary and page url of top 5 wiki pages
        wiki_data = self.get_wiki_data(query=answer)
        return wiki_data
    
    def retrieve_wiki_data_2(self, prompt: str, **kwargs) -> List[Dict]:
        # get entity in query
        answers = self.search_entities_2(prompt)
        # get title, content (4000 characters), summary and page url of top 5 wiki pages
        wiki_data = []
        for entity, _ in answers:
            data = self.get_wiki_data(query=entity)
            wiki_data.extend(data)
        return wiki_data


    def answer_using_wiki(
        self, model_name: str, prompt: str, extracts: str, title: str, **kwargs
    ) -> str:
        # input text is the prompt hydrated with documents
        input_text = self.format_wiki_answer(
            prompt=prompt, title=title, extracts=extracts
        )
        answer = self.llm_adapter.wiki_rag_completions(model_name, input_text, prompt)
        return answer


# class RAGAgent:
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         self.llm_adapter = SemanticAdapter(model)
#         self.query = None

#     def answer_using_wiki(
#         self, prompt: str, extracts: str, title: str, **kwargs
#     ) -> str:
#         input_text = format_wiki_answer(
#             prompt=prompt, title=title, extracts=extracts
#         )
#         answer = self.llm_adapter.wiki_rag_completions(input_text, prompt)
#         return answer

#     def __call__(
#         self,
#         prompt: str,
#     ) -> Tuple[Any, dict, List[str]]:
#         n_extracts = 3

#         wiki_data = retrieve_wiki_data(prompt=prompt)
#         page = find_most_relevant_page(wiki_data=wiki_data, prompt=prompt)
#         passages = find_most_relevant_passages(
#             wiki_data=page, prompt=prompt, k=n_extracts
#         )
#         joint_passages = " \n ".join(passages)
#         # print(f"Retrieved Wikipedia page:\n Title: {page['title']}")
#         answer = self.answer_using_wiki(
#             prompt=prompt, extracts=joint_passages, title=page["title"]
#         )
#         return answer, page, passages
