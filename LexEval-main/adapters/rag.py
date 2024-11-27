import json
from typing import List, Dict, Any, Tuple

import wikipedia
import textwrap
import spacy

from adapters.SemanticAdapter import SemanticAdapter

from adapters.OAI_Embeddings import OAIEmbedAdapter

from similarity.cosine_similarity import similarity
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel

embedding_adapter = OAIEmbedAdapter()
llm_adapter = SemanticAdapter()
NER = spacy.load(
    "en_core_web_trf"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")


def create_embeddings(
    wiki_data: List[Dict],
):
    if not wiki_data:
        return None
    res = []
    for data in wiki_data:
        res.append(embedding_adapter.encode(data))
    embeddings = []
    for i in range(len(wiki_data)):
        embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))

    return embeddings


def create_retriever(wiki_data: List[Dict]):
    if not wiki_data:
        return None
    retriever = BM25Retriever.from_documents(
        [Document(page_content=json.dumps(data["title"])) for data in wiki_data]
    )
    return retriever


def retrieve_bm25(BM25_retriever, prompt: str):
    if not BM25_retriever:
        return None
    return BM25_retriever.invoke(prompt)


def contriever_retriever(wiki_data):
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

    inputs = tokenizer(
        [wiki_data],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    embeddings = model(**inputs)
    return mean_pooling(embeddings[0], inputs["attention_mask"])


def create_contriever_db(
    wiki_data: List[Dict],
):
    if not wiki_data:
        return None
    # print("Creating pages retrieval database...")
    texts = [data["title"] for data in wiki_data]
    res = []
    for text in texts:
        res.append(contriever_retriever(text))

    embeddings = []
    for i in range(len(wiki_data)):
        embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))
    return embeddings


def find_closest_contriever_match(wiki_data: List[Dict], prompt: str):
    if not wiki_data:
        return None
    prompt_embedding = contriever_retriever(prompt)
    closest_match = None
    max_similarity = 0
    contriever_db = create_contriever_db(wiki_data)
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


def create_summary_db(
    wiki_data: List[Dict],
):
    if not wiki_data:
        return None
    # print("Creating pages retrieval database...")
    texts = [data["summary"] for data in wiki_data]
    res = []
    for text in texts:
        res.append(embedding_adapter.encode(text))

    embeddings = []
    for i in range(len(wiki_data)):
        embeddings.append((f"embedding_id_{i}", res[i], {"text": wiki_data[i]}))
    return embeddings


def find_most_relevant_page(
    wiki_data: List[Dict],
    prompt: str,
) -> Dict:
    if not wiki_data:
        return None
    page_embeds = create_embeddings(wiki_data=wiki_data)
    prompt_embed = embedding_adapter.encode(prompt)

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
    wiki_data: Dict, prompt: str, k: int = 3
) -> List[str]:
    if not wiki_data:
        return None
    page = wiki_data["title"]
    passages = page.split("\n\n")
    prompt_embed = embedding_adapter.encode(prompt)
    passage_embeds = create_embeddings(wiki_data=passages)
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


def get_wiki_data(query: str) -> List[Dict]:
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


def search_query(prompt: str) -> str:
    text = f'You are a helpful assistant whose job it is to extract entities from the given string. Do not attempt to answer the question, your job is just to perform named entity recognition. As a point of reference, these are (proper) nouns in the string. For example: Who released the song "Smells Like Teen Spirit"? should return Smells Like Teen Spirit'
    text = text.format(prompt=prompt)
    return text


def format_wiki_answer(prompt: str, title: str, extracts: str) -> str:
    extracts = extracts.replace("{", "")
    extracts = extracts.replace("}", "")
    text = f"You are a helpful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. You have retrieved the following extracts from the Wikipedia page {title}: {extracts}. You are expected to give truthful and concise answers based on the previous extracts. If it doesn't include relevant information for the request just say so and don't make up false information. "
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


def search_entities(prompt: str):
    input_text = search_query(prompt)
    # print(input_text)
    answer = llm_adapter.wiki_rag_completions('gpt-3.5-turbo', input_text, prompt)
    return answer


def search_entities_NER(prompt: str):
    entity_list = NER(prompt)
    return [li.text for li in list(entity_list.ents)]


def retrieve_wiki_data(prompt: str, **kwargs) -> List[Dict]:
    # print(input_text)
    answer = search_entities(prompt)
    # print(answer)
    # print("Searching for pages using query:", answer)
    wiki_data = get_wiki_data(query=answer)
    # print(
    #     "Retrieved pages:\n",
    #     "\n".join([data["title"] for data in wiki_data]),
    # )
    return wiki_data


def answer_using_wiki(
    model_name: str, prompt: str, extracts: str, title: str, **kwargs
) -> str:
    input_text = format_wiki_answer(
        prompt=prompt, title=title, extracts=extracts
    )
    answer = llm_adapter.wiki_rag_completions(model_name, input_text, prompt)
    return answer


class RAGAgent:
    def __init__(
        self,
        **kwargs,
    ):
        self.llm_adapter = SemanticAdapter()
        self.query = None

    def answer_using_wiki(
        self, prompt: str, extracts: str, title: str, **kwargs
    ) -> str:
        input_text = format_wiki_answer(
            prompt=prompt, title=title, extracts=extracts
        )
        answer = self.llm_adapter.wiki_rag_completions(input_text, prompt)
        return answer

    def __call__(
        self,
        prompt: str,
    ) -> Tuple[Any, dict, List[str]]:
        n_extracts = 3

        wiki_data = retrieve_wiki_data(prompt=prompt)
        page = find_most_relevant_page(wiki_data=wiki_data, prompt=prompt)
        passages = find_most_relevant_passages(
            wiki_data=page, prompt=prompt, k=n_extracts
        )
        joint_passages = " \n ".join(passages)
        # print(f"Retrieved Wikipedia page:\n Title: {page['title']}")
        answer = self.answer_using_wiki(
            prompt=prompt, extracts=joint_passages, title=page["title"]
        )
        return answer, page, passages
