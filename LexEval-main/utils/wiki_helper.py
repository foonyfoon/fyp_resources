from wiki_cache.cache import add_article, get_article_by_title, get_articles_by_entities
from adapters.OAI_Embeddings import EmbedAdapter

import logging
import random
import time

import numpy as np
import wikipedia
from wikipedia import WikipediaPage
import torch
import requests

class WikiHelper:
    def __init__(self, encoder):
        self.encoder: EmbedAdapter = encoder
        
    def get_wiki_page(self, entity: str, results_num: int = 2) -> list:
        wiki_data = []

        # 1. Try fetching from the cache
        articles = get_articles_by_entities([entity])
        if articles:
            for article in articles:
                embedding_vector = np.frombuffer(article.embedding, dtype=np.float32)
                embedding_vector = torch.from_numpy(embedding_vector)
                wiki_data.append({
                    "title": article.title,               # str
                    "content": article.content[:4000],    # str (truncated to 4000 chars)
                    "summary": article.summary,           # str
                    "context": article.context,           # str
                    "links": article.links,               # list of str
                    "url": article.url,                   # str (URL)
                    "embedding": embedding_vector,        # torch.Tensor (1D vector)
                })

            return wiki_data

        # 2. Otherwise, fetch from Wikipedia
        search_results = wikipedia.search(entity, results=results_num)
        for result_title in search_results:
            page_data = self.fetch_wiki_page_with_retry(result_title)
            if page_data is None:
                continue

            # Generate embedding and context
            context = f"Title: {page_data['title']}\n{page_data['content']}"
            embedding_vector = self.encoder.encode(context)
            page_data["context"] = context
            page_data["keywords"] = entity
            page_data["embedding"] = embedding_vector
            add_article(page_data, entity, embedding_vector)
            wiki_data.append(page_data)

        return wiki_data

    def fetch_wiki_page_with_retry(self, page_title: str, max_retries=3, base_delay=1, max_delay=8):
        """
        Attempts to fetch a Wikipedia page for the given title with retry logic.
        If a DisambiguationError or PageError occurs, returns None immediately.
        For other exceptions, retries with exponential backoff and jitter.
        """
        article = get_article_by_title(page_title)
        if article:
            # If the article is found in the cache, return it
            # Convert the binary data back to a numpy array
            embedding_vector = np.frombuffer(article.embedding, dtype=np.float32)
            return {
                "title": article.title,
                "content": article.content[:4000],
                "summary": article.summary,
                "url": article.url,
                "context": article.context,
                "links": article.links,
                "embedding": embedding_vector
            }
        else:
            attempt = 0
            while True:
                if attempt >= max_retries:
                    logging.error(f"Max retries exceeded for '{page_title}'.")
                    return None
                try:
                    # Attempt to fetch the Wikipedia page
                    page = wikipedia.page(page_title)
                    return {
                        "title": page.title,
                        "content": page.content[:4000],
                        "summary": page.summary,
                        "url": page.url,
                        "links": page.links,
                    }
                except wikipedia.exceptions.DisambiguationError as e:
                    # selects random page since extraneous info does not have to be exact
                    logging.error(f"wikipedia.exceptions.DisambiguationError for '{page_title}'. picking random")
                    s = random.choice(e.options)
                    
                    page = wikipedia.page(s)
                    return {
                        "title": page.title,
                        "content": page.content[:4000],
                        "summary": page.summary,
                        "url": page.url,
                        "links": page.links,
                    }
                except wikipedia.exceptions.PageError:
                    logging.error(f"wikipedia.exceptions.PageError for '{page_title}'.")
                    return None
                except Exception as e:
                    attempt += 1
                    delay = min(base_delay * 2 ** attempt, max_delay)
                    delay += random.uniform(0, 1)
                    logging.info(f"Error fetching page '{page_title}': {e}. Retrying in {delay:.2f} seconds (attempt {attempt}/{max_retries}).")
                    time.sleep(delay)


def get_exact_page_from_entity(wiki_title: str) -> WikipediaPage:
    # Fetch Wikipedia page details (summary, content, and URL)
    try:
        page = wikipedia.page(wiki_title, redirect=False, auto_suggest=False)
        return page
    except wikipedia.exceptions.PageError:
        raise Exception("Wikipedia page not found.")
    except wikipedia.exceptions.DisambiguationError as e:
        raise Exception(f"get_exact_page_from_entity Disambiguation error: {e.options}")