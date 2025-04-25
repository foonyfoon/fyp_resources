from wiki_cache.models import WikipediaArticle
from wiki_cache.db import Session
from sqlalchemy import select
import numpy as np

def add_article(wiki_data, keywords, embedding_vector):
    session = Session()
    try:
        embedding_blob = embedding_vector.detach().cpu().numpy().astype(np.float32).tobytes()
        article = WikipediaArticle(
            title=wiki_data['title'],
            summary=wiki_data['summary'],
            url=wiki_data['url'],
            keywords=keywords,
            content=wiki_data['content'],
            embedding=embedding_blob,
            context=wiki_data['context'],
            links=wiki_data['links']
        )
        session.merge(article)
        session.commit()
    finally:
        session.close()
        Session.remove()

def get_article_by_title(title):
    session = Session()
    try:
        return session.query(WikipediaArticle).filter_by(title=title).first()
    finally:
        session.close()
        Session.remove()

def get_articles_by_entities(entities: list):
    session = Session()
    try:
        stmt = select(WikipediaArticle).where(WikipediaArticle.keywords.in_(entities))
        result = session.execute(stmt)
        return result.scalars().all()
    finally:
        session.close()
        Session.remove()

def clear_cache_db():
    session = Session()
    try:
        session.query(WikipediaArticle).delete()
        session.commit()
    finally:
        session.close()
        Session.remove()
