from sqlalchemy import Column, String, Text, LargeBinary, JSON
from wiki_cache.db import Base

class WikipediaArticle(Base):
    __tablename__ = 'wikipedia_articles'

    title = Column(String, primary_key=True)
    keywords = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    links = Column(JSON, nullable=False)
    