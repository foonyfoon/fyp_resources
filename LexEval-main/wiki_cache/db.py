from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine("sqlite:///wikipedia.db",
                       connect_args={"check_same_thread": False},
                       echo=False)
Session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()