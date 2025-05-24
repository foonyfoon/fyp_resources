from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import random
import string

def generate_random_string(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length)).lower()
                
# TODO: idk why sqlite:///memory fails
formatted_date_time = datetime.now().strftime("%m%d_%H%M%S")
rnd_str = generate_random_string()  
engine = create_engine(
    f"sqlite:///wiki_{formatted_date_time}_{rnd_str}.db",
    connect_args={"check_same_thread": False},
    echo=False
)
Session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()