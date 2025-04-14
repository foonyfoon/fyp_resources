from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

class EmbedAdapter(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def encode(self, prompt):
        pass

class RobertaEmbedder(EmbedAdapter):
    def __init__(self):
        # Initialize the RoBERTa model
        self.model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    def encode(self, prompt):
        # Convert non-string inputs to string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Encode the prompt using the RoBERTa model
        return self.model.encode(prompt)

class ContrieverEmbedder(EmbedAdapter):
    def __init__(self):
        # Initialize the Contriever model
        self.model = SentenceTransformer('facebook/contriever')

    def encode(self, prompt):
        # Convert non-string inputs to string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Encode the prompt using the Contriever model
        return self.model.encode(prompt)
