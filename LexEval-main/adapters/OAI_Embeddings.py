from abc import ABC, abstractmethod
import torch
import gc
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def clear_cache():
    # Clear any leftover tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    
class EmbedAdapter(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def encode(self, prompt):
        pass

class RobertaEmbedder(EmbedAdapter):
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    def encode(self, prompts: str | list[str]) -> torch.Tensor:
        is_single = isinstance(prompts, str)
        prompts = [prompts] if is_single else prompts
        try:
            embeddings = self.model.encode(prompts, convert_to_tensor=True)
        finally:
            clear_cache()
        if is_single:
            return embeddings[0] # (,hidden_size)
        else:
            return embeddings  # (batch_size, hidden_size)

class ContrieverEmbedder(EmbedAdapter):
    def __init__(self):
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    def encode(self, prompts: str | list[str]) -> torch.Tensor:
        def mean_pooling(token_embeddings, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts
        
        is_single = isinstance(prompts, str)
        prompts = [prompts] if is_single else prompts

        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        finally:
            clear_cache()

        if is_single:
            return embeddings[0] # (,hidden_size)
        else:
            return embeddings  # (batch_size, hidden_size)
