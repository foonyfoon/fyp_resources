import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

# def similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
#     embedding1 = embedding1 / np.linalg.norm(embedding1)
#     embedding2 = embedding2 / np.linalg.norm(embedding2)
#     cosine_similarity = np.dot(embedding1, embedding2)

#     return cosine_similarity
def similarity(embedding1: Union[np.ndarray, torch.Tensor], 
               embedding2: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(embedding1, np.ndarray) and isinstance(embedding2, np.ndarray):
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))
    
    elif isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
        embedding1 = embedding1 / torch.norm(embedding1)
        embedding2 = embedding2 / torch.norm(embedding2)
        return float(torch.dot(embedding1, embedding2))
    
    else:
        raise TypeError()
    
    
def similarities(db_embeddings: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
    db_embeddings = F.normalize(db_embeddings, p=2, dim=1) # (n, h)
    prompt_embedding = F.normalize(prompt_embedding, p=2, dim=1) # size (m, h)
    sims = torch.matmul(db_embeddings, prompt_embedding.T) # (n, m)
    # if we have more than one prompt, reshape to (m, n, 1)
    if prompt_embedding.size(0) > 1:
        sims = sims.T.unsqueeze(-1) # (m, n, 1)

    return sims