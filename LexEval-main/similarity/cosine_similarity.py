import numpy as np


def similarity(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    cosine_similarity = np.dot(embedding1, embedding2)

    return cosine_similarity
