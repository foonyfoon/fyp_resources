from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

class OAIEmbedAdapter:
    def encode(self, prompt):
        if type(prompt) == dict:
            prompt = str(prompt)
        return model.encode(prompt)
