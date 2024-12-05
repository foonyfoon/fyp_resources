from adapters.SemanticAdapter import SemanticAdapter
from model.engine import GemmaAdapter

model = GemmaAdapter("google/gemma-2-9b-it")
semantic_adapter = SemanticAdapter(model)
print("test sem_perturb...")
print(semantic_adapter.sem_perturb("Who is the composer who wrote eine kleine nachtmusik"))
print("test sem_check...")
print(semantic_adapter.sem_check("Who is the composer who wrote eine kleine nachtmusik"))
print("test sem_perturb_combined...")
print(semantic_adapter.sem_perturb_combined("Who is the composer who wrote eine kleine nachtmusik", 3))
