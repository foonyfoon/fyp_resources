class Node:
    def __init__(self, type, prompt, parent=None, id=None):
        self.id = id
        self.type = type
        self.prompt = prompt
        self.parent = parent
        self.children = []
        self.embedding = None
        self.rag_closest_match = None
        self.contriever_closest_match = None
        self.bm25_closest_match = None
        self.rag_entities = None
        self.ner_entities = None
        self.answers = {}

    def add_child(self, child):
        child.id = f"{self.id}.{len(self.children)}"
        self.children.append(child)
        child.parent = self


class RootNode(Node):
    def __init__(self, prompt):
        super().__init__("root", prompt, parent=None)
        self.id = "1"
        self.root_similarity_score = 1.0


class SemanticNode(Node):
    def __init__(
        self,
        prompt,
        semantic_similarity_score,
        root_similarity_score,
        threshold,
        embedding,
        rag_closest_match,
        contriever_closest_match,
        bm25_closest_match,
        rag_entities,
        ner_entities,
        parent=None,
    ):
        super().__init__("semantic", prompt, parent)
        self.semantic_similarity_score = semantic_similarity_score
        self.root_similarity_score = root_similarity_score
        self.threshold = threshold
        self.embedding = embedding
        self.rag_closest_match = rag_closest_match
        self.rag_entities = rag_entities
        self.contriever_closest_match = contriever_closest_match
        self.bm25_closest_match = bm25_closest_match
        self.ner_entities = ner_entities


class SyntacticNode(Node):
    def __init__(
        self,
        prompt,
        syntax_similarity_score,
        threshold,
        rag_closest_match,
        contriever_closest_match,
        bm25_closest_match,
        rag_entities,
        ner_entities,
        parent=None,
    ):
        super().__init__("syntactic", prompt, parent)
        self.syntax_similarity_score = syntax_similarity_score
        self.threshold = threshold
        self.rag_closest_match = rag_closest_match
        self.rag_entities = rag_entities
        self.contriever_closest_match = contriever_closest_match
        self.bm25_closest_match = bm25_closest_match
        self.ner_entities = ner_entities
