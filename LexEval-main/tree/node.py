class Node:
    def __init__(self, type, prompt, parent=None, id=None):
        self.id = id
        self.type = type
        self.prompt = prompt
        self.parent = parent
        self.children = []
        self.embedding = None
        self.rag_closest_match = None
        self.rag_entities = None
        self.ner_entities = None
        self.answers = {}
        self.metadata = {}
        

    def add_child(self, child):
        child.id = f"{self.id}.{len(self.children)}"
        self.children.append(child)
        child.parent = self
        
    def move_to_cpu(self):
        self.embedding = self.embedding.cpu()
        for child in self.children:
            child.move_to_cpu()
            
    def move_to_device(self, device):
        self.embedding = self.embedding.to(device)
        for child in self.children:
            child.move_to_cpu()
            

class RootNode(Node):
    def __init__(self,
                 prompt,
                 wiki_title=[],
                 complexity_score=0,
                 fk_score=0,
                 dc_score=0):
        super().__init__("root", prompt, parent=None)
        self.id = "1"
        self.root_similarity_score = 1.0
        self.complexity_score = complexity_score,
        self.fk_score = fk_score,
        self.dc_score = dc_score
        self.wiki_title = wiki_title
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "prompt": self.prompt,
            "parent": self.parent.id if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "rag_closest_match": self.rag_closest_match,
            "rag_entities": self.rag_entities,
            "ner_entities": self.ner_entities,
            "answers": self.answers,
            "root_similarity_score": self.root_similarity_score,
            "complexity_score": self.complexity_score,
            "fk_score": self.fk_score,
            "dc_score": self.dc_score,
            "wiki_title": self.wiki_title
        }


class SemanticNode(Node):
    def __init__(
        self,
        prompt,
        semantic_similarity_score,
        root_similarity_score,
        threshold,
        embedding,
        rag_closest_match,
        rag_entities,
        ner_entities,
        wiki_title,
        parent=None,
        complexity_score=0,
        fk_score=0,
        dc_score=0,
        
    ):
        super().__init__("semantic", prompt, parent)
        self.semantic_similarity_score = semantic_similarity_score
        self.root_similarity_score = root_similarity_score
        self.threshold = threshold
        self.embedding = embedding
        self.rag_closest_match = rag_closest_match
        self.rag_entities = rag_entities
        self.ner_entities = ner_entities
        self.complexity_score = complexity_score
        self.fk_score = fk_score
        self.dc_score = dc_score
        self.wiki_title = wiki_title

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "prompt": self.prompt,
            "parent": self.parent.id if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "rag_closest_match": self.rag_closest_match,
            "rag_entities": self.rag_entities,
            "ner_entities": self.ner_entities,
            "answers": self.answers,
            "root_similarity_score": self.root_similarity_score,
            "semantic_similarity_score": self.semantic_similarity_score,
            "threshold": self.threshold,
            "complexity_score": self.complexity_score,
            "fk_score": self.fk_score,
            "dc_score": self.dc_score,
            "wiki_title": self.wiki_title
        }
        

class SyntacticNode(Node):
    def __init__(
        self,
        prompt,
        syntax_similarity_score,
        threshold,
        rag_closest_match,
        wiki_title,
        rag_entities,
        ner_entities,
        parent=None,
    ):
        super().__init__("syntactic", prompt, parent)
        self.syntax_similarity_score = syntax_similarity_score
        self.threshold = threshold
        self.rag_closest_match = rag_closest_match
        self.rag_entities = rag_entities
        self.ner_entities = ner_entities
        self.wiki_title = wiki_title
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "prompt": self.prompt,
            "parent": self.parent.id if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "rag_closest_match": self.rag_closest_match,
            "rag_entities": self.rag_entities,
            "ner_entities": self.ner_entities,
            "answers": self.answers,
            "syntax_similarity_score": self.syntax_similarity_score,
            "threshold": self.threshold,
            "wiki_title": self.wiki_title
            
        }
