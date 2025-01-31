from tree.node import RootNode
import pickle

class ReadTree:
    def __init__(self, root_prompt, prev_state=None):
        self.root_prompt = root_prompt
        self.root = RootNode(root_prompt) if prev_state is None else prev_state["root"]
        self.thresholds = [] if prev_state is None else prev_state["thresholds"]
        self.prompt_list = [root_prompt] if prev_state is None else prev_state["prompt_list"]
        self.time_semantic = 0 if prev_state is None else prev_state["time_semantic"]
        self.time_syntactic = 0 if prev_state is None else prev_state["time_syntactic"]
        self.time_check = {} if prev_state is None else prev_state["time_check"]
        self.metrics = {} if prev_state is None else prev_state["metrics"]
        
    @staticmethod
    def load_read_tree(file_path):
        prev_state = {}
        with open(file_path, "rb") as file:
            prev_state = pickle.load(file)
        root_prompt = prev_state["root_prompt"]
        return ReadTree(root_prompt, prev_state=prev_state) 