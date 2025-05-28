from typing import Tuple
from collections import deque
import textstat
import os
from tree.tree import Tree
from tree.node import Node, TerminalNode
from adapters.SemanticPerturb import PromptPackage
from adapters.TerminalPerturb import TerminalPerturber
from similarity.cosine_similarity import similarity
from model.engine import LLMAdapter
from adapters.OAI_Embeddings import RobertaEmbedder
import utils.constants as constants
from itertools import repeat
import pickle
import argparse

'''
each question, answer tree has new value:
- is terminal (node)
- terminal_preturb applied (list of string)
'''

# ########### set ###########
start_idx = 0
end_idx = 170

# terminal_type = 'sg_dialect'
# dataset_name = "TQA"
# stategy_path = "para"

terminal_type = 'position'
dataset_name = "TQA"
stategy_path = "para-prefix"
# ###########################
missing_tree_path = []
gen_modelIds = constants.MODELS

def get_term_perturb(term_type: str) -> list:
    if term_type == "sg_dialect":
        from adapters.TerminalPerturb import DialectPerturber
        dialect = "sg"
        term_perturber = DialectPerturber(dialect)
        perturbers = [term_perturber]

    elif term_type == "position":
        from adapters.TerminalPerturb import PositionPerturber
        positions = ["suffix", "middle"]
        perturbers = [PositionPerturber(position) for position in positions]

    else:
        raise ValueError(f"Unknown terminal perturbation type: {term_type}")
    
    return perturbers


def save_tree(file_path, **kwargs):
    # Make sure the root node is moved to CPU
    if "root" in kwargs and hasattr(kwargs["root"], "move_to_cpu"):
        kwargs["root"].move_to_cpu()

    # Prepare node dict from kwargs
    node = {
        "root": kwargs["root"],
        "thresholds": kwargs["thresholds"],
        "prompt_list": kwargs["prompt_list"],
        "time_semantic": kwargs["time_semantic"],
        "time_syntactic": kwargs["time_syntactic"],
        "time_check": kwargs["time_check"],
        "metrics": kwargs["metrics"],
        "root_prompt": kwargs["root_prompt"],
        "possible_answers": kwargs["possible_answers"],
        "rag_entities": kwargs["rag_entities"],
        "ner_entities": kwargs["ner_entities"],
        "rag_closest_match": kwargs["root"].rag_closest_match,
        "gt_passage": kwargs["gt_passage"],
    }
    print(node)

    # Ensure directory exists
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save to file
    with open(file_path, "wb") as file:
        pickle.dump(node, file)
        
def generate_terminal_node(
    tree: Tree,
    parent_node: Node,
    term_perturber: TerminalPerturber,
    use_parent_passage=True,
) -> Tuple[TerminalNode, bool]:
    # take parent closest match
    if not use_parent_passage:
        # TODO: hehe
        raise NotImplementedError()
    new_state = {**parent_node.metadata}
    perturb_pkg = PromptPackage(text=parent_node.prompt, state=new_state)
    try:
        new_perturb_pkg: PromptPackage = term_perturber.terminal_perturb(perturb_pkg)
        # unpack results
        perturbation = new_perturb_pkg.text  # the new prompt
        perturb_state = new_perturb_pkg.state  # metadata collected by perturbers
        is_valid = perturb_state.get("is_valid", True)
    except RuntimeError as e:
        print(f"RuntimeError: {e} from prompt {perturb_pkg.text}\n {perturb_pkg.state}")
        is_valid = False
        perturb_state = perturb_pkg.state
        perturbation = perturb_pkg.text
    except AssertionError as e:
        print(f"AssertionError: {e} from prompt {perturb_pkg.text}\n {perturb_pkg.state}")
        is_valid = False
        perturb_state = perturb_pkg.state
        perturbation = perturb_pkg.text

    if is_valid:
        perturb_embedding = tree.embed_model.encode(perturbation)
        parent_embedding = parent_node.embedding.to("cuda:0")
        root_embedding = tree.root.embedding.to("cuda:0")
        sem_sim = similarity(perturb_embedding, parent_embedding)
        root_sim = similarity(perturb_embedding, root_embedding)
        rag_closest_match = parent_node.rag_closest_match
        rag_entities = parent_node.rag_entities
        ner_entities = parent_node.ner_entities
        
        wiki_title = parent_node.wiki_title
        fk_score = textstat.flesch_kincaid_grade(perturbation)
        dc_score = textstat.dale_chall_readability_score(perturbation)
        complexity_score = (fk_score + dc_score) / 2
        term_node = TerminalNode(
                    perturbation,
                    sem_sim,
                    root_sim,
                    tree.embed_model.encode(perturbation),
                    rag_closest_match,
                    rag_entities,
                    ner_entities,
                    wiki_title=wiki_title,
                    parent=parent_node,
                    fk_score=fk_score,
                    dc_score=dc_score,
                    complexity_score=complexity_score,
                )
        term_node.metadata.update(perturb_state)
        print(term_node.metadata)
    else:
        term_node = None
    return (term_node, is_valid)

def apply_terminal_preturb(term_perturber: TerminalPerturber, tree: Tree):
    type_name = term_perturber.name
    root = tree.root
    level = 0
    queue = deque([(root, level)])

    while queue:  # BFS
        node, curr_level = queue.popleft()
        if isinstance(node, TerminalNode):
            continue  # Skip terminal nodes
        elif isinstance(node, Node) and type_name in node.metadata.get(
            "terminal_applied", []
        ):
            continue  # Skip processed nodes
        elif isinstance(node, Node):
            # Generate a terminal node based on the current node
            term, is_valid = generate_terminal_node(tree, node, term_perturber, use_parent_passage=True)
            if is_valid:
                # Update node's metadata to include the type_name
                type_names = node.metadata.get("terminal_applied", [])
                if not isinstance(type_names, list):
                    type_names = [type_names]
                if type_name not in type_names:
                    type_names.append(type_name)

                node.metadata["terminal_applied"] = type_names

                # Set the metadata for the terminal node
                term.metadata = {
                    **term.metadata,
                    "level": curr_level
                }
            # Add children of the current node to the queue
            for child in node.children:
                queue.append((child, curr_level + 1))
            
            if is_valid: # Append the terminal node as a child
                node.add_child(term)

def process_tree(tree: Tree, tree_idx, model_name):
    tree.run_check_pop_qa_batched(tree_idx, model_name)

def get_generator(modelId: str) -> LLMAdapter:
    model: LLMAdapter = None
    if "gemma-3" in modelId.lower():
        from model.engine import Gemma3Adapter
        model = Gemma3Adapter(modelId)
    elif "gemma" in modelId.lower():
        from model.engine import GemmaAdapter
        model = GemmaAdapter(modelId)
    elif "mistralai" in modelId.lower():
        from model.engine import MistralInstructAdapter
        model = MistralInstructAdapter(modelId)
    elif "mistral.mistral-7b-instruct-v0:2" in modelId.lower():
        from model.engine import MistralInstructAwsAdapter
        model = MistralInstructAwsAdapter(modelId)
    else:
        raise NotImplementedError(f"No adapter implemented for: {modelId}")
    return model

def clone_terminal_nodes(target, candidates) -> bool:
    def bfs_clone_terminal_nodes(t_root, c_root) -> bool:
        queue = deque([(t_root, c_root)])
        
        while queue:
            t_node, c_node = queue.popleft()
            # Get terminal names from metadata (default to empty list if not found)
            c_node_terminals = c_node.metadata.get('terminal_applied', [])
            t_node_terminals = t_node.metadata.get('terminal_applied', [])
            print(c_node.id, c_node_terminals, t_node_terminals)
            # Convert to sets for intersection
            missing_terminals = set(t_node_terminals) - set(c_node_terminals)

            # Get terminal children of t_node that match the common terminal names
            terminal_children = [
                child for child in t_node.children
                if isinstance(child, TerminalNode) and child.metadata.get('terminal_name') in missing_terminals
            ]
            # Traverse non-terminal children
            t_non_term_children = [
                child for child in t_node.children
                if not isinstance(child, TerminalNode)
            ]
            c_non_term_children = [
                child for child in c_node.children
                if not isinstance(child, TerminalNode)
            ]
            # update children and self
            c_node.children.extend(terminal_children)
            c_node.metadata.setdefault('terminal_applied', []).extend(missing_terminals)

            for t_child, c_child in zip(t_non_term_children, c_non_term_children):
                queue.append((t_child, c_child))

        return True

    
    for t_tree, c_tree in zip(repeat(target, len(candidates)), candidates):
        t_root = t_tree.root
        c_root = c_tree.root
        print('=====================================')
        if not bfs_clone_terminal_nodes(t_root, c_root):
            return False
    return True
  
def process_dialect():
    perturbers = get_term_perturb(terminal_type)
    embedder = RobertaEmbedder()
    tree_ids = []
    inter_dir = f'{constants.TREE_DIR}{dataset_name}_treenodes/{stategy_path}/gemma3-12b_perturb/3_2_0/tree/'
    for filename in os.listdir(inter_dir):
        if filename.endswith('.pkl'):
            tree_ids.append(int(filename[: -len('.pkl')]))
    tree_ids = tree_ids[start_idx:end_idx]
    print(tree_ids)
    
    # perturb
    print("****  start dialect perturb  ****")
    
    for tree_id in tree_ids:
        print(f"perturb tree q_id={tree_id}")
        inter_path = f"{inter_dir}{tree_id}.pkl"
        try:
            inter_tree = Tree.load_tree(embedder.model.device, inter_path, eval="terminal", embedder=embedder)
        except FileNotFoundError as e:
            print(e)
            continue
        for perturber in perturbers:
            apply_terminal_preturb(perturber, inter_tree)
        inter_tree.save_tree(inter_path)
        # replicate to checked trees
        target_tree = Tree.load_tree(embedder.model.device, inter_path, embedder=embedder, eval="terminal")
        candidate_trees = []
        candidate_paths = []
        for gen_modelId in gen_modelIds:
            final_path = f"{constants.TREE_DIR}{dataset_name}_treenodes/{stategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/{tree_id}_checked.pkl"
            try:
                cand_tree = Tree.load_tree(embedder.model.device, final_path, embedder=embedder, eval="terminal")
            except FileNotFoundError as e:
                missing_tree_path.append(final_path)
                print(e)
                continue
            candidate_trees.append(cand_tree)
            candidate_paths.append(final_path)
        
        if candidate_trees:
          clone_terminal_nodes(target_tree, candidate_trees)
        
        target_tree.save_tree(inter_path)
        for cand_tree, can_path in zip(candidate_trees, candidate_paths):
            cand_tree.save_tree(can_path)
    
    # eval
    print("****  start dialect eval  ****")
    for gen_modelId in gen_modelIds:
            with get_generator(gen_modelId) as generator:
                print(type(generator))
                device = generator.device
                # read dataset
                for tree_id in tree_ids:
                
                    final_path = f"{constants.TREE_DIR}{dataset_name}_treenodes/{stategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/{tree_id}_checked.pkl"
                    try:
                        full_tree = Tree.load_tree(device, final_path, embedder=embedder, generator=generator, eval='eval')
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    # process final tree
                    process_tree(full_tree, tree_id, gen_modelId)
                    full_tree.save_tree(final_path)
                    # write new full tree to term dir
                    print(f"tree saved to {final_path}")


def main():
    global start_idx, end_idx, terminal_type, dataset_name, stategy_path

    parser = argparse.ArgumentParser(description="Process input flags.")
    parser.add_argument('--start_idx', type=int, default=start_idx, help='Start index')
    parser.add_argument('--end_idx', type=int, default=end_idx, help='End index')
    parser.add_argument('--term_type', type=str, default=terminal_type, help='Terminal type')
    parser.add_argument('--dataset', type=str, default=dataset_name, help='Dataset name')
    parser.add_argument('--strategy_path', type=str, default=stategy_path, help='Strategy path')

    args = parser.parse_args()

    # Update globals with parsed values
    start_idx = args.start_idx
    end_idx = args.end_idx
    terminal_type = args.term_type
    dataset_name = args.dataset
    stategy_path = args.strategy_path

    print(f"[INFO] Called get_term_perturb with term_type={terminal_type}, strategy_path={stategy_path}, dataset_name={dataset_name}")
    print(f"processing trees from index {start_idx} to {end_idx}")

    process_dialect()

    print("===================== missing candidate trees =====================")
    print(missing_tree_path)

if __name__ == "__main__":
    main()

    
    