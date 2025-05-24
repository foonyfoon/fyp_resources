from typing import Tuple
from collections import deque
import textstat
import os
from tree.tree import Tree
from tree.node import Node, TerminalNode
from adapters.SemanticPerturb import PromptPackage
from adapters.TerminalPerturb import DialectPerturber
from similarity.cosine_similarity import similarity
from model.engine import LLMAdapter
from adapters.OAI_Embeddings import RobertaEmbedder
import utils.constants as constants
from utils.dataset_sampling import read_popqa_dataset

'''
each question, answer tree has new value:
- is terminal (node)
- terminal_preturb applied (list of string)
'''
type_name = "sg_dialect"
dialect = 'sg'
term_perturber = DialectPerturber(dialect)

def generate_terminal_node(
    tree,
    parent_node : Node,
    use_parent_passage=True,) -> Tuple[TerminalNode, bool]:
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
        perturbation =perturb_pkg.text
    except AssertionError as e:
        print(f"AssertionError: {e} from prompt {perturb_pkg.text}\n {perturb_pkg.state}")
        is_valid = False
        perturb_state = perturb_pkg.state
        perturbation =perturb_pkg.text       
    
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

def apply_terminal_preturb(tree: Tree):
    type_name = term_perturber.name
    root = tree.root
    level = 0
    queue = deque([(root, level)])
    
    while queue:  # BFS
        node, curr_level = queue.popleft()
        if isinstance(node, TerminalNode):
            continue  # Skip terminal nodes
        elif isinstance(node, Node) and type_name in node.metadata.get("terminal_name", []):
            continue  # Skip processed nodes
        elif isinstance(node, Node):
            # Generate a terminal node based on the current node
            term, is_valid = generate_terminal_node(tree, node, use_parent_passage=True)
            if is_valid:
                # Update node's metadata to include the type_name
                type_names = node.metadata.get("terminal_name", [])
                if not isinstance(type_names, list):
                    type_names = [type_names]
                if type_name not in type_names:
                    type_names.append(type_name)

                node.metadata["terminal_name"] = type_names

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
  
def main():
  start_idx = 300
  end_idx = 400
  embedder = RobertaEmbedder()
  gen_modelIds = ['google/gemma-3-12b-it','mistralai/Mistral-7B-Instruct-v0.2',]
  for gen_modelId in gen_modelIds:
        with get_generator(gen_modelId) as generator:
            print(type(generator))
            device = generator.model.device
            dataset_name = "POPQA"
            statrgy_path = "para"
                    
            # read popqa dataset
            df = read_popqa_dataset(1000)
            for i, row in df.iloc[start_idx : end_idx + 1].iterrows():
      
                tree_id = df["original_index"].iloc[i]
            
                print(f"processing tree q_id={tree_id}")
                
                final_path = f"{constants.TREE_DIR}{dataset_name}_treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/google-gemma-3-1b-it/complete/terminal/{tree_id}_checked.pkl"
                terminal_tree_path = f"{constants.TREE_DIR}{dataset_name}_treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/terminal/{tree_id}_checked.pkl"
                try:
                    full_tree = Tree.load_tree(device, final_path, embedder=embedder, generator=generator, eval='eval')
                except FileNotFoundError as e:
                    print(e)
                    continue
                # # add term to final tree
                apply_terminal_preturb(full_tree)
                # # process final tree
                process_tree(full_tree, tree_id, gen_modelId)
                # write new full tree to term dir
                full_tree.save_tree(terminal_tree_path)
                print(f"tree saved to {terminal_tree_path}")
  
if __name__ == "__main__":
    main()