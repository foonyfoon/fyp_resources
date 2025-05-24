from typing import Tuple
from collections import deque
import textstat
import os
import torch
from datetime import datetime
import gc
from tree.tree import Tree
from tree.node import Node, TerminalNode
from adapters.SemanticPerturb import PromptPackage
from adapters.TerminalPerturb import TerminalPerturber, PositionPerturber
from similarity.cosine_similarity import similarity
from model.engine import LLMAdapter
from adapters.OAI_Embeddings import RobertaEmbedder

positions = ["suffix", "middle"]

models = [
    "mistralai/Mistral-7B-Instruct-v0.2",
]

gc.enable()


def clear_cache() -> None:
    """ free CUDA & Python memory."""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:  # pragma: no cover – best‑effort cleanup
            pass
    gc.collect()
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"after clearing cache, {free_mem}/{total_mem} memory available")


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
        print(f"RuntimeError: {e} from prompt {perturb_pkg.text}\n {perturb_pkg.state}")
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
                term.metadata = {**term.metadata, "level": curr_level}
            # Add children of the current node to the queue
            for child in node.children:
                queue.append((child, curr_level + 1))

            if is_valid:  # Append the terminal node as a child
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


def main(partition):
    ############ setup ############
    embedder = RobertaEmbedder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare list of inter_path files
    tree_ids = []
    inter_dir = "/vol/bitbucket/lst20/long_POPQA_treenodes/prefix/gemma3-12b_perturb/3_2_0/tree/"
    file_ext = ".pkl"
    for filename in os.listdir(inter_dir):
        if filename.endswith(file_ext):
            tree_ids.append(int(filename[: -len(file_ext)]))
    preturbers = [PositionPerturber(position) for position in positions]
    
    if partition == 0:
        tree_ids = [tree_id for tree_id in tree_ids if tree_id <= 7398]
    else:
        tree_ids = [tree_id for tree_id in tree_ids if tree_id > 7398]
    ############ setup ############

    # for tree_id in tree_ids:
    #     inter_path = f"{inter_dir}{tree_id}{file_ext}"
    #     try:
    #         inter_tree = Tree.load_tree(
    #             device, inter_path, embedder=embedder, eval="term"
    #         )
    #     except FileNotFoundError:
    #         continue
    #     # update inter-tree
    #     print(f"processing tree_id {tree_id}")
        
    #     for term_perturber in preturbers:
    #         apply_terminal_preturb(term_perturber, inter_tree)
            
    #     inter_tree.save_tree(inter_path)
        
    #     # BFS walk to replicate term nodes across tree for updating any checked trees
    #     for m in models:
    #         checked_path = f"/vol/bitbucket/lst20/long_POPQA_treenodes/prefix/gemma3-12b_perturb/3_2_0/{m.replace('/', '-')}/complete/{tree_id}_checked.pkl"
    #         try:
    #             checked_tree = Tree.load_tree(
    #                 device, checked_path, embedder=embedder, eval="term"
    #             )
    #         except FileNotFoundError:
    #             continue
    #         inter = inter_tree.root
    #         full = checked_tree.root
    #         queue = deque([(inter, full)])
    #         # bfs
    #         while queue:
    #             node_inter, node_full = queue.popleft()

    #             if len(node_inter.children) != len(node_full.children):
    #                 # add terminal node from this perturbation
    #                 for node in node_inter.children:
    #                     full_terminals = node_full.metadata.get("terminal_applied", [])
    #                     if isinstance(node, TerminalNode) and node.metadata["terminal_name"] not in full_terminals:
    #                         node_full.children.append(node)

    #             for child_inter, child_full in zip(node_inter.children, node_full.children):
    #                 queue.append((child_inter, child_full))
    #         # save tree for eval later
    #         checked_tree.save_tree(checked_path)
            
    # 2. process checked trees for terminal nodes
    for m in models:
        with get_generator(m) as generator:
            device = generator.model.device
            for tree_id in tree_ids:
                inter_path = f"{inter_dir}{tree_id}{file_ext}"
                checked_path = f"/vol/bitbucket/lst20/long_POPQA_treenodes/prefix/gemma3-12b_perturb/3_2_0/{m.replace('/', '-')}/complete/{tree_id}_checked.pkl"
                if not os.path.exists(checked_path):
                    try:
                        print(f"processing tree id {tree_id}")
                        full_tree = Tree.load_tree(device, inter_path, embedder=embedder, generator=generator, eval='eval')
                    except FileNotFoundError:
                        print(f"no tree {checked_path}")
                        continue
                    process_tree(full_tree, tree_id, m)
                    full_tree.save_tree(checked_path)
                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{timestamp}: tree saved to {checked_path}")
        
        clear_cache()

if __name__ == "__main__":
    partition = 1
    print(f"processiing partition {partition + 1} / 2")
    main(partition)
