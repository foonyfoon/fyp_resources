import os
from tree.tree import Tree
from model.engine import LLMAdapter
from adapters.OAI_Embeddings import RobertaEmbedder
import utils.constants as constants
import argparse

dataset_name = "TQA"
strategy_path = "para-prefix"
start_idx = 0
end_idx = 500

gen_modelIds = [
    "google/gemma-3-1b-it",
]

VALID_TERMS = {
    'question_position_middle;sg_dialect',
}

missing_tree_path = []

def process_tree(tree: Tree, tree_idx, model_name):
    tree.run_check_pop_qa_batched(tree_idx, model_name, valid_terms=VALID_TERMS)


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


def eval_trees():


    embedder = RobertaEmbedder()

    for gen_modelId in gen_modelIds:
        tree_ids = []
        inter_dir = f"{constants.TREE_DIR}{dataset_name}_treenodes/{strategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/"
        for filename in os.listdir(inter_dir):
            if filename.endswith("_checked.pkl"):
                tree_ids.append(int(filename[: -len("_checked.pkl")]))
        tree_ids.sort()  # Sort the list in ascending order
        tree_ids = tree_ids[start_idx:end_idx]
        with get_generator(gen_modelId) as generator:
            print(type(generator))
            device = generator.device
            # read dataset
            for tree_id in tree_ids:
                print(f"gen_modelId={gen_modelId},  tree_id={tree_id}")
                final_path = f"{constants.TREE_DIR}{dataset_name}_treenodes/{strategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/{tree_id}_checked.pkl"
                try:
                    full_tree = Tree.load_tree(device, final_path, embedder=embedder, generator=generator, eval='eval')
                except FileNotFoundError as e:
                    print(e)
                    continue
                # process final tree
                process_tree(full_tree, tree_id, gen_modelId)
                try:
                    full_tree.save_tree(final_path)
                except Exception as e:
                    missing_tree_path.append(final_path)
                    print(e)
                    continue
                # write new full tree to term dir
                print(f"tree saved to {final_path}")


def main():
    global start_idx, end_idx, dataset_name, strategy_path, long, eval_only, perturb_only

    parser = argparse.ArgumentParser(description="Process input flags.")
    parser.add_argument('--start_idx', type=int, default=start_idx, help='Start index')
    parser.add_argument('--end_idx', type=int, default=end_idx, help='End index') 
    parser.add_argument('--dataset', type=str, default=dataset_name, help='Dataset name')
    parser.add_argument('--strategy_path', type=str, default=strategy_path, help='Strategy path')
    args = parser.parse_args()

    # Update globals with parsed values
    start_idx = args.start_idx
    end_idx = args.end_idx 
    dataset_name = args.dataset
    strategy_path = args.strategy_path

    print(f"[INFO] Called get_term_perturb with strategy_path={strategy_path}, dataset_name={dataset_name}")
    print(f"processing trees from index {start_idx} to {end_idx}")

    eval_trees()

    print("===================== missing candidate trees =====================")
    print(missing_tree_path)

if __name__ == "__main__":
    main()