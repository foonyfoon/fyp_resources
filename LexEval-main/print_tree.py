import sys

from tree.tree import ReadTree

checked_tree_dir="/vol/bitbucket/lst20/treenodes/base/3_2_1/complete"

def print_tree(index): 
    checked_tree_file_path = f"{checked_tree_dir}/{index}_checked.pkl"
    # Load tree in evaluation mode using the generator
    test_tree = ReadTree.load_read_tree(checked_tree_file_path)
    test_tree.print_tree()

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2]) 
for index in range(start_idx, end_idx + 1):
    print_tree(index)