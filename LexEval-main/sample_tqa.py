from utils.dataset_sampling import read_tqa_dataset
columns=["question", "possible_answers", "s_wiki_title", "original_index"]
df = read_tqa_dataset(1000, columns=columns)
print(df.head)