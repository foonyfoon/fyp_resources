import datasets
from sklearn.model_selection import train_test_split


class PopQA:
    def __init__(self):
        self.popqa = datasets.load_dataset("akariasai/PopQA")["test"]

    def prep_header(self):
        qna = []
        for i in range(100):
            qna.append([])
            qna[i].append(self.popqa[i]["question"])
            qna[i].append(self.popqa[i]["possible_answers"])

        return qna

    def full_dataset(self):
        return self.popqa

    def get_subset(self, subset_size):
        if subset_size < 100:
            raise ValueError("Subset size must be larger than 100.")

        popqa_df = self.popqa.to_pandas()

        # Split the data to get a subset with the same distribution of classes
        a, b, c, d = train_test_split(
            popqa_df,
            popqa_df["prop"],
            test_size=subset_size,
            shuffle=True,
            random_state=42,
        )
        return b, d
