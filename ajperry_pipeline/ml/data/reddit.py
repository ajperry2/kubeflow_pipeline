from torch.utils.data import Dataset
import pandas as pd

from ajperry_pipeline.ml.utils.data_splitting import string_to_float_hash


class RedditDataset(Dataset):
    """
    A dataset which serves a folder of reddit post information

    Attributes:
        w_queries (Parameter): Query weights
        w_keys (Parameter): Key weights
        w_values (Parameter): Value weights
        w_agg (Parameter): Aggregation weights
    """
    
    def __init__(self, data_path: str, is_train: bool, train_split_perc: float = 0.8):
        
        def selected(data_id):
            hash_val = string_to_float_hash(data_id)
            return (
                is_train and hash_val <= train_split_perc
                or not is_train and hash_val > train_split_perc
            )

        self.all_data = pd.read_csv(data_path)
        self.all_data["selected"] = self.all_data['id'].apply(selected)
        self.all_data = self.all_data[self.all_data['selected']]
        
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        row = self.all_data.iloc[idx]
        return row["title"], row["top_comment"]
