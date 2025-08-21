from torch.utils.data import Dataset
from pathlib import Path
import json

from ajperry_pipeline.ml.utils.data_splitting import string_to_float_hash


class RedditDataset(Dataset):
    def __init__(self, data_path: str, is_train: bool, train_split_perc: float = 0.8):
        self.data_path = Path(data_path)
        self.paths = []
        for p in  self.data_path.glob("*"):
            with p.open() as f:
                data = json.load(f)
                hash_val = string_to_float_hash(data["id"])
                if is_train and hash_val <= train_split_perc:
                    self.paths.append(p)
                elif not is_train and hash_val > train_split_perc:
                    self.paths.append(p)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with self.paths[idx].open() as f:
            data = json.load(f)

        {
            id: data["id"],
            "title": title,
            "url": url,
            "text": text,
            "top_comment": comment_text,
            "comment_score": comment_score
        } 
        return data["title"], data["top_comment"]
