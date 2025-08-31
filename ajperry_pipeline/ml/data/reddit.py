import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
from zlib import crc32
from datetime import datetime

def string_to_float_hash(s, encoding="utf-8"):
    """
    Generates a float hash between 0 and 1 from a string.
    """
    byte_string = s.encode(encoding)
    hash_value = crc32(byte_string) & 0xFFFFFFFF
    normalized_hash = float(hash_value) / (2**32)
    return normalized_hash


class RedditDataset(Dataset):
    """
    A dataset which serves a folder of reddit post information

    Attributes:
        w_queries (Parameter): Query weights
        w_keys (Parameter): Key weights
        w_values (Parameter): Value weights
        w_agg (Parameter): Aggregation weights
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int,
        is_train: bool,
        train_split_perc: float = 0.8,
        is_date: bool = False,
        date_split: datetime = datetime.now()
    ):
        def selected(data_id):
            if is_date:
                hash_val = datetime.strptime(data_id, "%Y-%m-%d %H:%M:%S")
                return (
                    is_train
                    and hash_val <= date_split
                    or not is_train
                    and hash_val > date_split
                )
            else:
                hash_val = string_to_float_hash(data_id)
                return (
                    is_train
                    and hash_val <= train_split_perc
                    or not is_train
                    and hash_val > train_split_perc
                )

        def select_train(data_id):
            if is_date:
                hash_val = datetime.strptime(data_id, "%Y-%m-%d %H:%M:%S")
                return hash_val <= date_split
            else:
                hash_val = string_to_float_hash(data_id)
                return hash_val <= train_split_perc

        self.all_data = pd.read_csv(data_path)
        # collect train
        if is_date:
            self.all_data["selected"] = self.all_data["date_added"].apply(select_train)
        else:
            self.all_data["selected"] = self.all_data["id"].apply(select_train)

        # build tokenizers from all training data
        input_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        input_tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        input_tokenizer.train_from_iterator(
            self.all_data[self.all_data["selected"]]["title"], trainer=trainer
        )
        self.input_tokenizer = input_tokenizer

        output_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        output_tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        output_tokenizer.train_from_iterator(
            self.all_data[self.all_data["selected"]]["top_comment"], trainer=trainer
        )
        self.output_tokenizer = output_tokenizer

        # Switch selection based on to train/test
        # collect train
        if is_date:
            self.all_data["selected"] = self.all_data["date_added"].apply(selected)
        else:
            self.all_data["selected"] = self.all_data["id"].apply(selected)
        self.all_data = self.all_data[self.all_data["selected"]]

        # convenience variables
        self.sos_token = torch.tensor(
            [input_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [input_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [input_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

        self.sequence_length = sequence_length
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.all_data)

    @staticmethod
    def make_causal_mask(size):
        causal_mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return causal_mask == 0

    def __getitem__(self, idx):
        row = self.all_data.iloc[idx]
        source_text = row["title"]
        target_text = row["top_comment"]

        enc_input_tokens = self.input_tokenizer.encode(source_text).ids
        dec_input_tokens = self.output_tokenizer.encode(target_text).ids
        # Truncation
        if len(enc_input_tokens) >= self.sequence_length - 2:
            enc_input_tokens = enc_input_tokens[: self.sequence_length - 2]
        if len(dec_input_tokens) >= self.sequence_length - 1:
            dec_input_tokens = dec_input_tokens[: self.sequence_length - 1]

        enc_num_padding_tokens = self.sequence_length - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.sequence_length - len(dec_input_tokens) - 1

        # Add sos and eos to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & RedditDataset.make_causal_mask(decoder_input.size(0)),
            "label": label,
            "input_text": source_text,
            "output_text": target_text,
        }
