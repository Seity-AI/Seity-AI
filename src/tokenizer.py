import os
from logging import getLogger
from pathlib import Path
from typing import (
    TypedDict,
    Literal,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

Role = Literal['system','user','assistant']

class Message(TypedDict):
    role: Role
    message: str

class Tokenizer:

    num_reserved_special_tokens = 512

    """
    We use pre-made pat string for just now. in next update we will change it to our pat string.

    pat_string = ""
    """
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    pat_str = getattr(cl100k_base, "_pat_str")

    def __init__(self, model_path: str):
        """
        Initializes the tokenizer of model
        We use tiktoken package for first updates. We will make our tokenizer package.

        Args:
          model_path : str
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens_list = [
            "<BEGIN_TEXT>",
            "<END_TEXT>",
            "<PAD>"
            "<REVERSED_SPECIAL_TOKEN_0>",
            "<REVERSED_SPECIAL_TOKEN_1>",
            "<REVERSED_SPECIAL_TOKEN_2>",
            "<REVERSED_SPECIAL_TOKEN_3>",
            "<START_HEADER>",
            "<END_HEADER>",
            "<REVERSED_SPECIAL_TOKEN_4>",
            "<EOT>"
        ] + [
            f"<|reversed_special_tokens_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens_list)
        }

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = self.special_tokens.get(
            "<|pad|>",
            self.special_tokens["<|end_of_text|>"]
        )
        self.stop_token = [
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        ]
        logger.info(f"WORDS: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, message: Message):
        pass

    def decode(self, message: Message):
        pass


class MsgFormat:
    def __init__(self):
        pass
