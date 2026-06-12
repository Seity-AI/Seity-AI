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

    """Tokenizer wrapper for a local BPE model with custom special tokens."""

    # We borrow the CL100K base regex from tiktoken for now.
    # This is an internal pattern used by the tokenizer and may change in future tiktoken versions.
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    pat_str = getattr(cl100k_base, "_pat_str")

    def __init__(self, model_path: str):
        """Initialize tokenizer and register reserved special tokens."""
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # Explicit custom special tokens used by this project.
        # BUGFIX: add missing comma after "<PAD>" to avoid concatenating two tokens.
        special_tokens_list = [
            "<BEGIN_TEXT>",
            "<END_TEXT>",
            "<PAD>",
            "<REVERSED_SPECIAL_TOKEN_0>",
            "<REVERSED_SPECIAL_TOKEN_1>",
            "<REVERSED_SPECIAL_TOKEN_2>",
            "<REVERSED_SPECIAL_TOKEN_3>",
            "<START_HEADER>",
            "<END_HEADER>",
            "<REVERSED_SPECIAL_TOKEN_4>",
            "<EOT>",
        ]

        # Extend with generated reserved tokens to reach the expected token count.
        # BUGFIX: original range logic could produce one extra token and mismatch num_reserved_special_tokens.
        remaining_reserved = self.num_reserved_special_tokens - len(special_tokens_list)
        special_tokens_list += [
            f"<REVERSED_SPECIAL_TOKEN_{i}>"
            for i in range(5, 5 + remaining_reserved)
        ]

        # Map token text to ids starting after the base vocabulary.
        self.special_tokens = {
            token: num_base_tokens + i
            for i, token in enumerate(special_tokens_list)
        }

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # Expose frequently used ids for the tokenizer's control tokens.
        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<BEGIN_TEXT>"]
        self.eos_id: int = self.special_tokens["<END_TEXT>"]
        self.pad_id: int = self.special_tokens.get("<PAD>", self.eos_id)

        # stop_token uses the same registered special token ids.
        self.stop_token = [
            self.special_tokens["<END_TEXT>"],
            self.special_tokens["<EOT>"],
        ]

        logger.info(
            f"WORDS: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, message: Message):
        """Encode a structured Message into a list of token ids."""
        return self.model.encode(message["message"])

    def decode(self, token_ids):
        """Decode token ids back into a text string."""
        return self.model.decode(token_ids)


class MsgFormat:
    def __init__(self):
        pass
