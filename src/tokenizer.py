import os
import token
from logging import getLogger
from pathlib import Path
from typing import (
    List,
    TypedDict,
    AbstractSet,
    Union, Literal,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

Role = Literal['system','user','assistant']

class Message(TypedDict):
    role: Role
    message: str

class Tokenizer:

    """
    We use pre-made pat string for just now. in next update we will change it to our pat string.

    pat_string = ""
    """
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    pat_str = cl100k_base._pat_str

    def __init__(self, model_path: str):
        """
        I don't have idea for this section. I will change it :|

         model_path : Path
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens = [
            ""
        ]

        self.special_tokens = {
            token: num_base_tokens
        }

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=
        )
        pass

    def encode(self, message: Message) -> Message:
        pass

    def decode(self, message: Message) -> Message:
        pass
