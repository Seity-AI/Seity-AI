import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    List,
    Sequence,
    TypedDict,
    Literal,
    Union,
    cast,
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
        # Ensure we don't try to create a negative number of reserved tokens
        remaining_reserved = max(0, self.num_reserved_special_tokens - len(special_tokens_list))
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

    def encode(
            self,
            s: str,
            *,
            bos: bool,
            eos: bool,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = "all",
            disallowed_special: Union[Literal["all"], AbstractSet[str]] = "all",
            ) -> List[int]:
        """Encode a structured Message into a list of token ids."""

        assert type(s) is str

        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        MAX_NO_WHITESPACE_CHARS = 100_000

        substr = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespace_nonwhitespace(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACE_CHARS
            )
        )

        t: List[int] = []
        for substr in substr:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )  
            )

        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)

        return t

    def _split_whitespace_nonwhitespace(self, s: str, max_non_whitespace: int):
        """Yield substrings from `s` such that no contiguous run of non-whitespace
        characters exceeds `max_non_whitespace`.

        This preserves whitespace runs and only splits extremely long words.
        """
        import re

        if not s:
            return

        token_re = re.compile(r"\S+|\s+")
        buffer = []

        def flush_buffer():
            if buffer:
                yield "".join(buffer)
                buffer.clear()

        for match in token_re.finditer(s):
            token = match.group(0)
            if token.isspace() or len(token) <= max_non_whitespace:
                buffer.append(token)
            else:
                # token is a very long non-whitespace run; flush any accumulated buffer first
                if buffer:
                    yield from flush_buffer()

                # split the long token into chunks
                for i in range(0, len(token), max_non_whitespace):
                    yield token[i : i + max_non_whitespace]

        if buffer:
            yield "".join(buffer)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids back into a text string."""
        return self.model.decode(cast(List[int], token_ids))


class MsgFormat:
    def __init__(self):
        pass
