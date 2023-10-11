import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import selfies as sf

from transformers import PreTrainedTokenizer


# allowing different constraints - motivated by NIST database
default_constraints = sf.get_semantic_constraints()
default_constraints["I"] = 5
default_constraints["Ti"] = 13
default_constraints["P"] = 6
default_constraints["P-1"] = 6
sf.set_semantic_constraints(default_constraints)


class SelfiesTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = 200, **kwargs):
        super().__init__(max_len=max_len, **kwargs)
        self.__token_ids: Dict[str, int] = vocab
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
    
    def _tokenize(self, text: str, **kwargs):
        return sf.split_selfies(text)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def _convert_token_to_id(self, token: str):
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        json.dump(self.__token_ids, open(vocab_path, 'w'))
        return str(vocab_path),

    def get_vocab_size(self, with_added_tokens=True) -> int:
        return self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.__token_ids


def hardcode_build_selfies_tokenizer() -> SelfiesTokenizer:
    vocab = sf.get_semantic_robust_alphabet()
    vocab = sorted(list(vocab))
    special_tokens = ['[eos]', '[ukn]', '[pad]', '[bos]']
    sources = ['[nist]', '[rassp]', '[neims]', '[trafo]', '[source1]', '[source2]', '[source3]']
    vocab = special_tokens + vocab + sources


    vocab_dict = {key: i for i, key in enumerate(vocab)}
        
    sel_tokenizer = SelfiesTokenizer(vocab_dict, max_len=200)

    # tell your tokenizer about your special tokens
    sel_tokenizer.add_special_tokens({
        'unk_token': '[ukn]',
        'pad_token': '[pad]',
        'bos_token': '[bos]',
        'eos_token': '[eos]',
        'additional_special_tokens': sources
    })
    return sel_tokenizer