import re
from collections import defaultdict
from typing import Union, Dict, List

class BartSpektroTokenizer():
    def __init__(self, vocab_tok_to_id: Dict[int, str] = None, unk_tok=None, pad_tok=None, eos_tok=None, bos_tok=None, max_mz=499):
        # vocabulary for SMILES tokenization
        self.max_mz = max_mz # hard coded - hyperparameter to be tested (space left for mz values, !!!!MUST corespond with the data filtering!!!!)
        
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.tl_to_tok = {} # two-letter
        self.tok_to_tl = {} # two-letter
        
        self.unk_tok = unk_tok
        self.pad_tok = pad_tok
        self.eos_tok = eos_tok
        self.bos_tok = bos_tok
        self.add_special_tokens(unk_tok=unk_tok, pad_tok=pad_tok, eos_tok=eos_tok, bos_tok=bos_tok)
        
        self.first_free = 1 # the first free index (from the first 32 ASCII tokens) used to store two-letter atom names
        
    def add_token(self, token):
        if token in self.tok_to_id.keys():
            print(f"### Token {token} already present (id = {self.tok_to_id[token]})")
            return 1
        if not self.tok_to_id.values():
            new_id = self.max_mz + 1
        else:
            new_id = sorted(self.tok_to_id.values())[-1] + 1
            print(f"setting {token} as {new_id}")
        self.tok_to_id[token] = new_id
        self.id_to_tok[new_id] = token
        
    def add_tl_token(self, token):
        if token in self.tl_to_tok.keys():
            print(f"### Two-letter token {token} already present (token = {self.tl_to_tok[token]})")
            return 1
        new_tok = chr(self.first_free) # new one letter erpresentation assigned to two-letter token
        print(f"adding {token} represented as {new_tok}")
        self.tl_to_tok[token] = new_tok
        self.tok_to_tl[new_tok] = token
        self.add_token(new_tok)
        self.first_free += 1
        
    def add_tokens(self, tok_arr):
        for tok in tok_arr:
            self.add_token(tok)
            
    def add_tl_tokens(self, tok_arr):
        for tok in tok_arr:
            self.add_tl_token(tok)
    
    def add_special_tokens(self, unk_tok=None, pad_tok=None, eos_tok=None, bos_tok=None):
        unk_before = self.unk_tok # check if we changed the unk_tok for add_unk_token function
        for name, var in [("unk_tok", unk_tok), ("pad_tok", pad_tok), ("eos_tok", eos_tok), ("bos_tok", bos_tok)]:
            if getattr(self, name) is not None:
                print(f"{name} already set, you can change it manually")
                continue
            if var:
                setattr(self, name, var)
                self.add_tl_token(var)
        return unk_before != self.unk_tok # unk_tok successfully changed
    
    def add_unk_token(self, name="<unk>"):
        if not self.tok_to_id.values():
            new_id = self.max_mz + 1
        else:
            new_id = sorted(self.tok_to_id.values())[-1] + 1
        # do not change default value, if unk_tok was not changed (is already set)
        if not self.add_special_tokens(unk_tok = name):
            return
        self.tok_to_id = defaultdict(lambda: new_id, self.tok_to_id)
        self.id_to_tok = defaultdict(lambda: self.tl_to_tok[name], self.tok_to_id)
        
    def tokenize_smiles(self, smiles):
        # SMILES -> tokens
        
        # find all known two-letter atoms and replace them with their oneletter ASCII representation
        # mrkni se, jestli tam jsou neznamy two-letter v [], prirad jim ASCII reprezentaci a nahrad je ji
        
        known_pattern = re.compile("|".join(self.tl_to_tok.keys())) # known two-letter keys in pattern
        oneletter = re.sub(known_pattern, self._assign_known, smiles)
        
        unknown_pattern = re.compile("\[[A-Z][a-z]\]") # general pattern for two-letter atoms in []
        oneletter = re.sub(unknown_pattern, self._assign_unknown, oneletter)
        
        return oneletter
    
    def detokenize(self, oneletter):
        # convert oneletter SMILES string (with substitute chars) to its multiletter form (with whole 'Cl, <eos>, ...')
        pattern = re.compile("|".join(self.tok_to_tl.keys()))
        return re.sub(pattern, lambda x : self.tok_to_tl[x.group(0)], oneletter)
        
    
    def smiles_to_ids(self, smiles: Union[List[int], str]):
        return [self.tok_to_id[tok] for tok in self.tokenize_smiles(smiles)]
        
    def ids_to_smiles(self, ids):
        oneletter = "".join([self.id_to_tok[i] for i in ids])
        return self.detokenize(oneletter)

    def get_tokens(self):
        return self.tok_to_id.keys()
    
    def get_ids(self):
        return self.id_to_tok.keys()
    
    def get_vocab(self):
        return self.tok_to_id
    
    # fctions used in tokenize_smiles, don't use othetwise!!!
    def _assign_known(self, match):
        return self.tl_to_tok[match.group(0)]
        
    def _assign_unknown(self, match):
        tok = match.group(0)[1:3]
        self.add_tl_token(tok)
        return "[" + self.tl_to_tok[tok] +"]"
        
    def init_tokenizer(self):
        # otherwise all possible chars are
        brackets = ["[", "]", "(", ")"]
        bond_syms = [".", "-", "=", "#", "$", ":", "/", "\\"]
        other_syms = ["%", "@", "+"]
        nums = ["0","1","2","3","4","5","6","7","8","9"]
        upper_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # without JQ
        lower_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] # without jq
        smiles_chars = brackets + bond_syms + other_syms + nums + upper_letters + lower_letters

        self.add_unk_token()
        self.add_special_tokens(unk_tok=None, pad_tok="<pad>", eos_tok="<eos>", bos_tok="<bos>")
        self.add_tokens(smiles_chars)
        self.add_tl_tokens(["Cl", "Br"])
        
        return self