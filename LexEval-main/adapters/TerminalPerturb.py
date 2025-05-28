import torch
import gc
import logging
from dataclasses import replace
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from multivalue import Dialects
from adapters.prompt_package import PromptPackage
import nltk
from nltk.tokenize import sent_tokenize

# Make sure to download the tokenizer once (if not already done)
# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """ free CUDA & Python memory."""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:  # pragma: no cover – best‑effort cleanup
            pass
    gc.collect()
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    logging.info("after clearing cache, %s/%s memory available", free_mem, total_mem)

class TerminalPerturber(ABC):
    @abstractmethod
    def terminal_perturb(self, pkg: PromptPackage, **kwargs) -> PromptPackage: 
        """Return new datapackage after perturbation."""

    @abstractmethod
    def post_process(self) -> None:
        """Called *once* after the whole tree has been generated."""

    @abstractmethod
    def release(self) -> None:
        """Free large resources"""
            
class DialectPerturber(TerminalPerturber):
    def __init__(self, dialect: str):
        # Define perturbation functions
        self.dialect = dialect
        self.prompt_history: list[str] = []  # dedup queue
        self.name = f"{self.dialect}_dialect"
        if dialect == 'sg':
            self.model = Dialects.ColloquialSingaporeDialect()
        else:
            # TODO: omg....
            self.model = Dialects.ColloquialSingaporeDialect()
    
    def terminal_perturb(self,
                         pkg: PromptPackage,
                         max_retries: int = 5,
                        **kwargs) -> PromptPackage: 
        """Return a *new* package after semantic perturbation."""
        candidate = None
        rules = None
        is_valid = False
        retry = 0
        og_prompt = pkg.text
        while not is_valid and retry < max_retries:
            # generate paraphrase
            candidate = self.model.transform(og_prompt)
            rules = self.model.executed_rules
            clear_cache()
            # check validity
            is_valid = candidate not in self.prompt_history and not candidate == og_prompt
            
            if not is_valid:
                retry += 1
        
        # return results
        new_state = {
                    **pkg.state,
                    "terminal_name":self.name,
                    "terminal_prompt": candidate,
                    "rules": rules
                }
        new_state["is_valid"] = is_valid
        new_pkg = replace(
                pkg,
                text=candidate,
                state=new_state
            )
        return new_pkg

    def post_process(self) -> None:
        """Called *once* after the whole tree has been generated."""
        self.prompt_history.clear()

    def release(self) -> None:
        """Free model"""
        self.model = None
        clear_cache()

class PositionPerturber(TerminalPerturber):
    '''
    applies perturbation by rearraning question position
    '''
    def __init__(self, index: str):
        VALID_POS = ["prefix", "suffix", "middle"]
        if index not in VALID_POS:
            raise ValueError()
        self.position = index
        self.name = f"question_position_{self.position}"
    
    def terminal_perturb(self,
                         pkg: PromptPackage,
                        **kwargs) -> PromptPackage: 
        """Return a *new* package after semantic perturbation."""
        
        candidate = ""
        prefixed_text_prompt = pkg.text
        base_prompt = pkg.state.get('base_prompt', "")
        extra_text = pkg.state.get('prefix_text', "")
        if not extra_text:
            candidate = prefixed_text_prompt
            sent_list = []
        else:
            sent_list = sent_tokenize(extra_text)
            if  self.position == 'prefix':
                insert_pos = len(sent_list)
            elif  self.position == 'middle':
                insert_pos = len(sent_list) // 2
            else:
                insert_pos = 0
            prefix_text = " ".join(sent_list[:insert_pos])
            suffix_text = " ".join(sent_list[insert_pos:])
            candidate = f"{prefix_text} {base_prompt} {suffix_text}".strip()

        new_state = {
                **pkg.state,
                "terminal_name": self.name,
                "terminal_prompt": candidate,
                "is_valid": True,
                "position": self.position,
                "num_extra_sent": len(sent_list)
                
            }
        
        new_pkg = replace(
                pkg,
                text=candidate,
                state=new_state
            ) 
    
        return new_pkg

    def post_process(self) -> None:
        """Called *once* after the whole tree has been generated."""
        pass

    def release(self) -> None:
        pass
                