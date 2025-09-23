import gc
import math
import os
import pickle
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List

import numpy as np
import spacy
import torch

from adapters.OAI_Embeddings import EmbedAdapter
from adapters.WikiKnowledgeGraph import KnowledgeGraph
from adapters.coref_resolve import CorefResolution
from model.engine import LLMAdapter
from similarity.cosine_similarity import similarity, similarities
import utils.constants as constants
from adapters.prompt_package import PromptPackage
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

# ---------------------------------------------------------------------------
# Abstract base perturber
# ---------------------------------------------------------------------------

class SemanticPerturber(ABC):
    """Base class – every perturber works on a :class:`PromptPackage`."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @abstractmethod
    def setup_perturber(self, pkg: PromptPackage) -> None:  
        """Prepare any expensive index/graph before perturbations start."""

    @abstractmethod
    def sem_perturb(self, pkg: PromptPackage, **kwargs) -> PromptPackage: 
        """Return a *new* package after semantic perturbation."""

    @abstractmethod
    def post_process(self) -> None:
        """Called *once* after the whole tree has been generated."""

    @abstractmethod
    def release(self) -> None:
        """Free large buffers/graphs so the GPU can breathe."""

# ---------------------------------------------------------------------------
# Composite perturber
# ---------------------------------------------------------------------------

class CombinedPerturber(SemanticPerturber):
    def __init__(self, perturbations: List[SemanticPerturber]):
        self.perturbers = perturbations

    def setup_perturber(self, prompt: str, root_entity: str): 
        for p in self.perturbers:
            p.setup_perturber(prompt, root_entity)

    def sem_perturb(self, pkg: PromptPackage, **kwargs) -> PromptPackage:
        # temporary logic accounting for para->prefix combo 
        for p in self.perturbers:           
            pkg = p.sem_perturb(pkg, **kwargs)
        prefix = pkg.state.get("prefix_text", "")  
        base = pkg.state["base_prompt"]
        new_text = f"{prefix} {base}".strip()
        # return new PromptPackage with updated .text
        return replace(pkg, text=new_text)

    def post_process(self):
        for p in self.perturbers:
            p.post_process()

    def release(self):
        for p in self.perturbers:
            p.release()
        clear_cache()

# ---------------------------------------------------------------------------
# Paraphrase perturber
# ---------------------------------------------------------------------------

class ParaphrasePerturber(SemanticPerturber):
    def __init__(self, model: LLMAdapter, embedder: EmbedAdapter):
        self.model = model
        self.embedder = embedder
        self.prompt_history: list[str] = []  # dedup queue


    def setup_perturber(self, prompt: str, root_entity: str): 
        # treat the initial root prompt as already used
        if prompt not in self.prompt_history:
            self.prompt_history.append(prompt)

    def sem_perturb(
        self,
        pkg: PromptPackage,
        *,
        max_retries: int = 5,
        min_temp: float = 0.7,
        max_temp: float = 1.5,
        upper_thresh: float = 0.995,
        lower_thresh: float = 0.849,
        **kwargs,
    ) -> PromptPackage:
        """
        Paraphrase until cosine similarity is within the desired band.

        Constructs a new PromptPackage by modifying the base prompt with a semantically 
        perturbed version that remains similar to the root prompt.

        The state dictionary is updated with:
        - "base_prompt": The selected paraphrased candidate.
        - "similarity": Cosine similarity between the root and paraphrased prompt.
        - "is_valid": True if the candidate is sufficiently different and not a duplicate.

        Returns:
            PromptPackage: A new instance containing the perturbed prompt and updated state.
        """
        retry = 0
        is_valid = False
        root_prompt = pkg.state.get("root_prompt", pkg.text)

        def get_paraphrase(prompt: str, temp: float) -> str:
            forbid_text  = "\n".join(f"{i}. {p}" for i, p in enumerate(self.prompt_history, 1))
            instruction = (
                "Generate a variation of the following user prompt "
                "by perturbing its semantics while preserving its core intent. "
                "Return just the string of the perturbed prompt."
            )
            if forbid_text :
                instruction += (
                    "\nHowever, there are several sentence variations generated by others "
                    "that you MUST NOT repeat since we wna tnew variations Sentence sare "
                    f"in this list:\n {forbid_text}"
                )

            chat = self.model.format_prompt(
                prompt,
                state=[{"role": "system", "content": instruction}],
            )
            new_p, _ = self.model.complete(chat, temperature=temp)
            return new_p.strip()
        
        candidate= ""
        
        while not is_valid and retry < max_retries:
            temperature = min(max_temp, min_temp + 1.5 * (retry / max_retries))
            logging.info("Paraphrase retry %d, temp=%.2f", retry, temperature)
            base_prompt = pkg.state["base_prompt"]
            candidate = get_paraphrase(base_prompt, temperature)
            sim = similarity(
                self.embedder.encode(root_prompt),
                self.embedder.encode(candidate),
            )
            
            is_unique = candidate not in self.prompt_history
            within_bounds = lower_thresh <= sim <= upper_thresh
            is_valid = is_unique and within_bounds

            # Construct new state and package with the accepted paraphrased prompt
            if is_valid:
                new_state = {
                    **pkg.state,
                    "base_prompt": candidate,
                    "similarity": sim,
                    "is_valid": True,
                }
                
                new_pkg = replace(
                    pkg,
                    text=candidate,
                    state=new_state
                )
            else:
                retry += 1
                if not is_unique:
                    failure_reason = "duplicate candidate"
                elif sim < lower_thresh:
                    failure_reason = f"similarity {sim:.3f} below lower threshold {lower_thresh}"
                elif sim > upper_thresh:
                    failure_reason = f"similarity {sim:.3f} above upper threshold {upper_thresh}"
                else:
                    failure_reason = "unknown"
                    
            # add to ban list if rejected or accepted
            if candidate and candidate not in self.prompt_history:
                self.prompt_history.append(candidate)
                
        if not is_valid:
            logging.info(
                "generate_semantic_node: could not generate a unique perturbation. Reason: %s",
                failure_reason,
            )
            # return last candidate as invalid perturbed text
            new_state = {
                    **pkg.state,
                    "base_prompt": candidate,
                    "similarity": sim,
                    "is_valid": False,
                }
            # Construct and return a new PromptPackage (no in-place mutation)  
            new_pkg = replace(
                pkg,
                text=candidate,
                state=new_state
            )
        return new_pkg

    def post_process(self):
        self.prompt_history.clear()

    def release(self):
        if self.model is not None:
            self.model.cleanup()
            self.model = None
        clear_cache()

# ---------------------------------------------------------------------------
# Knowledge‑prefix perturber
# ---------------------------------------------------------------------------

class PrefixPerturber(SemanticPerturber):

    def __init__(self, embedder: EmbedAdapter, tree_size: tuple[int, int]):
        self.embedder = embedder
        self.k_hop = tree_size[0] 
        self.top_k = tree_size[1]
        self.NER = spacy.load("en_core_web_trf")

        # will be initialised in `setup_setup_perturber`
        self.knowledge_graph: KnowledgeGraph | None = None
        self.root_embeddings: list[np.ndarray] | None = None
        self.cr = CorefResolution()

    def setup_perturber(self, prompt: str, root_entity: str): 
        # create new knowledge graph instance
        self.knowledge_graph = KnowledgeGraph(
            root_entity, #wiki title
            self.cr,
            k_hop=self.k_hop,
            top_k=self.top_k,
            prompt=prompt,
            embedder=self.embedder,
        )
        root_passages = [c.content for c in self.knowledge_graph.graph.children]
        self.root_embeddings = [self.embedder.encode(p) for p in root_passages]


    def sem_perturb(
        self,
        pkg: PromptPackage,
        *,
        K: int = 15,
        T_init: float = 1.0,
        T_decay: float = 0.8,
        temp: bool = False,
        **kwargs,
    ) -> PromptPackage:
        """
        Performs semantic perturbation by prepending a relevant knowledge-based prefix.

        Uses knowledge graph documents to select the best context prefix via simulated 
        annealing and cosine similarity scoring.

        Constructs a new state dictionary with:
        - "base_prompt": Retains the original prompt unless overwritten.
        - "prefix_similarity": Negative cosine similarity to prioritize semantic distance.
        - "prefix_text": Appended prefix text from the selected knowledge graph node.
        - "is_valid": Inherited from the previous state (default: True).

        Returns:
            PromptPackage: A new instance containing the prefix-augmented prompt and updated state.
        """
        if not pkg.text:
            return pkg

        def similarity_score(prompt: str) -> float:
            emb = self.embedder.encode(prompt)
            sims = [similarity(r, emb) for r in self.root_embeddings]
            return np.mean(sims)
    
        def similarity_scores(prompts: List[str]) -> List[float]:
            # TODO: kinda sus?? fix this
            print(f"len(prompts): {len(prompts)}")
            embs = self.embedder.encode(prompts).to("cuda")  # move embeddings to GPU
            root_emb_tensor = torch.stack([t.to("cuda") for t in self.root_embeddings]).float()
            print(f"root_emb_tensor.shape: {root_emb_tensor.shape}")
            print(f"embs.shape: {embs.shape}")
            sims = similarities(embs, root_emb_tensor) # can be shape (m, n, 1) / or (n, 1)
            print("scores: sims.shape", sims.shape)
            if sims.dim() == 2:  # e.g., shape (n, 1)
                ret = sims.squeeze(-1).tolist()  # flatten if needed

            else:  # shape (m, n, 1)
                sims = sims.squeeze(-1)  # shape (m, n)
                ret = (sims.mean(dim=0)).tolist()  # mean over root embeddings
                
            print(f"ret: {type(ret)} {ret}")
            return ret
        

        node_visited: set[str] = set()
        base_prompt = pkg.text
        current_prefix = ""
        current_prompt = f"{base_prompt}".strip()
        current_score = similarity_score(current_prompt)
        best_prefix = current_prefix
        best_score = current_score
        best_idx = -1
        last_candidate, last_idx = None, None
        T = T_init
    
        # 1. hoist get_next_document() in seperate for loop (node_visited track visited)
        prompt_list = []
        # collect prompts
        max_attempts = 100
        attempts = 0
        while len(prompt_list) < K and attempts < max_attempts:
            attempts += 1
            ctx, title, content, idx = self.knowledge_graph.get_next_document()
            candidate_prefix = ctx.strip()
            if candidate_prefix in node_visited:
                continue
            else:
                node_visited.add(candidate_prefix)
                new_prompt = f"{candidate_prefix} {base_prompt}"
                prompt_list.append({
                    "new_prompt": new_prompt,
                    "candidate_prefix": candidate_prefix,
                    "idx": idx
                })
        if len(prompt_list) < K:
            # optionally warn or handle the fact we stopped early
            logging.warning(
                f"Stopped after {attempts} attempts; only collected {len(prompt_list)} of {K} prompts."
            )
        
        print("prompt_list: ", prompt_list)
        
        # 2. calculate scores
        sim_scores = similarity_scores([p["new_prompt"] for p in prompt_list])
        
        print(sim_scores)
        # 3. find best prefix
        for prompt_list_idx, prompt in enumerate(prompt_list):
            candidate_prefix = prompt["candidate_prefix"]
            new_prompt = prompt["new_prompt"]
            new_score = sim_scores[prompt_list_idx]
            doc_list_idx = prompt["idx"]

            delta = current_score - new_score # always accept where new score < current score (less similar)
            p_accept = min(1, math.exp(delta / T))
            if (not temp and new_score < current_score) or (temp and random.random() <= p_accept):
                current_prefix, current_prompt, current_score = candidate_prefix , new_prompt, new_score
                best_prefix, best_score, best_idx = current_prefix, current_score, doc_list_idx
            T *= T_decay

        if not best_prefix.strip() and last_candidate:
            best_prefix, best_idx = last_candidate, last_idx

        # fallback
        if best_idx == -1 and prompt_list:
            random_prompt = random.choice(prompt_list)
            best_prefix, best_idx = random_prompt["candidate_prefix"], random_prompt["idx"]
        elif best_idx == -1:
            _, _, _, best_idx = self.knowledge_graph.get_next_document()  

        self.knowledge_graph.update_visit_status(best_idx)
        
        # Build new state and construct updated PromptPackage with best prefix
        
        prefix_text = f"{pkg.state.get('prefix_text', '')} {best_prefix}".strip()
        
        new_state = {
            **pkg.state,
            "base_prompt": pkg.state.get("base_prompt", base_prompt),
            "prefix_similarity": best_score,
            "is_valid": pkg.state.get("is_valid", True),
            "prefix_text": prefix_text,
        }

        # Construct and return a new PromptPackage (no in-place mutation)  
        new_pkg = replace(
            pkg,
            text=f"{best_prefix} {base_prompt}".strip(),
            state=new_state
        )
        return new_pkg


    def post_process(self):
        if not self.knowledge_graph:
            return
        # nothing else to keep
        del self.knowledge_graph, self.root_embeddings
        self.knowledge_graph, self.root_embeddings = None, None

    def release(self):
        if self.knowledge_graph is not None:
            del self.knowledge_graph
            self.knowledge_graph = None
        if self.root_embeddings is not None:
            del self.root_embeddings
            self.root_embeddings = None
        if self.cr is not None:
            self.cr.cleanup()
            del self.cr
        clear_cache()
