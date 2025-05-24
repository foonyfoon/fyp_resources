from fastcoref import FCoref
import torch

import gc
from typing import Tuple, List, Dict
from collections import deque, defaultdict

Span = Tuple[int, int]

gc.enable()

def clear_cache():
    # Clear any leftover tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
class Chunk:
    def __init__(self, span: Span, text: str):
        self.span: Span = span
        self.text: str = text
        self.dependants: List["Chunk"] = []
        self.resolved_text: str = text
        # key: span of the chunk this one depends on
        # value: (start, end) *relative* indices inside self.text
        self.depend_on: Dict[Span, Tuple[int,int]] = {}
        self.in_degree: int = 0

    def __repr__(self):
        return f"Chunk({self.span}, '{self.text}', depend_on={self.depend_on.keys()}, in_degree={self.in_degree})"


class CorefResolution():
  def __init__(self, device='cuda:0'):
    self.model = FCoref(device=device)
  
  
  def cleanup(self):
        """
        Free up all CUDA memory held by the coref model and delete it.
        """
        clear_cache()
        # delete the model object
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        # TODO: do we need this?
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        
  def build_chunks(self, text: str, clusters: List[List[Span]]) -> Dict[Span, Chunk]:
      chunk_map: Dict[Span, Chunk] = {}

      # 1) Cluster-based dependencies
      for cluster in clusters:
          # canonical = first span in cluster
          canon_span = cluster[0]
          canon_text = text[canon_span[0]:canon_span[1]]
          canon_chunk = Chunk(canon_span, canon_text)
          chunk_map[canon_span] = canon_chunk

          for span in cluster[1:]:
              span_text = text[span[0]:span[1]]
              ch = Chunk(span, span_text)
              # non-canonical depends on canonical
              # here the entire span is replaced, so relative = full slice
              rel = (0, span[1] - span[0])
              ch.depend_on[canon_span] = rel
              ch.in_degree += 1
              # link back for topological ordering
              canon_chunk.dependants.append(ch)
              chunk_map[span] = ch

      # 2) Composite-within-span dependencies
      #    e.g. “She and Kitty” chunk should depend on “She” chunk
      for outer in list(chunk_map.values()):
          for inner in list(chunk_map.values()):
              if outer is inner:
                  continue
              # if inner.span is fully inside outer.span
              if inner.span[0] >= outer.span[0] and inner.span[1] <= outer.span[1]:
                  # skip if already a cluster-dependency
                  if inner.span in outer.depend_on:
                      continue
                  # compute relative slice inside outer.text
                  rel_start = inner.span[0] - outer.span[0]
                  rel_end   = inner.span[1] - outer.span[0]
                  outer.depend_on[inner.span] = (rel_start, rel_end)
                  outer.in_degree += 1
                  # register for graph traversal
                  inner.dependants.append(outer)

      return chunk_map
      

  def resolve_chunks(self, chunk_map: Dict[Span, Chunk]) -> Dict[Span, Chunk]:
      # 1) Initialize queue with all chunks whose in_degree == 0
      q = deque(ch for ch in chunk_map.values() if ch.in_degree == 0)
      resolved_order = []

      # 2) Process
      while q:
          chunk = q.popleft()
          # chunk.depend_on maps dep_span -> (rel_start, rel_end) within chunk.text
          for dep_span, (rs, re) in chunk.depend_on.items():
              dep = chunk_map[dep_span]
              # replace that slice in chunk.resolved_text
              before = chunk.resolved_text[:rs]
              after  = chunk.resolved_text[re:]
              chunk.resolved_text = before + dep.resolved_text + after

          resolved_order.append(chunk)

          # decrement in_degree of dependants
          for child in chunk.dependants:
              child.in_degree -= 1
              if child.in_degree == 0:
                  q.append(child)

      # 3) Return map with updated resolved_text
      return { ch.span: ch for ch in resolved_order }


  def apply_resolved_chunks_incremental(
      self,
      text: str,
      coref_replacements: List[Tuple[str, List[Span]]]
  ) -> str:
      # 1) Flatten into (start, end, replacement_text)
      flat: List[Tuple[int,int,str]] = []
      for replacement, spans in coref_replacements:
          for (s, e) in spans:
              flat.append((s, e, replacement))

      # 2) Sort by original start index
      flat.sort(key=lambda x: x[0])

      # 3) Apply in-order, tracking offset
      resolved = text
      offset = 0
      for orig_start, orig_end, replacement in flat:
          start = orig_start + offset
          end   = orig_end   + offset

          # splice in the replacement
          resolved = resolved[:start] + replacement + resolved[end:]

          # update offset by difference in length
          offset += len(replacement) - (end - start)

      return resolved


  def apply_coref_resolution(self, text: str) -> str:
      """
      High-level pipeline: build → resolve → collect replacements →
      apply incrementally → return final text.
      """
      corefres_pred = self.model.predict(
        texts=[text]
      )
      clear_cache()
      clusters = corefres_pred[0].get_clusters(as_strings=False)
      chunk_map = self.build_chunks(text, clusters)
      resolved_map = self.resolve_chunks(chunk_map)

      # collect only those chunks whose resolved_text changed
      replacements = defaultdict(list)
      for span, ch in resolved_map.items():
          if ch.resolved_text != ch.text:
              replacements[ch.resolved_text].append(span)

      coref_replacements = list(replacements.items())
      return self.apply_resolved_chunks_incremental(text, coref_replacements)
