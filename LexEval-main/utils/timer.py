import os
import time
import pickle
from typing import Dict, List, Tuple


class Timers:
    _timings: List[Tuple[str, str, str, float]] = []  # (q_id, model_id, step_name, seconds)
    _start_times: Dict[Tuple[str, str, str], float] = {}
    _path: str = "timers.pkl"

    @classmethod
    def start(cls, q_id: str, model_id: str, step_name: str):
        key = (q_id, model_id, step_name)
        cls._start_times[key] = time.perf_counter()

    @classmethod
    def end(cls, q_id: str, model_id: str, step_name: str):
        key = (q_id, model_id, step_name)
        if key not in cls._start_times:
            raise ValueError(f"Step '{step_name}' for ({q_id}, {model_id}) was not started.")
        elapsed = time.perf_counter() - cls._start_times.pop(key)
        cls._timings.append((q_id, model_id, step_name, elapsed))

    @classmethod
    def report(cls) -> List[Tuple[str, str, str, float]]:
        return cls._timings.copy()

    @classmethod
    def print_report(cls):
        print("\nTiming Report:")
        for q_id, model_id, step_name, seconds in cls._timings:
            print(f"  Q: {q_id}, Model: {model_id}, Step: {step_name} -> {seconds:.4f}s")
    
    @classmethod
    def save(cls, path: str = None):
        if path is None:
            path = cls._path
        with open(path, "wb") as f:
            pickle.dump(cls._timings, f)
        print(f"Timers saved to {path}")

    @classmethod
    def load(cls, path: str = None):
        if path is None:
            path = cls._path
        if os.path.exists(path):
            with open(path, "rb") as f:
                cls._timings = pickle.load(f)
            print(f"Timers loaded from {path}")
        else:
            print(f"No timing file found at {path}, starting fresh.")
            
    @classmethod
    def reset(cls):
        """Clear all timing data and in-progress steps."""
        cls._timings.clear()
        cls._start_times.clear()
        print("Timers reset.")