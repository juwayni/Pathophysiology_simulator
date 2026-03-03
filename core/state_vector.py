import numpy as np
from typing import List, Dict, Optional
from models.schema import ModelSchema, VariableType

class StateVector:
    def __init__(self, model: ModelSchema):
        self.model = model
        self.state_vars = [v for v in model.variables if v.type == VariableType.STATE]
        self.n_states = len(self.state_vars)
        self.name_to_idx = {v.name: i for i, v in enumerate(self.state_vars)}
        self.idx_to_name = {i: v.name for i, v in enumerate(self.state_vars)}
        self.vector = np.array([v.initial_value for v in self.state_vars], dtype=np.float64)

    def get_value(self, name: str) -> float:
        return self.vector[self.name_to_idx[name]]

    def set_value(self, name: str, value: float):
        self.vector[self.name_to_idx[name]] = value

    def update_from_vector(self, vector: np.ndarray):
        if vector.shape != self.vector.shape:
            raise ValueError(f"Vector shape mismatch: {vector.shape} != {self.vector.shape}")
        self.vector[:] = vector

    def get_vector(self) -> np.ndarray:
        return self.vector.copy()

    def __getitem__(self, name: str) -> float:
        return self.get_value(name)

    def __setitem__(self, name: str, value: float):
        self.set_value(name, value)
