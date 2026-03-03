import numpy as np
from typing import List, Dict, Optional
from models.schema import ModelSchema

class Parameters:
    def __init__(self, model: ModelSchema):
        self.model = model
        self.params = [p for p in model.parameters]
        self.n_params = len(self.params)
        self.name_to_idx = {p.name: i for i, p in enumerate(self.params)}
        self.vector = np.array([p.value for p in self.params], dtype=np.float64)

    def get_value(self, name: str) -> float:
        return self.vector[self.name_to_idx[name]]

    def set_value(self, name: str, value: float):
        self.vector[self.name_to_idx[name]] = value

    def get_vector(self) -> np.ndarray:
        return self.vector.copy()

    def update_from_vector(self, vector: np.ndarray):
        if vector.shape != self.vector.shape:
            raise ValueError(f"Vector shape mismatch")
        self.vector[:] = vector
