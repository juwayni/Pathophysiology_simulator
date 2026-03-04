from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import enum

class UnitSystem(str, enum.Enum):
    SI = "SI"
    PHYSIO = "PHYSIO"

class VariableType(str, enum.Enum):
    STATE = "state"
    ALGEBRAIC = "algebraic"

class VariableSchema(BaseModel):
    name: str
    unit: str
    initial_value: float
    type: VariableType = VariableType.STATE
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: Optional[str] = None

class ParameterSchema(BaseModel):
    name: str
    unit: str
    value: float
    description: Optional[str] = None

class EquationSchema(BaseModel):
    target: str
    expression: str
    description: Optional[str] = None

class EventSchema(BaseModel):
    name: str
    condition: str
    action: str
    description: Optional[str] = None

class ModelMetadata(BaseModel):
    name: str
    version: str
    confidence_score: float = 1.0
    references: List[str] = []

class ModelSchema(BaseModel):
    metadata: ModelMetadata
    variables: List[VariableSchema]
    parameters: List[ParameterSchema]
    equations: List[EquationSchema]
    events: List[EventSchema] = []
    unit_system: UnitSystem = UnitSystem.PHYSIO

    @validator('variables')
    def unique_vars(cls, v):
        if len([x.name for x in v]) != len(set([x.name for x in v])):
            raise ValueError("Duplicate variables")
        return v
