from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import enum

class UnitSystem(str, enum.Enum):
    SI = "SI"
    CGS = "CGS"
    PHYSIO = "PHYSIO"

class VariableType(str, enum.Enum):
    STATE = "state"
    ALGEBRAIC = "algebraic"
    PARAMETER = "parameter"
    CONSTANT = "constant"

class VariableSchema(BaseModel):
    name: str
    unit: str
    description: Optional[str] = None
    initial_value: float
    type: VariableType = VariableType.STATE
    min_val: Optional[float] = None
    max_val: Optional[float] = None

class ParameterSchema(BaseModel):
    name: str
    unit: str
    value: float
    description: Optional[str] = None

class EquationSchema(BaseModel):
    target: str  # Variable name (for state variables, this is d(target)/dt)
    expression: str
    description: Optional[str] = None

class EventSchema(BaseModel):
    name: str
    condition: str  # Expression that triggers event when true
    action: str     # Expression for state/parameter modification
    description: Optional[str] = None

class ModelMetadata(BaseModel):
    name: str
    version: str
    author: Optional[str] = None
    description: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    references: List[str] = []

class ModelSchema(BaseModel):
    metadata: ModelMetadata
    variables: List[VariableSchema]
    parameters: List[ParameterSchema]
    equations: List[EquationSchema]
    events: List[EventSchema] = []
    unit_system: UnitSystem = UnitSystem.SI
    functional_primitives: List[str] = ["sin", "cos", "exp", "log", "pow", "sqrt", "abs"]

    @validator('variables')
    def unique_variable_names(cls, v):
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return v

    @validator('parameters')
    def unique_parameter_names(cls, v):
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")
        return v
