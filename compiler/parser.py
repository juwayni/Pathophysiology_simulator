import json
from models.schema import ModelSchema

class ModelLoader:
    @staticmethod
    def load_from_json(filepath: str) -> ModelSchema:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ModelSchema(**data)

    @staticmethod
    def load_from_dict(data: dict) -> ModelSchema:
        return ModelSchema(**data)
