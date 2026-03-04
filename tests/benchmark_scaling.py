import time
import numpy as np
from core.engine import Engine
from models.schema import ModelSchema, VariableSchema, VariableType, ParameterSchema, EquationSchema, ModelMetadata

def benchmark_scaling():
    # Scale down to 200 for CI environment timeout limits
    N = 200
    variables = [VariableSchema(name=f"x{i}", unit="1", initial_value=1.0, type=VariableType.STATE) for i in range(N)]
    parameters = [ParameterSchema(name="k", unit="1", value=0.1)]
    equations = []
    for i in range(N):
        expr = f"-k * x{i}"
        if i > 0:
            expr += f" + 0.01 * x{i-1}"
        if i < N-1:
            expr += f" + 0.01 * x{i+1}"
        equations.append(EquationSchema(target=f"x{i}", expression=expr))

    metadata = ModelMetadata(name=f"ScaleTest_{N}", version="1.0")
    model = ModelSchema(metadata=metadata, variables=variables, parameters=parameters, equations=equations)

    print(f"Compiling model with {N} variables...")
    start_time = time.time()
    engine = Engine(model)
    compile_time = time.time() - start_time
    print(f"Compilation took {compile_time:.2f} seconds.")

    print("Running simulation...")
    t_span = (0, 10)
    start_time = time.time()
    history = engine.run(t_span)
    sim_time = time.time() - start_time
    print(f"Simulation took {sim_time:.2f} seconds for {len(history)} steps.")

    assert len(history) > 0
    print("Scaling benchmark passed!")

if __name__ == "__main__":
    benchmark_scaling()
