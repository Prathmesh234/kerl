"""
Simple utilities for extracting input info from PyTorch code.
"""

import tempfile
import importlib.util
import sys
import os


def extract_inputs(pytorch_code: str):
    """
    Execute get_inputs() from PyTorch code and return the tensor list.
    """
    import torch

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(pytorch_code)
        temp_file = f.name

    try:
        spec = importlib.util.spec_from_file_location("temp_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module"] = module
        spec.loader.exec_module(module)

        if hasattr(module, "get_inputs"):
            inputs = module.get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            return inputs
        else:
            raise ValueError("PyTorch code must define 'get_inputs' function")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
        if "temp_module" in sys.modules:
            del sys.modules["temp_module"]


def get_shapes(pytorch_code: str):
    """
    Get shapes from get_inputs().
    """
    inputs = extract_inputs(pytorch_code)
    return [tuple(t.shape) for t in inputs if hasattr(t, 'shape')]
