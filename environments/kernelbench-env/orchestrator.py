"""
Orchestrator - entry point for the KernelBench execution environment.

Validates PyTorch + Triton code inputs, then dispatches to Modal H100
for benchmarking via modal_app.
"""

import re
import modal
from modal_app import app as modal_app, benchmark_kernelbench, benchmark_batch
from typing import Optional


def _validate_pytorch_code(pytorch_code: str) -> list[str]:
    """Validate pytorch_code has the required nn.Module pattern. Returns list of errors."""
    errors = []
    if "class " not in pytorch_code or "nn.Module" not in pytorch_code:
        errors.append("pytorch_code must define a class inheriting from nn.Module")
    if "def get_inputs" not in pytorch_code:
        errors.append("pytorch_code must define get_inputs()")
    if "def forward" not in pytorch_code:
        errors.append("pytorch_code must define forward() in the nn.Module class")
    return errors


def _validate_triton_code(triton_code: str) -> list[str]:
    """Validate triton_code has the required wrapper. Returns list of errors."""
    errors = []
    if "def triton_kernel_wrapper" not in triton_code:
        errors.append("triton_code must define triton_kernel_wrapper()")
    if "@triton.jit" not in triton_code:
        errors.append("triton_code must contain at least one @triton.jit kernel")
    return errors


class KernelBenchOrchestrator:
    """
    Entry point for the KernelBench environment.

    Validates inputs and dispatches to Modal H100 for execution.

    Usage:
        with KernelBenchOrchestrator() as orch:
            result = orch.run(triton_code=..., pytorch_code=...)
    """

    def __init__(self):
        self._ctx = None

    def __enter__(self):
        self._ctx = modal_app.run()
        self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx:
            self._ctx.__exit__(*args)
            self._ctx = None

    def run(
        self,
        triton_code: str,
        pytorch_code: str,
        kernel_name: Optional[str] = None,
        entry_point: str = "Model",
        n_correctness: int = 5,
        n_trials: int = 20,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> dict:
        """
        Validate inputs and benchmark a Triton kernel on Modal H100.

        Args:
            triton_code: Must define triton_kernel_wrapper().
            pytorch_code: Must define nn.Module, get_inputs(), get_init_inputs().
            kernel_name: Optional name for logging.
            entry_point: nn.Module class name (default: "Model").
            n_correctness: Number of correctness checks.
            n_trials: Number of timing trials.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            {correctness, speedup, fast_0, fast_1, fast_2,
             reference_time_ms, kernel_time_ms, error}
        """
        # Validate inputs
        errors = _validate_pytorch_code(pytorch_code) + _validate_triton_code(triton_code)
        if errors:
            return {
                "correctness": False,
                "speedup": 0.0,
                "fast_0": False,
                "fast_1": False,
                "fast_2": False,
                "reference_time_ms": None,
                "kernel_time_ms": None,
                "error": "Validation failed: " + "; ".join(errors),
            }

        try:
            return benchmark_kernelbench.remote(
                triton_code=triton_code,
                pytorch_code=pytorch_code,
                n_correctness=n_correctness,
                n_trials=n_trials,
                kernel_name=kernel_name or "unnamed_kernel",
                entry_point=entry_point,
                rtol=rtol,
                atol=atol,
            )
        except Exception as e:
            return {
                "correctness": False,
                "speedup": 0.0,
                "fast_0": False,
                "fast_1": False,
                "fast_2": False,
                "reference_time_ms": None,
                "kernel_time_ms": None,
                "error": str(e),
            }

    def run_batch(self, kernels: list[dict]) -> list[dict]:
        """
        Validate and benchmark multiple kernels in one container.

        Each dict needs: triton_code, pytorch_code.
        Optional: kernel_name, entry_point, n_correctness, n_trials.
        """
        # Validate each kernel upfront
        validated = []
        results = []
        for spec in kernels:
            errs = (_validate_pytorch_code(spec.get("pytorch_code", ""))
                    + _validate_triton_code(spec.get("triton_code", "")))
            if errs:
                results.append({
                    "kernel_name": spec.get("kernel_name", "unnamed_kernel"),
                    "correctness": False, "speedup": 0.0,
                    "fast_0": False, "fast_1": False, "fast_2": False,
                    "reference_time_ms": None, "kernel_time_ms": None,
                    "error": "Validation failed: " + "; ".join(errs),
                })
            else:
                validated.append((len(results), spec))
                results.append(None)  # placeholder

        if not validated:
            return results

        # Build list for Modal batch call
        batch_specs = [spec for _, spec in validated]
        try:
            batch_results = benchmark_batch.remote(batch_specs)
        except Exception as e:
            batch_results = [{
                "correctness": False, "speedup": 0.0,
                "fast_0": False, "fast_1": False, "fast_2": False,
                "reference_time_ms": None, "kernel_time_ms": None,
                "error": str(e),
            } for _ in batch_specs]

        # Merge back
        for (idx, _), res in zip(validated, batch_results):
            results[idx] = res

        return results
