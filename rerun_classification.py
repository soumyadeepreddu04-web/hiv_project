"""Rerun the modular classification pipeline using cached local artefacts."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys


def load_pipeline_module():
    file_path = pathlib.Path(__file__).resolve().parent / "drug_interaction_ml"
    loader = importlib.machinery.SourceFileLoader("drug_interaction_ml", str(file_path))
    spec = importlib.util.spec_from_loader("drug_interaction_ml", loader, origin=str(file_path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    sys.modules["drug_interaction_ml"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    load_pipeline_module().main()
