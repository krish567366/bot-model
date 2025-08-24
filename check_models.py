#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from arbi.ai.registry import ModelRegistry

def list_models():
    """List all models in registry"""
    print("Checking model registry...")
    
    registry = ModelRegistry("model_registry.db")
    
    # List all models
    models = registry.list_models()
    print(f"Found {len(models)} models:")
    
    for model_metadata in models:
        print(f"  {model_metadata.model_id}: {model_metadata.symbol} - score: {model_metadata.validation_score:.4f}")
    
    if models:
        # Get latest model for TEST symbol
        latest = registry.get_latest_model("TEST")
        print(f"\nLatest model for TEST: {latest.model_id if latest else 'None'}")

if __name__ == "__main__":
    list_models()
