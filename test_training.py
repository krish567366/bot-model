#!/usr/bin/env python3
"""
Test the training pipeline functionality
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbi.ai.training_v2 import test_training_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing training pipeline...")
    model_id = test_training_pipeline()
    print(f"Success! Model ID: {model_id}")
