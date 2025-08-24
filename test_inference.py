#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from arbi.ai.inference_v2 import test_inference_engine
import asyncio

if __name__ == "__main__":
    asyncio.run(test_inference_engine())
