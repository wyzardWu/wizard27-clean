"""Where checkpoints, data and the helper Python environments are, from environment variables."""
import os
import sys

WAN_CKPT = os.environ.get("WAN_CKPT", "checkpoints/Wan2.2-TI2V-5B")      # the official Wan2.2-TI2V-5B folder (VAE, T5, tokenizer)
DATA_ROOT = os.environ.get("WIZARD_DATA", "data")                         # the training data root (wan_v7/, wan_memroam/, ...)
PY_MOGE = os.environ.get("WIZARD_PY_MOGE", sys.executable)                # a Python with MoGe-2 installed
PY_MAPANY = os.environ.get("WIZARD_PY_MAPANY", sys.executable)            # a Python with MapAnything installed
PY_DA3 = os.environ.get("WIZARD_PY_DA3", sys.executable)                  # a Python with Depth Anything 3 installed
