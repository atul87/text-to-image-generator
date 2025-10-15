#!/usr/bin/env python3
"""
Configuration file for image generation settings.
"""

# Default settings
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_HEIGHT = None  # Use model default
DEFAULT_WIDTH = None   # Use model default
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_DIR = "outputs"

# Device settings
DEFAULT_DEVICE = "auto"  # Will be resolved at runtime

# Scheduler settings
DEFAULT_SCHEDULER = "DPMSolverMultistepScheduler"

# Safety and performance
MAX_PROMPT_LENGTH = 1000
MAX_IMAGE_DIMENSION = 1024
MIN_IMAGE_DIMENSION = 64