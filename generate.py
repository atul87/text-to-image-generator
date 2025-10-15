#!/usr/bin/env python3
"""
Simple CLI to generate images from text prompts using Hugging Face Diffusers (Stable Diffusion).

This script provides a command-line interface for generating images from text descriptions
using pre-trained Stable Diffusion models via the Hugging Face diffusers library.

Features:
- Support for multiple prompts from command line or file
- Configurable generation parameters
- Progress tracking with tqdm
- Error handling and logging
- Cross-platform compatibility

Notes:
- Provide HF_TOKEN in the environment if the model requires authentication.
- For faster results, run on a GPU-enabled machine with CUDA and a compatible torch.
"""

import os
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union
import torch
from PIL import Image
from tqdm import tqdm

# Import configuration
try:
    from config import (
        DEFAULT_MODEL,
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_DEVICE,
        DEFAULT_NUM_IMAGES,
    )
except ImportError:
    # Fallback defaults if config.py is not available
    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_OUTPUT_DIR = "outputs"
    DEFAULT_DEVICE = "auto"
    DEFAULT_NUM_IMAGES = 1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("image_generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def check_dependencies() -> None:
    """
    Check if required dependencies are installed.

    Raises:
        SystemExit: If dependencies are missing.
    """
    try:
        import diffusers
        import transformers

        logger.debug("Dependencies checked successfully")
    except ImportError as e:
        logger.error(
            "Missing dependencies. Please install from requirements.txt: %s", e
        )
        sys.exit(1)


def get_device(device_preference: str = DEFAULT_DEVICE) -> torch.device:
    """
    Determine the best device to use for generation.

    Args:
        device_preference (str): Preferred device ("cuda", "cpu", or "auto")

    Returns:
        torch.device: Selected device
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "cpu":
        return torch.device("cpu")
    else:
        # Auto-select based on availability
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipe(model_id: str, device: torch.device, hf_token: Optional[str] = None):
    """
    Load and configure the Stable Diffusion pipeline.

    Args:
        model_id (str): Hugging Face model identifier
        device (torch.device): Device to run the model on (CPU or CUDA)
        hf_token (Optional[str]): Hugging Face authentication token

    Returns:
        StableDiffusionPipeline: Configured pipeline instance
    """
    try:
        # Lazy import to fail early with friendly message if not installed
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        logger.info("Loading model pipeline: %s", model_id)

        # Create pipeline with appropriate settings
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_auth_token=hf_token,
        )

        # Configure scheduler for better quality
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Optimize for device
        if device.type == "cuda":
            pipe = pipe.to(device)
            try:
                pipe.enable_attention_slicing()
                logger.debug("Attention slicing enabled")
            except Exception as e:
                logger.warning("Could not enable attention slicing: %s", e)
        else:
            logger.warning("Running on CPU will be slow and memory intensive")

        return pipe
    except Exception as e:
        logger.error("Failed to load model pipeline: %s", e)
        sys.exit(1)


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Create a safe filename from text.

    Args:
        text (str): Text to sanitize
        max_length (int): Maximum length of filename

    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    safe_text = "".join(c for c in text if c.isalnum() or c in " _-").rstrip()
    # Limit length
    return safe_text[:max_length]


def generate(
    prompt: str,
    pipe,
    out_path: Path,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> List[str]:
    """
    Generate images from a text prompt.

    Args:
        prompt (str): Text description of the desired image
        pipe: Stable Diffusion pipeline instance
        out_path (Path): Directory to save generated images
        guidance_scale (float): Guidance scale for generation (default: 7.5)
        num_inference_steps (int): Number of denoising steps (default: 30)
        seed (Optional[int]): Random seed for reproducibility
        height (Optional[int]): Height of generated image
        width (Optional[int]): Width of generated image

    Returns:
        List[str]: Paths to saved image files
    """
    try:
        generator = torch.Generator(device=pipe.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        # Prepare generation parameters
        call_kwargs = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        if height is not None and width is not None:
            call_kwargs.update({"height": height, "width": width})

        logger.info(
            "Generating image with %d steps, guidance scale %.1f",
            num_inference_steps,
            guidance_scale,
        )

        # Generate images
        images = pipe(**call_kwargs).images

        # Save images
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, img in enumerate(images):
            # Create descriptive filename
            safe_prompt = sanitize_filename(prompt)
            fname = out_path / f"{safe_prompt}_{i+1}.png"

            img.save(fname)
            saved_paths.append(str(fname))
            logger.info("Saved image: %s", fname)

        return saved_paths
    except Exception as e:
        logger.error("Failed to generate image: %s", e)
        raise


def load_prompts_from_file(prompt_file: Union[str, Path]) -> List[str]:
    """
    Load prompts from a text file, one per line.

    Args:
        prompt_file (Union[str, Path]): Path to the prompt file

    Returns:
        List[str]: List of prompts

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    pfile = Path(prompt_file)
    if not pfile.exists():
        raise FileNotFoundError(f"Prompt file not found: {pfile}")

    with pfile.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    logger.info("Loaded %d prompts from file", len(lines))
    return lines


def main():
    """Main entry point for the image generation script."""
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="Generate images from text using Stable Diffusion (diffusers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "a red cube" --num_images 2
  python generate.py --prompt_file example_prompts.txt --steps 20
  python generate.py --prompt "fantasy landscape" --out results --guidance 8.0
        """,
    )

    parser.add_argument("--prompt", type=str, help="Text prompt to generate")
    parser.add_argument(
        "--prompt_file", type=str, help="Path to a file with one prompt per line"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model repo id (diffusers) to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to generate per prompt (default: {DEFAULT_NUM_IMAGES})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Height of generated images"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Width of generated images"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.prompt and not args.prompt_file:
        parser.error("Provide --prompt or --prompt_file. See --help for examples.")

    # Setup authentication
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    # Setup device
    device = get_device()
    logger.info("Using device: %s", device)

    # Load prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        try:
            prompts.extend(load_prompts_from_file(args.prompt_file))
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # Check if we have prompts to process
    if not prompts:
        logger.warning("No prompts to process")
        return

    # Load model pipeline
    logger.info("Loading model pipeline. This may take a while...")
    pipe = get_pipe(args.model, device, hf_token=hf_token)

    # Setup output directory
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_base)

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        logger.info("[%d/%d] Processing prompt: %s", idx + 1, len(prompts), prompt)

        # Generate the requested number of images for this prompt
        for n in range(args.num_images):
            try:
                # Create output directory for this prompt and image
                image_dir = out_base / f"prompt_{idx+1}_image_{n+1}"

                # Calculate seed for this image
                image_seed = args.seed
                if args.seed is not None:
                    image_seed = args.seed + n

                # Generate the image
                saved_paths = generate(
                    prompt,
                    pipe,
                    image_dir,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    seed=image_seed,
                    height=args.height,
                    width=args.width,
                )

            except Exception as e:
                logger.error(
                    "Failed to process prompt '%s', image %d: %s", prompt, n + 1, e
                )
                continue

    logger.info("Image generation complete. Results saved to: %s", out_base)


if __name__ == "__main__":
    main()
