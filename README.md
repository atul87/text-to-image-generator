# Task-02: Image Generation with Pre-trained Models

This small project demonstrates how to generate images from text prompts using **Stable Diffusion** (via Hugging Face *diffusers*).

## Contents
- `generate.py` — CLI script to generate images from prompts.
- `requirements.txt` — Python dependencies.
- `example_prompts.txt` — A few sample prompts, inspired by your slide.

## Features
- Generate images from text descriptions using state-of-the-art diffusion models
- Support for multiple prompts from command line or file input
- Configurable generation parameters (steps, guidance, dimensions)
- Progress tracking and logging
- Cross-platform compatibility
- Organized output structure

## Requirements
- Python 3.8+
- A machine with GPU is strongly recommended for reasonable speed.
- A Hugging Face account and an access token with `read` rights to the model repo (if using gated/stable models).
  - Set the token as an environment variable: `export HF_TOKEN="your_token_here"` (Linux/Mac) or `$env:HF_TOKEN="your_token_here"` (Windows)

## Installation
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic usage:
```bash
python generate.py --prompt "A futuristic city skyline at sunset, cinematic, ultra-detailed" --num_images 2
```

### Using prompts from a file:
```bash
python generate.py --prompt_file example_prompts.txt --steps 20 --guidance 8.0
```

### Advanced options:
```bash
python generate.py \
  --prompt "abstract art of geometric shapes" \
  --model "runwayml/stable-diffusion-v1-5" \
  --steps 50 \
  --guidance 7.5 \
  --height 512 \
  --width 512 \
  --num_images 3 \
  --out results \
  --seed 42
```

### List example prompts:
```bash
cat example_prompts.txt  # Linux/Mac
type example_prompts.txt  # Windows
```

## Command Line Options
```
--prompt PROMPT           Text prompt to generate
--prompt_file PROMPT_FILE Path to a file with one prompt per line
--model MODEL             Model repo id (diffusers) to use (default: runwayml/stable-diffusion-v1-5)
--hf_token HF_TOKEN       Hugging Face token (or set HF_TOKEN env var)
--num_images NUM_IMAGES   Number of images to generate per prompt (default: 1)
--out OUT                 Output directory (default: outputs)
--guidance GUIDANCE       Guidance scale (default: 7.5)
--steps STEPS             Number of inference steps (default: 30)
--seed SEED               Random seed for reproducibility
--height HEIGHT           Height of generated images
--width WIDTH             Width of generated images
```

## Output Structure
Generated images are organized in the output directory as follows:
```
outputs/
├── prompt_1/
│   ├── prompt_text_1.png
│   └── prompt_text_2.png
├── prompt_2/
│   ├── another_prompt_1.png
│   └── another_prompt_2.png
└── ...
```

## Notes & Safety
- The script includes error handling and logging to image_generation.log
- Generating photorealistic images of real people / public figures may be restricted by model rules. Follow Hugging Face / model license.
- Running on CPU will be significantly slower than GPU
- Model downloads may take several minutes on first run
- Generated images are saved in PNG format

## Troubleshooting
- **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies are installed
- **CUDA out of memory**: Reduce image dimensions or use CPU (--height 256 --width 256)
- **Slow generation**: Use GPU or reduce steps (--steps 10-20)
- **Authentication error**: Set HF_TOKEN environment variable for gated models