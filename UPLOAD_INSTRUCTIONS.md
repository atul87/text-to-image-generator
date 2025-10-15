# Instructions for Uploading to GitHub

This repository is ready to be uploaded to GitHub. Here's what has been prepared:

## Files Included

1. **Core Application Files:**
   - `generate.py` - Main script for text-to-image generation
   - `config.py` - Configuration settings
   - `example_prompts.txt` - Sample prompts for testing
   - `requirements.txt` - Python dependencies

2. **Documentation:**
   - `README.md` - Comprehensive documentation
   - `LICENSE` - MIT License
   - `UPLOAD_INSTRUCTIONS.md` - This file (can be deleted before upload)

3. **Packaging Files:**
   - `setup.py` - Traditional Python packaging
   - `pyproject.toml` - Modern Python packaging

4. **Repository Management:**
   - `.gitignore` - Excludes unnecessary files
   - `init_repo.sh` - Bash script to initialize Git repository
   - `init_repo.bat` - Windows batch file to initialize Git repository

## Steps to Upload to GitHub

1. **Initialize the Git Repository:**
   - On Linux/Mac: Run `./init_repo.sh`
   - On Windows: Run `init_repo.bat`
   - Or manually run:
     ```bash
     git init
     git add .
     git commit -m "Initial commit: Text-to-Image Generator with Stable Diffusion"
     ```

2. **Create a New Repository on GitHub:**
   - Go to https://github.com/new
   - Create a new repository (don't initialize with README)
   - Copy the repository URL

3. **Connect Local Repository to GitHub:**
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

## What's Excluded

The following files/directories are excluded via `.gitignore` and will not be uploaded:
- Virtual environments (`.venv/`, `venv/`, `env/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Log files (`*.log`)
- Output directories (`outputs/`, `test_output/`, `example_output/`)
- IDE configuration files

## Ready for Use

The repository is now ready for:
- GitHub upload
- Collaboration
- Package installation via pip
- Easy setup on new machines

After uploading, users can install and use the package with:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate.py --prompt "a red cube"
```