#!/bin/bash
# Script to initialize the repository for GitHub

echo "Initializing Git repository..."
git init

echo "Adding all files..."
git add .

echo "Making first commit..."
git commit -m "Initial commit: Text-to-Image Generator with Stable Diffusion"

echo "Repository initialized successfully!"
echo ""
echo "To push to GitHub:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"