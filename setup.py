from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="text-to-image-generator",
    version="1.0.0",
    author="Atul",
    author_email="atul@example.com",
    description="A tool to generate images from text prompts using Stable Diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-to-image-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "text2image=generate:main",
        ],
    },
    include_package_data=True,
    keywords=["stable-diffusion", "text-to-image", "ai", "machine-learning", "diffusers", "huggingface"],
)