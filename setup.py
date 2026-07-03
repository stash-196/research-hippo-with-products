from setuptools import setup, find_packages

setup(
    name="lag-op-hippo",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "einops>=0.8.0",
        "matplotlib>=3.9.2",
        "numpy>=1.26.4",
        "scipy>=1.14.1",
        "seaborn>=0.13.2",
        "torch>=2.4.1",
        "tqdm>=4.66.5",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "flake8>=7.1.1",
            "black>=24.8.0",
            "isort>=5.13.2",
            "mypy>=1.11.2",
            "pre-commit>=3.8.0",
        ],
    },
    author="Anonymous",
    author_email="anonymous@domain.com",
    description="Research project on Lag Operator HiPPOs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
