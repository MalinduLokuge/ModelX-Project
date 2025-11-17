"""Setup file for CompeteML"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="competeml",
    version="1.0.0",
    description="Competition-Ready Automated ML System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CompeteML Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "autogluon>=1.0.0",
        "optuna>=3.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "imbalanced-learn>=0.11.0",
        "category-encoders>=2.6.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "ydata-profiling>=4.5.0",
            "plotly>=5.14.0",
            "shap>=0.43.0",
            "featuretools>=1.28.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "competeml=main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
