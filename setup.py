"""Setup configuration for MarketRegimeML package."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read version from package
version = "0.1.0"
try:
    exec(open("marketregimeml/__version__.py").read())
    version = __version__  # noqa
except:
    pass

setup(
    name="marketregimeml",
    version=version,
    author="MarketRegimeML Contributors",
    author_email="marketregimeml@github.com",
    description="Market Regime Detection using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/firestrand/marketregimeml",
    project_urls={
        "Bug Reports": "https://github.com/firestrand/marketregimeml/issues",
        "Source": "https://github.com/firestrand/marketregimeml",
        "Documentation": "https://github.com/firestrand/marketregimeml/blob/main/docs/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "scripts", "benchmarks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "hmmlearn>=0.3.0",
        "arch>=5.0.0",
        "statsmodels>=0.13.0",
        "pyyaml>=6.0",
        "joblib>=1.0.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "torch": [
            "torch>=1.10.0",
        ],
        "xgboost": [
            "xgboost>=1.5.0",
        ],
        "api": [
            "alpha-vantage>=2.3.0",
            "v20>=3.0.25.0",  # OANDA v20 API
        ],
        "viz": [
            "plotly>=5.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "isort>=5.10.0",
            "radon>=5.1.0",
            "jscpd>=1.0.0",
            "twine>=4.0.0",
            "wheel>=0.37.0",
        ],
        "all": [
            "torch>=1.10.0",
            "xgboost>=1.5.0",
            "alpha-vantage>=2.3.0",
            "v20>=3.0.25.0",
            "plotly>=5.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    keywords=[
        "market regime detection",
        "financial machine learning",
        "regime switching",
        "hmm",
        "ensemble learning",
        "quantitative finance",
        "market analysis",
        "trading",
        "finance",
    ],
    include_package_data=True,
    package_data={
        "marketregimeml": ["config/*.yaml"],
    },
    zip_safe=False,
)