from setuptools import setup, find_packages

setup(
    name="swallowing-manifold-analysis",
    version="0.1.0",
    description="Manifold-trajectory framework for MRI-based swallowing analysis",
    author="[Authors]",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "geomstats>=2.7.0",
        "fdasrsf>=2.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
