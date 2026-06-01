from setuptools import setup, find_packages

setup(
    name="data_analysis_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
        "scipy",
        "pydantic",
        "statsmodels"
    ],
)
