from setuptools import setup, find_packages

setup(
    name="beavervision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "mediapipe",
        "prometheus-client",
        "pydantic-settings",
        "python-magic",
        "moviepy",
    ],
    python_requires=">=3.10",
)