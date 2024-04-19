from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="texture-model-stylization",
    version="1.0.0",
    author="omprxkash",
    description="Tactile texture generation and neural style transfer for 3D surface stylization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omprxkash/texture-model-stylization",
    packages=find_packages(exclude=["notebooks", "assets", "paper"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "texture-stylize=inference.run_style_transfer:main",
            "texture-heightmap=inference.generate_heightmap:main",
            "texture-demo=demo.app:main",
        ]
    },
)
