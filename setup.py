#!/usr/bin/env python
"""Setup script for the ARGprism package."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
VERSION_NS: dict[str, str] = {}
exec((ROOT / "argprism" / "_version.py").read_text(), VERSION_NS)


def _read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    if not path.exists():
        return requirements
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


INSTALL_REQUIRES = _read_requirements(ROOT / "requirements.txt")
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="argprism",
    version=VERSION_NS["__version__"],
    description="Deep Learning-based Antibiotic Resistance Gene Prediction Pipeline",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Haseeb Manzoor, Muhammad Muneeb Nasir",
    author_email="haseebmanzoor667@gmail.com, talk2muneeb.ai@gmail.com",
    maintainer="Muhammad Muneeb Nasir",
    maintainer_email="talk2muneeb.ai@gmail.com",
    url="https://github.com/haseebmanzur/ARGPrism",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    package_data={
        "argprism": [
            "data/*.fasta",
            "data/*.json",
            "models/*.pth",
        ]
    },
    python_requires=">=3.11,<3.14",
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "argprism=argprism.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="bioinformatics antibiotic-resistance deep-learning protein-sequences",
)
