#!/usr/bin/env python
"""
ARGprism: Deep Learning-based Antibiotic Resistance Gene Prediction Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='argprism',
    version='1.0.0',
    description='Deep Learning-based Antibiotic Resistance Gene Prediction Pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Muneeb',
    author_email='your.email@example.com',
    url='https://github.com/muneebdev7/ARGprism',
    license='MIT',
    
    packages=find_packages(),
    package_data={
        'argprism': [
            'data/ARGPrismDB.fasta',
            'data/metadata_arg.json',
            'models/best_model_fold4.pth',
        ],
    },
    include_package_data=True,
    
    python_requires='>=3.11,<3.14',
    
    install_requires=[
        'numpy>=2.2.3,<3.0',
        'pandas>=2.2.3,<3.0',
        'scipy>=1.15.2,<2.0',
        'scikit-learn>=1.6.1,<2.0',
        'biopython>=1.85,<2.0',
        'h5py>=3.13.0,<4.0',
        'torch>=2.6.0,<3.0',
        'transformers>=4.49.0,<5.0',
        'sentencepiece>=0.2.0',
        'matplotlib>=3.10.1,<4.0',
        'seaborn>=0.13.2,<1.0',
        'tqdm>=4.67.1,<5.0',
        'requests>=2.32.3,<3.0',
        'pillow>=11.2.1,<12.0',
        'regex>=2024.11.6',
        'boto3>=1.37.7',  # Optional, for S3 support
    ],
    
    entry_points={
        'console_scripts': [
            'argprism=argprism.cli:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    
    keywords='bioinformatics antibiotic-resistance deep-learning protein-sequences',
)
