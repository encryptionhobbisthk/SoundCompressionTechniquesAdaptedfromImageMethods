from setuptools import setup, find_packages

setup(
    name='ultimate-compressor',
    version='2.4.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        # ... add from requirements.txt
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Advanced compression program with Vedic math, prime hierarchies, and codec integration',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ultimate-compressor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
