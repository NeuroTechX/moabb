from setuptools import find_packages, setup


setup(
    name="moabb",
    version="0.2.1",
    description="Mother of all BCI Benchmarks",
    url="",
    author="Alexandre Barachant, Vinay Jayaram",
    author_email="{alexandre.barachant, vinayjayaram13}@gmail.com",
    license="BSD (3-clause)",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "mne",
        "pyriemann",
        "pyyaml",
    ],
    zip_safe=False,
)
