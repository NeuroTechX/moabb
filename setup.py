from setuptools import setup, find_packages

setup(name='moabb',
      version='0.0.1',
      description='Mother of all BCI Benchmarks',
      url='',
      author='Alexandre Barachant',
      author_email='alexandre.barachant@gmail.com',
      license='BSD (3-clause)',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas',
                        'mne', 'pyriemann', 'pyyaml'],
      zip_safe=False)
