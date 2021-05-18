import os
from setuptools import setup, find_packages

pkg = 'jiveplot'
__version__ = "1.0.0"
build_root = os.path.dirname(__file__)

def readme():
    """Get readme content for package long description"""
    with open(os.path.join(build_root, 'README.rst')) as f:
        return f.read()


def requirements():
    """Get package requirements"""
    with open(os.path.join(build_root, 'requirements.txt')) as f:
        return [pname.strip() for pname in f.readlines()]


setup(name=pkg,
      version=__version__,
      description="Python based visualization tool for AIPS++/CASA MeasurementSet data.",
      long_description=readme(),
      author="Harro Verkouter",
      author_email="verkouter@jive.eu",
      packages=find_packages(),
      url="https://github.com/haavee/jiveplot",
      license="GNU GPL 3",
      classifiers=["Intended Audience :: Developers",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      keywords="ms dataset statistics models plots",
      platforms=["OS Independent"],
      install_requires=requirements(),
      python_requires='>=3.6',
      include_package_data=True,
      scripts=['jiveplot/bin/jplotter'])
