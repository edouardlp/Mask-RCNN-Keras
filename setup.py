import io
import os
from setuptools import setup
from setuptools import find_packages

NAME = 'maskrcnn'
DESCRIPTION = 'Mask-RCNN Keras'
URL = 'https://github.com/edouardlp/Mask-RCNN-Keras'
EMAIL = 'ed@laveryplante.com'
AUTHOR = 'Édouard Lavery-Plante'
REQUIRES_PYTHON = '>=2.7.0'
LICENSE = 'MIT'
VERSION = '0.1.0'
CLASSIFIERS = [
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ]

REQUIRED = [
    'Keras>=2.1.6',
    'tensorflow>=1.5.0',
    'numpy>=1.15.4'
]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
