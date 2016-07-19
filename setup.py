from __future__ import print_function
from distutils.core import setup
from distutils.extension import Extension
import os
from ast import literal_eval

try:
    from Cython.Build import cythonize
except ImportError:
    print('Cython is required.')
    raise


def get_version(source='vcfnp/__init__.py'):
    with open(source) as f:
        for line in f:
            if line.startswith('__version__'):
                return literal_eval(line.partition('=')[2].lstrip())
    raise ValueError("__version__ not found")


vcflib_dir = os.path.join(os.getcwd(), 'vcflib')
smithwaterman_dir = os.path.join(vcflib_dir, 'smithwaterman')
tabixpp_dir = os.path.join(vcflib_dir, 'tabixpp')
vcflib_sources = ('Variant.cpp', 'ssw.c', 'ssw_cpp.cpp', 'split.cpp')
smithwaterman_sources = ('BandedSmithWaterman.cpp',
                         'SmithWatermanGotoh.cpp',
                         'Repeats.cpp',
                         'disorder.c',
                         'LeftAlign.cpp',
                         'IndelAllele.cpp')
tabixpp_sources = ('bedidx.c', 'bgzf.c', 'index.c', 'knetfile.c', 'kstring.c',
                   'tabix.cpp')


def get_vcflib_sources():
    sources = list()
    sources += [os.path.join(vcflib_dir, s) for s in vcflib_sources]
    sources += [os.path.join(smithwaterman_dir, s)
                for s in smithwaterman_sources]
    sources += [os.path.join(tabixpp_dir, s) for s in tabixpp_sources]
    return sources


compat_extension = Extension(
    'vcfnp.compat',
    language='c++',
    sources=['vcfnp/compat.pyx']
)


vcflib_extension = Extension(
    'vcfnp.vcflib',
    sources=['vcfnp/vcflib.pyx'] + get_vcflib_sources(),
    language='c++',
    include_dirs=[vcflib_dir, smithwaterman_dir, tabixpp_dir, './vcfnp'],
    libraries=['m', 'z'],
    extra_compile_args=['-O3']
)


iter_extension = Extension(
    'vcfnp.iter',
    sources=['vcfnp/iter.pyx'] + get_vcflib_sources(),
    language='c++',
    include_dirs=[vcflib_dir, smithwaterman_dir, tabixpp_dir, './vcfnp'],
    libraries=['m', 'z'],
    extra_compile_args=['-O3']
)


setup(
    name='vcfnp',
    version=get_version(),
    author='Alistair Miles',
    author_email='alimanfoo@googlemail.com',
    url='https://github.com/alimanfoo/vcfnp',
    license='MIT License',
    description='Load numpy arrays from a VCF (variant call file).',
    long_description=open('README.rst').read(),
    classifiers=['Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Software Development :: Libraries :: Python Modules'
                 ],
    package_dir={'': '.'},
    packages=['vcfnp', 'vcfnp.test'],
    ext_modules=cythonize([
        compat_extension,
        vcflib_extension,
        iter_extension,
    ]),
    scripts=['scripts/vcf2npy',
             'scripts/qsub_vcf2npy',
             'scripts/vcfnpy2hdf5',
             'scripts/vcf2csv',
             'scripts/vcf2hdf5_parallel'
             ],
    install_requires=['numpy']
)
