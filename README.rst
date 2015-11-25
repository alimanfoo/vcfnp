vcfnp
=====

Load data from a VCF (variant call format) file into numpy arrays, and
(optionally) from there into an HDF5 file.

Installation
------------

Installation requires numpy and cython::

    $ pip install cython
    $ pip install numpy
    $ pip install vcfnp

...or::

	$ git clone --recursive git://github.com/alimanfoo/vcfnp.git
	$ cd vcfnp
	$ python setup.py build_ext --inplace

Usage
-----

For usage from Python, see the `IPython notebook example
<http://nbviewer.ipython.org/github/alimanfoo/vcfnp/blob/master/example.ipynb>`_,
or try::

    >>> from __future__ import print_function, division
    >>> import numpy as np
    >>> import matplotlib
    >>> matplotlib.use('TkAgg')
    >>> import matplotlib.pyplot as plt
    >>> import vcfnp
    >>> vcfnp.__version__
    '2.0.0'
    >>> filename = 'fixture/sample.vcf'
    >>> # load data from fixed fields (including INFO)
    ... v = vcfnp.variants(filename, cache=True).view(np.recarray)
    [vcfnp] 2015-01-23 11:10:46.670723 :: caching is enabled
    [vcfnp] 2015-01-23 11:10:46.670830 :: cache file available
    [vcfnp] 2015-01-23 11:10:46.670866 :: loading from cache file fixture/sample.vcf.vcfnp_cache/variants.npy
    >>> # print some simple variant metrics
    ... print('found %s variants (%s SNPs)' % (v.size, np.count_nonzero(v.is_snp)))
    found 9 variants (5 SNPs)
    >>> print('QUAL mean (std): %s (%s)' % (np.mean(v.QUAL), np.std(v.QUAL)))
    QUAL mean (std): 25.0667 (22.816)
    >>> # plot a histogram of variant depth
    ... fig = plt.figure(1)
    >>> ax = fig.add_subplot(111)
    >>> ax.hist(v.DP)
    (array([ 4.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,  2.]), array([  0. ,   1.4,   2.8,   4.2,   5.6,   7. ,   8.4,   9.8,  11.2,
            12.6,  14. ]), <a list of 10 Patch objects>)
    >>> ax.set_title('DP histogram')
    <matplotlib.text.Text object at 0x7f28f18f5c50>
    >>> ax.set_xlabel('DP')
    <matplotlib.text.Text object at 0x7f28f207c3c8>
    >>> plt.show()
    >>> # load data from sample columns
    ... c = vcfnp.calldata_2d(filename, cache=True).view(np.recarray)
    >>> # print some simple genotype metrics
    ... count_phased = np.count_nonzero(c.is_phased)
    >>> count_variant = np.count_nonzero(np.any(c.genotype > 0, axis=2))
    >>> count_missing = np.count_nonzero(~c.is_called)
    >>> print('calls (phased, variant, missing): %s (%s, %s, %s)'
    ...     % (c.flatten().size, count_phased, count_variant, count_missing))
    calls (phased, variant, missing): 27 (14, 12, 2)
    >>> # plot a histogram of genotype quality
    ... fig = plt.figure(2)
    >>> ax = fig.add_subplot(111)
    >>> ax.hist(c.GQ.flatten())
    (array([ 15.,   0.,   1.,   1.,   0.,   1.,   2.,   4.,   2.,   1.]), array([  0. ,   6.1,  12.2,  18.3,  24.4,  30.5,  36.6,  42.7,  48.8,
            54.9,  61. ]), <a list of 10 Patch objects>)
    >>> ax.set_title('GQ histogram')
    <matplotlib.text.Text object at 0x7f28f1eb1cc0>
    >>> ax.set_xlabel('GQ')
    <matplotlib.text.Text object at 0x7f28f18d4fd0>
    >>> plt.show()

Command line scripts are also provided to facilitate parallelizing the
conversion of a VCF file to NPY arrays split by genome region. For
example, the following command will create an NPY file containing a
variants array for the second 100kb on chromosome 2::

    $ vcf2npy \
        --vcf /path/to/my.vcf \
        --fasta /path/to/ref.fa \
        --output-dir /path/to/npy/output \
        --array-type variants \
        --chromosome chr20 \
        --task-size 100000 \
        --task-index 2 \
        --progress 1000

For those with access to a cluster running Sun Grid Engine a script is
provided to submit a job array parallelizing the conversion, e.g.::

    $ qsub_vcf2npy \
        --vcf /path/to/my.vcf \
        --fasta /path/to/ref.fa \
        --output-dir /path/to/npy/output \
        --array-type variants \
        --chromosome chr20 \
        --task-size 100000 \
        --progress 1000 \
        -l h_vmem=1G \
        -N test_vcfnp \
        -j y \
        -o /path/to/sge/logs \
        -q shortrun.q

It should be straightforward to adapt this script to run on other
parallel computing platforms, see the `scripts
<https://github.com/alimanfoo/vcfnp/tree/master/scripts>`_ folder for
the source code.

A script is also provided to load data from multiple NPY files into a
single HDF5 file. E.g., after having converted a VCF file to 100kb
variants and calldata_2d NPY splits, run something like::

    $ vcfnpy2hdf5 \
        --vcf /path/to/my.vcf \
        --input-dir /path/to/npy/output \
        --output /path/to/my.h5

If you want to group the data by chromosome, do something like the
following for each chromosome separately::

    $ vcfnpy2hdf5 \
        --vcf /path/to/my.vcf \
        --input-dir /path/to/npy/output \
        --input-filename-template {array_type}.chr20*.npy \
        --output /path/to/my.h5 \
        --group chr20

There is also a script which will process a VCF file in parallel on the
local machine and load into an HDF5 file, e.g.::

    $ vcf2hdf5_parallel \
        --vcf /path/to/my.vcf \
        --fasta /path/to/refseq.fa

Finally, there is a script fo converting the fixed fields of a VCF
file to CSV, e.g.::

    $ vcf2csv \
        --vcf /path/to/my.vcf \
        --dialect excel-tab \
        --flatten-filter

Acknowledgments
---------------

Based on Erik Garrison's `vcflib <https://github.com/ekg/vcflib>`_.
