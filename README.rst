vcfnp
=====

Load data from a VCF (variant call file) into numpy arrays or an HDF5 file.

Installation
------------

Installation requires numpy and cython::

	$ pip install vcfnp

...or::

	$ git clone --recursive git://github.com/alimanfoo/vcfnp.git
	$ cd vcfnp
	$ python setup.py build_ext --inplace

Usage
-----

From Python::

	import sys
	import vcfnp
	import numpy as np
	import matplotlib.pyplot as plt
	
	filename = '/path/to/my.vcf'
	
	# load data from fixed fields (including INFO)
	V = vcfnp.variants(filename, cache=True).view(np.recarray)
	
	# print some simple variant metrics
	print 'found %s variants (%s SNPs)' % (v.size, np.count_nonzero(v.is_snp))
	print 'QUAL mean (std): %s (%s)' % (np.mean(v.QUAL), np.std(v.QUAL))
	
	# plot a histogram of variant depth
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.hist(V.DP)
	ax.set_title('DP histogram')
	ax.set_xlabel('DP')
	plt.show()
	
	# load data from sample columns 
	C = vcfnp.calldata_2d(filename, cache=True).view(np.recarray)

	# print some simple genotype metrics
	count_phased = np.count_nonzero(C2d.is_phased)
	count_variant = np.count_nonzero(np.any(C2d.genotype > 0, axis=2))
	count_missing = np.count_nonzero(~C2d.is_called)
	print 'calls (phased, variant, missing): %s (%s, %s, %s)' % (C2d.flatten().size, count_phased, count_variant, count_missing)
	
	# plot a histogram of genotype quality
	fig = plt.figure(2)
	ax = fig.add_subplot(111)
	ax.hist(C2d.GQ.flatten())
	ax.set_title('GQ histogram')
	ax.set_xlabel('GQ')
	plt.show()  

Command line scripts are also provided to facilitate parallelizing the conversion of a VCF file to NPY arrays split
by genome region. For example, the following command will create an NPY file containing a variants array for the
second 100kb on chromosome 2::

    $ vcf2npy \
        --vcf /path/to/my.vcf \
        --fasta /path/to/ref.fa \
        --output-dir /path/to/npy/output \
        --array-type variants \
        --chromosome chr20 \
        --task-size 100000 \
        --task-index 2 \
        --progress 1000

For those with access to a cluster running Sun Grid Engine a script is provided to submit a job array parallelizing the
conversion, e.g.::

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

It should be straightforward to adapt this script to run on other parallel computing platforms, see the
`scripts <https://github.com/alimanfoo/vcfnp/tree/master/scripts>`_ folder for the source code.

A script is also provided to load data from multiple NPY files into a single HDF5 file. E.g., after having converted
a VCF file to 100kb variants and calldata_2d NPY splits, run something like::

    $ vcfnpy2hdf5 \
        --vcf /path/to/my.vcf \
        --input-dir /path/to/npy/output \
        --output /path/to/my.h5

If you want to group the data by chromosome, do something like the following for each chromosome separately::

    $ vcfnpy2hdf5 \
        --vcf /path/to/my.vcf \
        --input-dir /path/to/npy/output \
        --input-filename-template {array_type}.chr20*.npy \
        --output /path/to/my.h5 \
        --group chr20

Release Notes
-------------


* `1.10 <https://github.com/alimanfoo/vcfnp/issues?milestone=7&state=closed>`_
* `1.9 <https://github.com/alimanfoo/vcfnp/issues?milestone=6&state=closed>`_
* `1.8 <https://github.com/alimanfoo/vcfnp/issues?milestone=5&state=closed>`_
* `1.7 <https://github.com/alimanfoo/vcfnp/issues?milestone=4&page=1&state=closed>`_
* `1.6 <https://github.com/alimanfoo/vcfnp/issues?milestone=3&page=1&state=closed>`_
* `1.5 <https://github.com/alimanfoo/vcfnp/issues?milestone=1&state=closed>`_
* `1.0 <https://github.com/alimanfoo/vcfnp/issues?milestone=2&page=1&state=closed>`_ - Note that as of version 1.0 the info() function has been removed and the variants() function now loads data from any of the VCF fixed fields including INFO. I.e., the variants() function gives access to all variant-level data in a single structured array. This is convenient for many use cases, e.g., using PyTables in-kernel queries to select variants passing some filtering criteria.

Acknowledgments
---------------

Based on Erik Garrison's `vcflib <https://github.com/ekg/vcflib>`_.
