vcfnp
=====

Load numpy arrays from a VCF (variant call file).

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

::

	import sys
	import vcfnp
	import numpy as np
	import matplotlib.pyplot as plt
	
	filename = '/path/to/my.vcf'
	
	# load data from fixed fields (including INFO)
	V = vcfnp.variants(filename).view(np.recarray)
	
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
	C = vcfnp.calldata(filename).view(np.recarray)
	C2d = vcfnp.view2d(C)
	
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

Release Notes
-------------

* `1.6 <https://github.com/alimanfoo/vcfnp/issues?milestone=3&page=1&state=closed>`_
* `1.5 <https://github.com/alimanfoo/vcfnp/issues?milestone=1&state=closed>`_
* `1.0 <https://github.com/alimanfoo/vcfnp/issues?milestone=2&page=1&state=closed>`_ - Note that as of version 1.0 the info() function has been removed and the variants() function now loads data from any of the VCF fixed fields including INFO. I.e., the variants() function gives access to all variant-level data in a single structured array. This is convenient for many use cases, e.g., using PyTables in-kernel queries to select variants passing some filtering criteria.

Acknowledgments
---------------

Based on Erik Garrison's `vcflib <https://github.com/ekg/vcflib>`_.
