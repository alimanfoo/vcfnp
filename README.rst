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
	
	# load data from fixed fields (except INFO)
	v = vcfnp.variants(filename).view(np.recarray)
	
	# print some simple variant metrics
	print 'found %s variants (%s SNPs)' % (v.size, np.count_nonzero(v.is_snp))
	print 'QUAL mean (std): %s (%s)' % (np.mean(v.QUAL), np.std(v.QUAL))
	
	# load data from INFO field
	i = vcfnp.info(filename).view(np.recarray)
	
	# plot a histogram of variant depth
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.hist(i.DP)
	ax.set_title('DP histogram')
	ax.set_xlabel('DP')
	plt.show()
	
	# load data from sample columns 
	c = vcfnp.calldata(filename).view(np.recarray)
	c = vcfnp.view2d(c)
	
	# print some simple genotype metrics
	count_phased = np.count_nonzero(c.is_phased)
	count_variant = np.count_nonzero(np.any(c.genotype > 0, axis=2)) 
	count_missing = np.count_nonzero(~c.is_called)
	print 'calls (phased, variant, missing): %s (%s, %s, %s)' % (c.flatten().size, count_phased, count_variant, count_missing)
	
	# plot a histogram of genotype quality
	fig = plt.figure(2)
	ax = fig.add_subplot(111)
	ax.hist(c.GQ.flatten())
	ax.set_title('GQ histogram')
	ax.set_xlabel('GQ')
	plt.show()  

Acknowledgments
---------------

Based on Erik Garrison's `vcflib <https://github.com/ekg/vcflib>`_.
