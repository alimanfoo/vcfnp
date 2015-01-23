from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import vcfnp
vcfnp.__version__

filename = 'fixture/sample.vcf'

# load data from fixed fields (including INFO)
v = vcfnp.variants(filename, cache=True).view(np.recarray)

# print some simple variant metrics
print('found %s variants (%s SNPs)' % (v.size, np.count_nonzero(v.is_snp)))
print('QUAL mean (std): %s (%s)' % (np.mean(v.QUAL), np.std(v.QUAL)))

# plot a histogram of variant depth
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.hist(v.DP)
ax.set_title('DP histogram')
ax.set_xlabel('DP')
plt.show()

# load data from sample columns
c = vcfnp.calldata_2d(filename, cache=True).view(np.recarray)

# print some simple genotype metrics
count_phased = np.count_nonzero(c.is_phased)
count_variant = np.count_nonzero(np.any(c.genotype > 0, axis=2))
count_missing = np.count_nonzero(~c.is_called)
print('calls (phased, variant, missing): %s (%s, %s, %s)'
    % (c.flatten().size, count_phased, count_variant, count_missing))

# plot a histogram of genotype quality
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.hist(c.GQ.flatten())
ax.set_title('GQ histogram')
ax.set_xlabel('GQ')
plt.show()
