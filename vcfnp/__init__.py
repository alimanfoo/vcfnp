# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


__version__ = '2.1.1'


import vcfnp.config as config
import vcfnp.eff as eff
from vcfnp.vcflib import PyVariantCallFile as VariantCallFile
from vcfnp.array import variants, calldata, calldata_2d, view2d
from vcfnp.table import VariantsTable
