# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# standard library dependencies
from itertools import chain
import logging


# internal dependencies
from vcfnp.vcflib import PyVariantCallFile
import vcfnp.config as config
from vcfnp.array import _filenames_from_arg, _warn_duplicates, \
    _variants_fields
from vcfnp.iter import itervariantstable


logger = logging.getLogger(__name__)
debug = logger.debug


class VariantsTable(object):
    def __init__(self, filename, region=None, fields=None, exclude_fields=None,
                 arities=None, flatten_filter=False, fill=b'.',
                 flatteners=None):
        self.filename = filename
        self.region = region
        self.fields = fields
        self.exclude_fields = exclude_fields
        self.arities = arities
        self.flatten_filter = flatten_filter
        self.fill = fill
        self.flatteners = flatteners

    def __iter__(self):

        vcf_fns = _filenames_from_arg(self.filename)

        # extract definitions from VCF header
        vcf = PyVariantCallFile(vcf_fns[0])

        # FILTER definitions
        filter_ids = vcf.filter_ids
        _warn_duplicates(filter_ids)
        filter_ids = sorted(set(filter_ids))
        if 'PASS' not in filter_ids:
            filter_ids.append('PASS')
        filter_ids = tuple(filter_ids)

        # INFO definitions
        _warn_duplicates(vcf.info_ids)
        info_ids = tuple(sorted(set(vcf.info_ids)))
        info_types = vcf.info_types

        # determine fields to extract
        fields = _variants_fields(self.fields, self.exclude_fields, info_ids)

        # turn arities into tuple for convenience
        if self.arities is None:
            arities = (None,) * len(fields)
        else:
            arities = tuple(self.arities.get(f) for f in fields)

        # determine if we need to parse the INFO field
        parse_info = any([f not in config.STANDARD_VARIANT_FIELDS
                          for f in fields])

        # convert to tuple
        info_types = tuple(info_types[f] if f in info_types else -1
                           for f in fields)

        # default flattening
        flatteners = self.flatteners
        if flatteners is None:
            flatteners = dict()
        for f in config.DEFAULT_FLATTEN:
            if f not in flatteners:
                ff, t = config.DEFAULT_FLATTEN[f]
                flatteners[f] = ff, t(self.fill)
        flatteners = tuple(flatteners[f] if f in flatteners else None
                           for f in fields)
        debug(flatteners)

        # make header row
        header = list()
        for f, flattener in zip(fields, flatteners):
            if self.flatten_filter and f == 'FILTER':
                for t in filter_ids:
                    header.append('FILTER_' + t)
            elif (self.arities is not None
                  and f in self.arities
                  and self.arities[f] > 1):
                for i in range(1, self.arities[f] + 1):
                    header.append(f + '_' + str(i))
            elif flattener is not None:
                fflds, _ = flattener
                header.extend(fflds)
            else:
                header.append(f)
        header = tuple(header)
        # make data rows
        data = itervariantstable(vcf_fns=vcf_fns, region=self.region,
                                 fields=fields, arities=arities,
                                 info_types=info_types, parse_info=parse_info,
                                 filter_ids=filter_ids,
                                 flatten_filter=self.flatten_filter,
                                 fill=self.fill, flatteners=flatteners)
        return chain((header,), data)
