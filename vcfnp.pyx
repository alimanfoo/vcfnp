# cython: profile = False
# cython: boundscheck = False
# cython: wraparound = False
# cython: embedsignature = True
from __future__ import print_function, division, absolute_import


__version__ = '2.0.0.dev0'


import sys
import re
from itertools import chain
import numpy as np
cimport numpy as np
import time
from itertools import islice
import os
from datetime import datetime
import logging
from cpython.version cimport PY_MAJOR_VERSION


if PY_MAJOR_VERSION < 3:
    string_types = basestring,
else:
    string_types = str,


logger = logging.getLogger(__name__)
import inspect
debug = lambda msg: logger.debug('%s: %s' % (inspect.stack()[0][3], msg))




def _calldata_fields(fields, exclude_fields, format_ids):
    """Utility function to determine which calldata (i.e., FORMAT) fields to
    extract."""
    if fields is None:
        # no fields specified by user
        # default to all standard fields plus all FORMAT fields in VCF header
        fields = STANDARD_CALLDATA_FIELDS + format_ids
    else:
        # fields specified by user
        for f in fields:
            # check if field is standard or defined in VCF header
            if f not in STANDARD_CALLDATA_FIELDS and f not in format_ids:
                # support extracting FORMAT even if not declared in header,
                # but warn...
                print('WARNING: no definition found for field %s' % f,
                      file=sys.stderr)
    # process exclusions
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]
    return tuple(fields)


def _calldata_arities(fields, arities, format_counts, ploidy):
    if arities is None:
        arities = dict()
    for f, vcf_count in zip(fields, format_counts):
        if f not in arities:
            if f == 'genotype':
                arities[f] = ploidy
            elif f in DEFAULT_CALLDATA_ARITY:
                arities[f] = DEFAULT_CALLDATA_ARITY[f]
            elif vcf_count == ALLELE_NUMBER:
                # default to 2 (biallelic)
                arities[f] = 2
            elif vcf_count == GENOTYPE_NUMBER:
                # arity = (n + p - 1) choose p (n is number of alleles; p is
                # ploidy)
                # default to biallelic (n = 2)
                arities[f] = ploidy + 1
            elif vcf_count <= 0:
                # catch any other cases of non-specific arity
                arities[f] = 1
            else:
                arities[f] = vcf_count
    return tuple(arities[f] for f in fields)


def _calldata_fills(fields, fills, format_types, ploidy):
    if fills is None:
        fills = dict()
    for f, vcf_type in zip(fields, format_types):
        if f not in fills:
            if f == 'GT':
                fills[f] = '/'.join(['.'] * ploidy)
            elif f in DEFAULT_CALLDATA_FILL:
                fills[f] = DEFAULT_CALLDATA_FILL[f]
            else:
                fills[f] = DEFAULT_FILL_MAP[vcf_type]
    return tuple(fills[f] for f in fields)


def _calldata_dtype(fields, dtypes, format_types, arities, samples, ploidy):

    # construct a numpy dtype for structured array cells
    cell_dtype = list()
    for f, vcf_type, n in zip(fields, format_types, arities):
        if dtypes is not None and f in dtypes:
            t = dtypes[f]
        elif f == 'GT':
            t = 'a%d' % ((ploidy*2)-1)
        elif f in DEFAULT_CALLDATA_DTYPE:
            # known field
            t = DEFAULT_CALLDATA_DTYPE[f]
        else:
            t = DEFAULT_TYPE_MAP[vcf_type]
        if n == 1:
            cell_dtype.append((f, t))
        else:
            cell_dtype.append((f, t, (n,)))

    # construct a numpy dtype for structured array
    dtype = [(s, cell_dtype) for s in samples]
    return dtype


class _CalldataLoader(_ArrayLoader):

    array_type = 'calldata'

    def build(self):
        log = self.log

        # open VCF file to inspect header
        vcf_fns = self.vcf_fns
        vcf = PyVariantCallFile(vcf_fns[0])

        # extract FORMAT definitions
        _warn_duplicates(vcf.format_ids)
        format_ids = tuple(sorted(set(vcf.format_ids)))
        format_types = vcf.format_types
        format_counts = vcf.format_counts

        # extract sample IDs
        all_samples = vcf.sampleNames

        # determine which samples to extract
        samples = self.samples
        if samples is None:
            samples = all_samples
        else:
            for s in samples:
                assert s in all_samples, 'unknown sample: %s' % s
        samples = tuple(samples)

        # determine which fields to extract
        fields = _calldata_fields(self.fields, self.exclude_fields, format_ids)

        # support for working around VCFs with bad FORMAT headers
        vcf_types = self.vcf_types
        for f in fields:
            if f not in STANDARD_CALLDATA_FIELDS and f not in format_ids:
                # fall back to unary string; can be overridden with
                # vcf_types, dtypes and arities args
                format_types[f] = FIELD_STRING
                format_counts[f] = 1
            if vcf_types is not None and f in vcf_types:
                # override type declared in VCF header
                format_types[f] = TYPESTRING2KEY[vcf_types[f]]

        # conveniences
        format_types = tuple(format_types[f] if f in format_types else -1
                             for f in fields)
        format_counts = tuple(format_counts[f] if f in format_counts else -1
                              for f in fields)

        # determine expected number of values for each field
        ploidy = self.plody
        arities = _calldata_arities(fields, self.arities, format_counts, ploidy)

        # determine fill values to use where number of values is less than
        # expectation
        fills = _calldata_fills(fields, self.fills, format_types, ploidy)

        # construct a numpy dtype for structured array
        dtype = _calldata_dtype(fields, self.dtypes, format_types, arities,
                                samples, ploidy)

        # zip up field parameters
        fieldspec = zip(fields, arities, fills, format_types)

        # set up iterator
        condition = self.condition
        region = self.region
        if condition is not None:
            it = _itercalldata_with_condition(vcf_fns, region, samples, ploidy,
                                             fieldspec, condition)
        else:
            it = _itercalldata(vcf_fns, region, samples, ploidy, fieldspec)

        # slice iterator
        slice_args = self.slice_args
        if slice_args:
            it = islice(it, *slice_args)

        # build an array from the iterator
        arr = _fromiter(it, dtype, self.count, self.progress, log)

        return arr


def calldata(vcf_fn, region=None, samples=None, ploidy=2, fields=None,
             exclude_fields=None, dtypes=None, arities=None, fills=None,
             vcf_types=None, count=None, progress=0, logstream=None,
             condition=None, slice_args=None, verbose=False, cache=False,
             cachedir=None, skip_cached=False):
    """
    Load a numpy 1-dimensional structured array with data from the sample
    columns of a VCF file.

    Parameters
    ----------

    vcf_fn: string or list
        Name of the VCF file or list of file names.
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'.
    fields: list or array-like
        List of fields to extract from the VCF.
    exclude_fields: list or array-like
        Fields to exclude from extraction.
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred ones
    arities: dict or dict-like
        Override the amount of values to expect.
    fills: dict or dict-like
        Dictionary containing field:fillvalue mappings used to override the
        default fill in values in VCF fields.
    vcf_types: dict or dict-like
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header.
    count: int
        Attempt to extract a specific number of records.
    progress: int
        If greater than 0, log parsing progress.
    logstream: file or file-like object
        Stream to use for logging progress.
    condition: array
        Boolean array defining which rows to load.
    slice_args: tuple or list
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every
        10th row from the first 1000.
    verbose: bool
        Log more messages.
    cache: bool
        If True, save the resulting numpy array to disk, and load from the
        cache if present rather than rebuilding from the VCF.
    cachedir: string
        Manually specify the directory to use to store cache files.
    skip_cached: bool
        If True and cache file is fresh, do not load and return None.

    Examples
    --------

        >>> from vcfnp import calldata, view2d
        >>> C = calldata('fixture/sample.vcf')
        >>> C
        array([ ((True, True, [0, 0], 0, 0, '0|0', [10, 10]), (True, True, [0, 0], 0, 0, '0|0', [10, 10]), (True, False, [0, 1], 0, 0, '0/1', [3, 3])),
               ((True, True, [0, 0], 0, 0, '0|0', [10, 10]), (True, True, [0, 0], 0, 0, '0|0', [10, 10]), (True, False, [0, 1], 0, 0, '0/1', [3, 3])),
               ((True, True, [0, 0], 1, 48, '0|0', [51, 51]), (True, True, [1, 0], 8, 48, '1|0', [51, 51]), (True, False, [1, 1], 5, 43, '1/1', [0, 0])),
               ((True, True, [0, 0], 3, 49, '0|0', [58, 50]), (True, True, [0, 1], 5, 3, '0|1', [65, 3]), (True, False, [0, 0], 3, 41, '0/0', [0, 0])),
               ((True, True, [1, 2], 6, 21, '1|2', [23, 27]), (True, True, [2, 1], 0, 2, '2|1', [18, 2]), (True, False, [2, 2], 4, 35, '2/2', [0, 0])),
               ((True, True, [0, 0], 0, 54, '0|0', [56, 60]), (True, True, [0, 0], 4, 48, '0|0', [51, 51]), (True, False, [0, 0], 2, 61, '0/0', [0, 0])),
               ((True, False, [0, 1], 4, 0, '0/1', [0, 0]), (True, False, [0, 2], 2, 17, '0/2', [0, 0]), (False, False, [-1, -1], 3, 40, './.', [0, 0])),
               ((True, False, [0, 0], 0, 0, '0/0', [0, 0]), (True, True, [0, 0], 0, 0, '0|0', [0, 0]), (False, False, [-1, -1], 0, 0, './.', [0, 0])),
               ((True, False, [0, -1], 0, 0, '0', [0, 0]), (True, False, [0, 1], 0, 0, '0/1', [0, 0]), (True, True, [0, 2], 0, 0, '0|2', [0, 0]))],
              dtype=[('NA00001', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00002', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00003', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])])
        >>> C['NA00001']
        array([(True, True, [0, 0], 0, 0, '0|0', [10, 10]),
               (True, True, [0, 0], 0, 0, '0|0', [10, 10]),
               (True, True, [0, 0], 1, 48, '0|0', [51, 51]),
               (True, True, [0, 0], 3, 49, '0|0', [58, 50]),
               (True, True, [1, 2], 6, 21, '1|2', [23, 27]),
               (True, True, [0, 0], 0, 54, '0|0', [56, 60]),
               (True, False, [0, 1], 4, 0, '0/1', [0, 0]),
               (True, False, [0, 0], 0, 0, '0/0', [0, 0]),
               (True, False, [0, -1], 0, 0, '0', [0, 0])],
              dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
        >>> C2d = view2d(C)
        >>> C2d
        array([[(True, True, [0, 0], 0, 0, '0|0', [10, 10]),
                (True, True, [0, 0], 0, 0, '0|0', [10, 10]),
                (True, False, [0, 1], 0, 0, '0/1', [3, 3])],
               [(True, True, [0, 0], 0, 0, '0|0', [10, 10]),
                (True, True, [0, 0], 0, 0, '0|0', [10, 10]),
                (True, False, [0, 1], 0, 0, '0/1', [3, 3])],
               [(True, True, [0, 0], 1, 48, '0|0', [51, 51]),
                (True, True, [1, 0], 8, 48, '1|0', [51, 51]),
                (True, False, [1, 1], 5, 43, '1/1', [0, 0])],
               [(True, True, [0, 0], 3, 49, '0|0', [58, 50]),
                (True, True, [0, 1], 5, 3, '0|1', [65, 3]),
                (True, False, [0, 0], 3, 41, '0/0', [0, 0])],
               [(True, True, [1, 2], 6, 21, '1|2', [23, 27]),
                (True, True, [2, 1], 0, 2, '2|1', [18, 2]),
                (True, False, [2, 2], 4, 35, '2/2', [0, 0])],
               [(True, True, [0, 0], 0, 54, '0|0', [56, 60]),
                (True, True, [0, 0], 4, 48, '0|0', [51, 51]),
                (True, False, [0, 0], 2, 61, '0/0', [0, 0])],
               [(True, False, [0, 1], 4, 0, '0/1', [0, 0]),
                (True, False, [0, 2], 2, 17, '0/2', [0, 0]),
                (False, False, [-1, -1], 3, 40, './.', [0, 0])],
               [(True, False, [0, 0], 0, 0, '0/0', [0, 0]),
                (True, True, [0, 0], 0, 0, '0|0', [0, 0]),
                (False, False, [-1, -1], 0, 0, './.', [0, 0])],
               [(True, False, [0, -1], 0, 0, '0', [0, 0]),
                (True, False, [0, 1], 0, 0, '0/1', [0, 0]),
                (True, True, [0, 2], 0, 0, '0|2', [0, 0])]],
              dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
        >>> C2d['genotype']
        array([[[ 0,  0],
                [ 0,  0],
                [ 0,  1]],

               [[ 0,  0],
                [ 0,  0],
                [ 0,  1]],

               [[ 0,  0],
                [ 1,  0],
                [ 1,  1]],

               [[ 0,  0],
                [ 0,  1],
                [ 0,  0]],

               [[ 1,  2],
                [ 2,  1],
                [ 2,  2]],

               [[ 0,  0],
                [ 0,  0],
                [ 0,  0]],

               [[ 0,  1],
                [ 0,  2],
                [-1, -1]],

               [[ 0,  0],
                [ 0,  0],
                [-1, -1]],

               [[ 0, -1],
                [ 0,  1],
                [ 0,  2]]], dtype=int8)
        >>> C2d['genotype'][3, :]
        array([[0, 0],
               [0, 1],
               [0, 0]], dtype=int8)

    """

    loader = _CalldataLoader(vcf_fn, region=region, samples=samples,
                             ploidy=ploidy, fields=fields,
                             exclude_fields=exclude_fields, dtypes=dtypes,
                             arities=arities, fills=fills, vcf_types=vcf_types,
                             count=count, progress=progress,
                             logstream=logstream, condition=condition,
                             slice_args=slice_args, verbose=verbose,
                             cache=cache, cachedir=cachedir,
                             skip_cached=skip_cached)
    arr = loader.load()
    return arr


class _Calldata2DLoader(_CalldataLoader):

    array_type = 'calldata_2d'

    def build(self):
        arr = super().build()
        return view2d(arr)


def calldata_2d(vcf_fn, **kwargs):
    """
    Load a numpy 2-dimensional structured array with data from the sample
    columns of a VCF file. Convenience function, equivalent to calldata()
    followed by view2d().

    Parameters
    ----------

    vcf_fn: string or list
        Name of the VCF file or list of file names.
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'.
    fields: list or array-like
        List of fields to extract from the VCF.
    exclude_fields: list or array-like
        Fields to exclude from extraction.
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred ones.
    arities: dict or dict-like
        Override the amount of values to expect
    fills: dict or dict-like
        Dictionary containing field:fillvalue mappings used to override the
        default fill in values in VCF fields.
    vcf_types: dict or dict-like
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header.
    count: int
        Attempt to extract a specific number of records.
    progress: int
        If greater than 0, log parsing progress.
    logstream: file or file-like object
        Stream to use for logging progress.
    condition: array
        Boolean array defining which rows to load.
    slice_args: tuple or list
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every
        10th row from the first 1000.
    verbose: bool
        Log more messages.
    cache: bool
        If True, save the resulting numpy array to disk, and load from the
        cache if present rather than rebuilding from the VCF.
    cachedir: string
        Manually specify the directory to use to store cache files.
    skip_cached: bool
        If True and cache file is fresh, do not load and return None.

    """

    loader = _Calldata2DLoader(vcf_fn, **kwargs)
    arr = loader.load()
    return arr


def _itercalldata(vcf_fns, region, tuple samples, int ploidy, list fieldspec):
    cdef VariantCallFile *variant_file
    cdef Variant *variant

    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        variant_file.parseInfo = False
        variant_file.parseSamples = True
        if region is not None:
            region_set = variant_file.setRegion(region)
            if not region_set:
                raise StopIteration
        variant = new Variant(deref(variant_file))

        while _get_next_variant(variant_file, variant):
            yield _mkcrow(variant, samples, ploidy, fieldspec)

        del variant_file
        del variant


def _itercalldata_with_condition(vcf_fns, region, tuple samples, int ploidy,
                                 list fieldspec, condition):
    cdef VariantCallFile *variant_file
    cdef Variant *variant
    cdef long i = 0
    cdef long n = len(condition)

    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        variant_file.parseInfo = False
        variant_file.parseSamples = False
        if region is not None:
            region_set = variant_file.setRegion(region)
            if not region_set:
                raise StopIteration
        variant = new Variant(deref(variant_file))

        while i < n:
            # only worth parsing samples if we know we want the variant
            if condition[i]:
                variant_file.parseSamples = True
                if not _get_next_variant(variant_file, variant):
                    break
                yield _mkcrow(variant, samples, ploidy, fieldspec)
            else:
                variant_file.parseSamples = False
                if not _get_next_variant(variant_file, variant):
                    break
            i += 1

        del variant_file
        del variant


cdef inline object _mkcrow(Variant *variant,
                             tuple samples,
                             int ploidy,
                             list fieldspec):
    out = [_mkcvals(variant, s, ploidy, fieldspec) for s in samples]
    return tuple(out)


cdef inline object _mkcvals(Variant *variant,
                            string sample,
                            int ploidy,
                            list fieldspec):
    out = [_mkcval(variant.samples[sample], ploidy, f, arity, fill, format_type)
           for (f, arity, fill, format_type) in fieldspec]
    return tuple(out)


cdef inline object _mkcval(map[string, vector[string]]& sample_data,
                           int ploidy,
                           string field,
                           int arity,
                           object fill,
                           int format_type):
    if field == FIELD_NAME_IS_CALLED:
        return _is_called(sample_data)
    elif field == FIELD_NAME_IS_PHASED:
        return _is_phased(sample_data)
    elif field == FIELD_NAME_GENOTYPE:
        return _genotype(sample_data, ploidy)
    else:
        return _mkval(sample_data[field], arity, fill, format_type)


cdef inline bool _is_called(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return gts.at(0).find('.') == npos


cdef inline bool _is_phased(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return gts.at(0).find('|') != npos


cdef inline object _genotype(map[string, vector[string]]& sample_data,
                             int ploidy):
    cdef vector[string] *gts
    cdef vector[int] alleles
    cdef vector[string] allele_strings
    cdef int i
    cdef int allele
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        if ploidy == 1:
            return -1
        else:
            return (-1,) * ploidy
    else:
        split(gts.at(0), GT_DELIMS, allele_strings)
        if ploidy == 1:
            if allele_strings.size() > 0:
                if allele_strings.at(0) == DOT:
                    return -1
                else:
                    return atoi(allele_strings.at(0).c_str())
            else:
                return -1
        else:
            for i in range(ploidy):
                if i < allele_strings.size():
                    if allele_strings.at(i) == DOT:
                        alleles.push_back(-1)
                    else:
                        alleles.push_back(atoi(allele_strings.at(i).c_str()))
                else:
                    alleles.push_back(-1)
            return tuple(alleles)


def view2d(a):
    """
    Utility function to view a structured 1D array where all fields have a
    uniform dtype (e.g., an array constructed by :func:calldata) as a 2D array.

    Parameters
    ----------

    a: numpy array or array-like
        The array to be viewed as 2D, must have a uniform dtype

    Returns
    -------

    A 2D view of the array.

    Examples
    --------

        >>> from vcfnp import calldata
        >>> a = calldata('sample.vcf')
        >>> a
        array([ ((True, True, [0, 0], '0|0', 0, 0, [10, 10]), (True, True, [0, 0], '0|0', 0, 0, [10, 10]), (True, False, [0, 1], '0/1', 0, 0, [3, 3])),
               ((True, True, [0, 0], '0|0', 0, 0, [10, 10]), (True, True, [0, 0], '0|0', 0, 0, [10, 10]), (True, False, [0, 1], '0/1', 0, 0, [3, 3])),
               ((True, True, [0, 0], '0|0', 48, 1, [51, 51]), (True, True, [1, 0], '1|0', 48, 8, [51, 51]), (True, False, [1, 1], '1/1', 43, 5, [0, 0])),
               ((True, True, [0, 0], '0|0', 49, 3, [58, 50]), (True, True, [0, 1], '0|1', 3, 5, [65, 3]), (True, False, [0, 0], '0/0', 41, 3, [0, 0])),
               ((True, True, [1, 2], '1|2', 21, 6, [23, 27]), (True, True, [2, 1], '2|1', 2, 0, [18, 2]), (True, False, [2, 2], '2/2', 35, 4, [0, 0])),
               ((True, True, [0, 0], '0|0', 54, 0, [56, 60]), (True, True, [0, 0], '0|0', 48, 4, [51, 51]), (True, False, [0, 0], '0/0', 61, 2, [0, 0])),
               ((True, False, [0, 1], '0/1', 0, 4, [0, 0]), (True, False, [0, 2], '0/2', 17, 2, [0, 0]), (True, False, [1, 1], '1/1', 40, 3, [0, 0])),
               ((True, False, [0, 0], '0/0', 0, 0, [0, 0]), (True, True, [0, 0], '0|0', 0, 0, [0, 0]), (False, False, [-1, -1], './.', 0, 0, [0, 0])),
               ((True, False, [0, -1], '0', 0, 0, [0, 0]), (True, False, [0, 1], '0/1', 0, 0, [0, 0]), (True, True, [0, 2], '0|2', 0, 0, [0, 0]))],
              dtype=[('NA00001', [('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))]), ('NA00002', [('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))]), ('NA00003', [('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))])])
        >>> from vcfnp import view2d
        >>> b = view2d(a)
        >>> b
        array([[(True, True, [0, 0], '0|0', 0, 0, [10, 10]),
                (True, True, [0, 0], '0|0', 0, 0, [10, 10]),
                (True, False, [0, 1], '0/1', 0, 0, [3, 3])],
               [(True, True, [0, 0], '0|0', 0, 0, [10, 10]),
                (True, True, [0, 0], '0|0', 0, 0, [10, 10]),
                (True, False, [0, 1], '0/1', 0, 0, [3, 3])],
               [(True, True, [0, 0], '0|0', 48, 1, [51, 51]),
                (True, True, [1, 0], '1|0', 48, 8, [51, 51]),
                (True, False, [1, 1], '1/1', 43, 5, [0, 0])],
               [(True, True, [0, 0], '0|0', 49, 3, [58, 50]),
                (True, True, [0, 1], '0|1', 3, 5, [65, 3]),
                (True, False, [0, 0], '0/0', 41, 3, [0, 0])],
               [(True, True, [1, 2], '1|2', 21, 6, [23, 27]),
                (True, True, [2, 1], '2|1', 2, 0, [18, 2]),
                (True, False, [2, 2], '2/2', 35, 4, [0, 0])],
               [(True, True, [0, 0], '0|0', 54, 0, [56, 60]),
                (True, True, [0, 0], '0|0', 48, 4, [51, 51]),
                (True, False, [0, 0], '0/0', 61, 2, [0, 0])],
               [(True, False, [0, 1], '0/1', 0, 4, [0, 0]),
                (True, False, [0, 2], '0/2', 17, 2, [0, 0]),
                (True, False, [1, 1], '1/1', 40, 3, [0, 0])],
               [(True, False, [0, 0], '0/0', 0, 0, [0, 0]),
                (True, True, [0, 0], '0|0', 0, 0, [0, 0]),
                (False, False, [-1, -1], './.', 0, 0, [0, 0])],
               [(True, False, [0, -1], '0', 0, 0, [0, 0]),
                (True, False, [0, 1], '0/1', 0, 0, [0, 0]),
                (True, True, [0, 2], '0|2', 0, 0, [0, 0])]],
              dtype=[('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))])
        >>> b['GT']
        array([['0|0', '0|0', '0/1'],
               ['0|0', '0|0', '0/1'],
               ['0|0', '1|0', '1/1'],
               ['0|0', '0|1', '0/0'],
               ['1|2', '2|1', '2/2'],
               ['0|0', '0|0', '0/0'],
               ['0/1', '0/2', '1/1'],
               ['0/0', '0|0', './.'],
               ['0', '0/1', '0|2']],
              dtype='|S3')

    """

    rows = a.size
    cols = len(a.dtype)
    dtype = a.dtype[0]
    b = a.view(dtype).reshape(rows, cols)
    return b


EFF_DEFAULT_DTYPE = [
    ('Effect', 'a33'),
    ('Effect_Impact', 'a8'),
    ('Functional_Class', 'a8'),
    ('Codon_Change', 'a7'), # N.B., will lose information for indels
    ('Amino_Acid_Change', 'a6'), # N.B., will lose information for indels
    ('Amino_Acid_Length', 'i4'),
    ('Gene_Name', 'a14'), # N.B., may be too short for some species
    ('Transcript_BioType', 'a20'),
    ('Gene_Coding', 'i1'),
    ('Transcript_ID', 'a20'),
    ('Exon', 'i1')
]


DEFAULT_INFO_DTYPE['EFF'] = EFF_DEFAULT_DTYPE
DEFAULT_VARIANT_ARITY['EFF'] = 1


EFF_DEFAULT_FILLS = ('.', '.', '.', '.', '.', -1, '.', '.', -1, '.', -1)


def eff_default_transformer(fills=EFF_DEFAULT_FILLS):
    """
    Return a simple transformer function for parsing EFF annotations. N.B.,
    ignores all but the first effect.

    """
    def _transformer(vals):
        if len(vals) == 0:
            return fills
        else:
            # ignore all but first effect
            match_eff_main = _prog_eff_main.match(vals[0])
            eff = [match_eff_main.group(1)] + match_eff_main.group(2).split('|')
            result = tuple(
                fill if v == ''
                else int(v) if i == 5 or i == 10
                else (1 if v == 'CODING' else 0) if i == 8
                else v
                for i, (v, fill) in enumerate(zip(eff, fills)[:11])
            )
            return result
    return _transformer


DEFAULT_TRANSFORMER['EFF'] = eff_default_transformer()


class VariantsTable(object):

    def __init__(self,
                 filename,
                 region=None,
                 fields=None,
                 exclude_fields=None,
                 arities=None,
                 flatten_filter=False,
                 fill='.',
                 flatten=None,
                 ):
        self.filename = filename
        self.region = region
        self.fields = fields
        self.exclude_fields = exclude_fields
        self.arities = arities
        self.flatten_filter = flatten_filter
        self.fill = fill
        self.flatten = flatten

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
        parse_info = any([f not in STANDARD_VARIANT_FIELDS for f in fields])

        # convert to tuple
        info_types = tuple(info_types[f] if f in info_types else -1
                           for f in fields)

        # default flattening
        if self.flatten is None:
            self.flatten = dict()
        for f in DEFAULT_FLATTEN:
            if f not in self.flatten:
                ff, t = DEFAULT_FLATTEN[f]
                self.flatten[f] = ff, t(self.fill)

        # make header row
        header = list()
        for f in fields:
            if self.flatten_filter and f == 'FILTER':
                for t in filter_ids:
                    header.append('FILTER_' + t)
            elif (self.arities is not None
                  and f in self.arities
                  and self.arities[f] > 1):
                for i in range(1, self.arities[f] + 1):
                    header.append(f + '_' + str(i))
            elif f in self.flatten and self.flatten[f] is not None:
                fflds, _ = self.flatten[f]
                header.extend(fflds)
            else:
                header.append(f)
        header = tuple(header)
        # make data rows
        data = _itervariantstable(vcf_fns=vcf_fns, region=self.region,
                                 fields=fields, arities=arities,
                                 info_types=info_types, parse_info=parse_info,
                                 filter_ids=filter_ids,
                                 flatten_filter=self.flatten_filter,
                                 fill=self.fill, flatten=self.flatten)
        return chain((header,), data)


def _itervariantstable(vcf_fns, region, fields, arities, info_types, parse_info,
                       filter_ids, flatten_filter, fill, flatten):
    cdef VariantCallFile *variant_file
    cdef Variant *variant

    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        variant_file.parseInfo = parse_info
        variant_file.parseSamples = False
        if region is not None:
            region_set = variant_file.setRegion(region)
            if not region_set:
                raise StopIteration
        variant = new Variant(deref(variant_file))

        while _get_next_variant(variant_file, variant):
            yield _mkvtblrow(variant, fields, arities, info_types, filter_ids,
                             flatten_filter, fill, flatten)

        del variant_file
        del variant


cdef inline object _mkvtblrow(Variant *variant, fields, arities, info_types,
                              filter_ids, flatten_filter, fill, flatten):
    out = list()
    cdef string field
    for field, arity, vcf_type in zip(fields, arities, info_types):
        if field == FIELD_NAME_CHROM:
            out.append(variant.sequenceName)
        elif field == FIELD_NAME_POS:
            out.append(variant.position)
        elif field == FIELD_NAME_ID:
            out.append(variant.id)
        elif field == FIELD_NAME_REF:
            out.append(variant.ref)
        elif field == FIELD_NAME_ALT:
            if arity is not None:
                vals = _mktblval_multi(variant.alt, arity, fill)
                out.extend(vals)
            elif variant.alt.size() == 0:
                out.append(fill)
            else:
                val = ','.join(variant.alt)
                out.append(val)
        elif field == FIELD_NAME_QUAL:
            out.append(variant.quality)
        elif field == FIELD_NAME_FILTER:
            if flatten_filter:
                out.extend(_mkfilterval(variant, filter_ids))
            elif variant.filter == DOT:
                out.append(fill)
            else:
                out.append(variant.filter)
        elif field == FIELD_NAME_NUM_ALLELES:
            out.append(variant.alt.size() + 1)
        elif field == FIELD_NAME_IS_SNP:
            out.append(_is_snp(variant))
        else:
            if vcf_type == FIELD_BOOL:
                # ignore arity, this is a flag
                val = (variant.infoFlags.count(field) > 0)
                out.append(val)
            else:
                if arity is not None:
                    vals = _mktblval_multi(variant.info[field], arity, fill)
                    out.extend(vals)
                elif str(field) in flatten and flatten[str(field)] is not None:
                    _, t = flatten[str(field)]
                    vals = t(variant.info[field])
                    out.extend(vals)
                elif variant.info[field].size() == 0:
                    out.append(fill)
                else:
                    out.append(','.join(variant.info[field]))
    return tuple(out)


cdef inline object _mktblval_multi(vector[string]& string_vals, int arity,
                                   object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(string_vals.at(i))
        else:
            out.append(fill)
    return out


EFF_FIELDS = (
    'Effect',
    'Effect_Impact',
    'Functional_Class',
    'Codon_Change',
    'Amino_Acid_Change',
    'Amino_Acid_Length',
    'Gene_Name',
    'Transcript_BioType',
    'Gene_Coding',
    'Transcript_ID',
    'Exon',
)


_prog_eff_main = re.compile(r'([^(]+)\(([^)]+)\)')


def flatten_eff(fill='.'):
    def _flatten(vals):
        if len(vals) == 0:
            return [fill] * 11
        else:
            # ignore all but first effect
            match_eff_main = _prog_eff_main.match(vals[0])
            eff = [match_eff_main.group(1)] + match_eff_main.group(2).split('|')
            eff = [fill if v == '' else v for v in eff[:11]]
            return eff
    return _flatten


DEFAULT_FLATTEN = {
    'EFF': (EFF_FIELDS, flatten_eff),
}
