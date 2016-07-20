# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# python standard library dependencies
import logging
import inspect
import sys
import os
import time
from datetime import datetime
from itertools import islice


# third party dependencies
import numpy as np


# internal dependencies
from vcfnp.vcflib import PyVariantCallFile, TYPE_STRING, NUMBER_ALLELE, \
    NUMBER_GENOTYPE
from vcfnp.compat import string_types
from vcfnp.iter import itervariants, itercalldata
import vcfnp.config as config


logger = logging.getLogger(__name__)


def debug(msg):
    logger.debug('%s: %s' % (inspect.stack()[1][3], msg))


def variants(vcf_fn, region=None, fields=None, exclude_fields=None,
             dtypes=None, arities=None, fills=None, transformers=None,
             vcf_types=None, count=None, progress=0, logstream=None,
             condition=None, slice_args=None, flatten_filter=False,
             verbose=True, cache=False, cachedir=None, skip_cached=False,
             compress_cache=False, truncate=True):
    """
    Load an numpy structured array with data from the fixed fields of a VCF
    file (including INFO).

    Parameters
    ----------

    vcf_fn: string or list
        Name of the VCF file or list of file names.
    region: string, optional
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'.
    fields: list or array-like, optional
        List of fields to extract from the VCF.
    exclude_fields: list or array-like, optional
        Fields to exclude from extraction.
    dtypes: dict or dict-like, optional
        Dictionary cotaining dtypes to use instead of the default inferred
        ones.
    arities: dict or dict-like, optional
        Dictionary containing field:integer mappings used to override the
        number of values to expect.
    fills: dict or dict-like, optional
        Dictionary containing field:fillvalue mappings used to override the
        defaults used for missing values.
    transformers: dict or dict-like, optional
        Dictionary containing field:function mappings used to preprocess
        any values prior to loading into array.
    vcf_types: dict or dict-like, optional
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header (e.g., MQ0Fraction declared
        as Integer).
    count: int, optional
        Attempt to extract a specific number of records.
    progress: int, optional
        If greater than 0, log progress.
    logstream: file or file-like object, optional
        Stream to use for logging progress.
    condition: array, optional
        Boolean array defining which rows to load.
    slice_args: tuple or list, optional
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every
        10th row from the first 1000.
    flatten_filter: bool, optional
        Return FILTER as multiple boolean fields, e.g., FILTER_PASS,
        FILTER_LowQuality, etc.
    verbose: bool, optional
        Log more messages.
    cache: bool, optional
        If True, save the resulting numpy array to disk, and load from the
        cache if present rather than rebuilding from the VCF.
    cachedir: string, optional
        Manually specify the directory to use to store cache files.
    skip_cached: bool, optional
        If True and cache file is fresh, do not load and return None.
    compress_cache: bool, optional
        If True, compress the cache file.
    truncate: bool, optional
        If True (default) only include variants whose start position is within
        the given region. If False, use default tabix behaviour.

    Examples
    --------

    >>> from vcfnp import variants
    >>> v = variants('fixture/sample.vcf')
    >>> v
    array([ (b'19', 111, b'.', b'A', b'C', 9.600000381469727, (False, False, False), 2, True, 0, b'', 0, 0.0, 0, False, 0, False, 0),
           (b'19', 112, b'.', b'A', b'G', 10.0, (False, False, False), 2, True, 0, b'', 0, 0.0, 0, False, 0, False, 0),
           (b'20', 14370, b'rs6054257', b'G', b'A', 29.0, (False, False, True), 2, True, 0, b'', 0, 0.5, 0, True, 14, True, 3),
           (b'20', 17330, b'.', b'T', b'A', 3.0, (True, False, False), 2, True, 0, b'', 0, 0.016998291015625, 0, False, 11, False, 3),
           (b'20', 1110696, b'rs6040355', b'A', b'G', 67.0, (False, False, True), 3, True, 0, b'T', 0, 0.3330078125, 0, True, 10, False, 2),
           (b'20', 1230237, b'.', b'T', b'.', 47.0, (False, False, True), 2, False, 0, b'T', 0, 0.0, 0, False, 13, False, 3),
           (b'20', 1234567, b'microsat1', b'G', b'GA', 50.0, (False, False, True), 3, False, 1, b'G', 3, 0.0, 6, False, 9, False, 3),
           (b'20', 1235237, b'.', b'T', b'.', 0.0, (False, False, False), 2, False, 0, b'', 0, 0.0, 0, False, 0, False, 0),
           (b'X', 10, b'rsTest', b'AC', b'A', 10.0, (False, False, True), 3, False, -1, b'', 0, 0.0, 0, False, 0, False, 0)],
          dtype=[('CHROM', 'S12'), ('POS', '<i4'), ('ID', 'S12'), ('REF', 'S12'), ('ALT', 'S12'), ('QUAL', '<f4'), ('FILTER', [('q10', '?'), ('s50', '?'), ('PASS', '?')]), ('num_alleles', 'u1'), ('is_snp', '?'), ('svlen', '<i4'), ('AA', 'S12'), ('AC', '<u2'), ('AF', '<f2'), ('AN', '<u2'), ('DB', '?'), ('DP', '<i4'), ('H2', '?'), ('NS', '<i4')])
    >>> v['QUAL']
    array([  9.60000038,  10.        ,  29.        ,   3.        ,
            67.        ,  47.        ,  50.        ,   0.        ,  10.        ], dtype=float32)
    >>> v['FILTER']['PASS']
    array([False, False,  True, False,  True,  True,  True, False,  True], dtype=bool)
    >>> v['AF']
    array([ 0.        ,  0.        ,  0.5       ,  0.01699829,  0.33300781,
            0.        ,  0.        ,  0.        ,  0.        ], dtype=float16)

    """  # flake8: noqa

    loader = _VariantsLoader(vcf_fn, region=region, fields=fields,
                             exclude_fields=exclude_fields, dtypes=dtypes,
                             arities=arities, fills=fills,
                             transformers=transformers, vcf_types=vcf_types,
                             count=count, progress=progress,
                             logstream=logstream, condition=condition,
                             slice_args=slice_args,
                             flatten_filter=flatten_filter, verbose=verbose,
                             cache=cache, cachedir=cachedir,
                             skip_cached=skip_cached,
                             compress_cache=compress_cache, truncate=truncate)
    return loader.load()


class _ArrayLoader(object):
    """Abstract class providing support for loading an array optionally via a
    cache layer."""

    # to be overridden in subclass
    array_type = None

    def __init__(self, vcf_fn, logstream=None, verbose=True, **kwargs):
        debug('init')
        self.vcf_fn = vcf_fn
        # deal with polymorphic vcf_fn argument
        self.vcf_fns = _filenames_from_arg(vcf_fn)
        self.log = _get_logger(logstream, verbose)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load(self):
        log = self.log
        array_type = self.array_type
        vcf_fn = self.vcf_fn
        region = self.region
        cache = self.cache
        cachedir = self.cachedir
        skip_cached = self.skip_cached
        compress_cache = self.compress_cache

        if cache:
            log('caching is enabled')
            cache_fn, is_cached = _get_cache(vcf_fn, array_type=array_type,
                                             region=region, cachedir=cachedir,
                                             compress=compress_cache, log=log)
            if not is_cached:
                log('building array')
                arr = self.build()
                log('saving to cache file', cache_fn)
                if compress_cache:
                    np.savez_compressed(cache_fn, data=arr)
                else:
                    np.save(cache_fn, arr)

            elif skip_cached:
                log('skipping load from cache file', cache_fn)
                arr = None

            else:
                log('loading from cache file', cache_fn)
                arr = np.load(cache_fn)
                if compress_cache:
                    arr = arr['data']

        else:
            log('caching is disabled')
            log('building array')
            arr = self.build()

        return arr

    # to be overridden in subclass
    def build(self):
        pass


def _filenames_from_arg(filename):
    """Utility function to deal with polymorphic filenames argument."""
    if isinstance(filename, string_types):
        filenames = [filename]
    elif isinstance(filename, (list, tuple)):
        filenames = filename
    else:
        raise Exception('filename argument must be string, list or tuple')
    for fn in filenames:
        if not os.path.exists(fn):
            raise ValueError('file not found: %s' % fn)
        if not os.path.isfile(fn):
            raise ValueError('not a file: %s' % fn)
    return filenames


# TODO replace this with use of Python standard libary logging support


class _Logger(object):

    def __init__(self, logstream=None):
        if logstream is None:
            logstream = sys.stderr
        self.logstream = logstream

    def __call__(self, *msg):
        s = ('[vcfnp] '
             + str(datetime.now())
             + ' :: '
             + ' '.join([str(m) for m in msg]))
        print(s, file=self.logstream)
        self.logstream.flush()


def _nolog(*args, **kwargs):
    pass


def _get_logger(logstream, verbose):
    if verbose:
        log = _Logger(logstream)
    else:
        log = _nolog
    return log


def _mk_cache_fn(vcf_fn, array_type, region=None, cachedir=None,
                 compress=False):
    """Utility function to construct a filename for a cache file, given a VCF
    file name (where the original data came from) and other parameters."""

    # ensure cache dir exists
    if cachedir is None:
        # use the VCF file name as the base for a directory name
        cachedir = vcf_fn + config.CACHEDIR_SUFFIX
    if not os.path.exists(cachedir):
        # ensure cache dir exists
        os.makedirs(cachedir)
    else:
        assert os.path.isdir(cachedir), \
            'unexpected error, cache directory is not a directory: %r' \
            % cachedir

    # handle compression
    if compress:
        suffix = 'npz'
    else:
        suffix = 'npy'

    # handle region
    if region is None:
        # loading the whole genome (i.e., all variants)
        cache_fn = os.path.join(cachedir, '%s.%s' % (array_type, suffix))
    else:
        # loading a specific region
        region = region.replace(':', '__').replace('-', '_')
        cache_fn = os.path.join(cachedir, '%s.%s.%s' % (array_type, region,
                                                        suffix))
    return cache_fn


def _get_cache(vcf_fn, array_type, region, cachedir, compress, log):
    """Utility function to obtain a cache file name and determine whether or
    not a fresh cache file is available."""

    # guard condition
    if isinstance(vcf_fn, (list, tuple)):
        raise Exception(
            'caching only supported when loading from a single VCF file'
        )

    # create cache file name
    cache_fn = _mk_cache_fn(vcf_fn, array_type=array_type, region=region,
                            cachedir=cachedir, compress=compress)

    # decide whether or not a fresh cache file is available
    # (if not, we will parse the VCF and build array from scratch)
    if not os.path.exists(cache_fn):
        log('no cache file found')
        is_cached = False
    elif os.path.getmtime(vcf_fn) > os.path.getmtime(cache_fn):
        is_cached = False
        log('cache file out of date')
    else:
        is_cached = True
        log('cache file available')

    return cache_fn, is_cached


class _VariantsLoader(_ArrayLoader):
    """Class for building variants array."""

    array_type = 'variants'

    def build(self):
        log = self.log

        # open VCF file to inspect header
        vcf_fns = self.vcf_fns
        vcf = PyVariantCallFile(vcf_fns[0])

        # extract FILTER definitions
        filter_ids = vcf.filter_ids
        _warn_duplicates(filter_ids)
        filter_ids = sorted(set(filter_ids))
        if 'PASS' not in filter_ids:
            filter_ids.append('PASS')
        filter_ids = tuple(filter_ids)

        # extract INFO definitions
        _warn_duplicates(vcf.info_ids)
        info_ids = tuple(sorted(set(vcf.info_ids)))
        info_types = vcf.info_types
        info_counts = vcf.info_counts

        # determine which fields to load
        fields = _variants_fields(self.fields, self.exclude_fields, info_ids)

        # determine whether we need to parse the INFO field at all
        parse_info = any([f not in config.STANDARD_VARIANT_FIELDS
                          for f in fields])

        # support for working around VCFs with bad INFO headers
        vcf_types = self.vcf_types
        for f in fields:
            if f not in config.STANDARD_VARIANT_FIELDS and f not in info_ids:
                # fall back to unary string; can be overridden with
                # vcf_types, dtypes and arities args
                info_types[f] = TYPE_STRING
                info_counts[f] = 1
            if vcf_types is not None and f in vcf_types:
                # override type declared in VCF header
                info_types[f] = config.TYPESTRING2KEY[vcf_types[f]]

        # convert to tuples for convenience
        info_types = tuple(info_types[f] if f in info_types else -1
                           for f in fields)
        info_counts = tuple(info_counts[f] if f in info_counts else -1
                            for f in fields)

        # determine expected number of values for each field
        arities = _variants_arities(fields, self.arities, info_counts)

        # determine fill values to use where number of values is less than
        # expectation
        fills = _variants_fills(fields, self.fills, info_types)

        # initialise INFO field transformers
        transformers = _info_transformers(fields, self.transformers)

        # determine dtype to use
        flatten_filter = self.flatten_filter
        dtype = _variants_dtype(fields, self.dtypes, arities, filter_ids,
                                flatten_filter, info_types)

        # set up iterator
        region = self.region
        truncate = self.truncate
        condition = self.condition
        if condition is not None:
            condition = np.asarray(condition).astype('uint8')
        it = itervariants(vcf_fns, region=region, fields=fields,
                          arities=arities, fills=fills,
                          info_types=info_types, transformers=transformers,
                          filter_ids=filter_ids, flatten_filter=flatten_filter,
                          parse_info=parse_info, condition=condition,
                          truncate=truncate)

        # slice iterator
        slice_args = self.slice_args
        if slice_args:
            it = islice(it, *slice_args)

        # load array
        arr = _fromiter(it, dtype, self.count, self.progress, log)

        return arr


def _warn_duplicates(fields):
    visited = set()
    for f in fields:
        if f in visited:
            print('WARNING: duplicate definition in header: %s' % f,
                  file=sys.stderr)
        visited.add(f)


def _variants_fields(fields, exclude_fields, info_ids):
    """Utility function to determine which fields to extract when loading
    variants."""
    if fields is None:
        # no fields specified by user
        # by default extract all standard and INFO fields
        fields = config.STANDARD_VARIANT_FIELDS + info_ids
    else:
        # fields have been specified
        for f in fields:
            # check for non-standard fields not declared in INFO header
            if f not in config.STANDARD_VARIANT_FIELDS and f not in info_ids:
                # support extracting INFO even if not declared in header,
                # but warn...
                print('WARNING: no INFO definition found for field %s' % f,
                      file=sys.stderr)
    # process any exclusions
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]
    return tuple(f for f in fields)


def _variants_arities(fields, arities, info_counts):
    """Utility function to determine arities (i.e., number of values to
    expect) for variants fields."""
    if arities is None:
        # no arities specified by user
        arities = dict()
    for f, vcf_count in zip(fields, info_counts):
        if f == 'FILTER':
            arities[f] = 1  # force one value for the FILTER field
        elif f not in arities:
            # arity not specified by user
            if f in config.STANDARD_VARIANT_FIELDS:
                arities[f] = config.DEFAULT_VARIANT_ARITY[f]
            elif vcf_count == NUMBER_ALLELE:
                # default to 1 (biallelic)
                arities[f] = 1
            elif vcf_count <= 0:
                # catch any other cases of non-specific arity
                arities[f] = 1
            else:
                # use arity (i.e., number) specified in INFO header
                arities[f] = vcf_count
    # convert to tuple for zipping with fields
    arities = tuple(arities[f] for f in fields)
    return arities


def _variants_fills(fields, fills, info_types):
    """Utility function to determine fill values for variants fields with
    missing values."""
    if fills is None:
        # no fills specified by user
        fills = dict()
    for f, vcf_type in zip(fields, info_types):
        if f == 'FILTER':
            fills[f] = False
        elif f not in fills:
            if f in config.STANDARD_VARIANT_FIELDS:
                fills[f] = config.DEFAULT_VARIANT_FILL[f]
            else:
                fills[f] = config.DEFAULT_FILL_MAP[vcf_type]
    # convert to tuple for zipping with fields
    fills = tuple(fills[f] for f in fields)
    return fills


def _info_transformers(fields, transformers):
    """Utility function to determine transformer functions for variants
    fields."""
    if transformers is None:
        # no transformers specified by user
        transformers = dict()
    for f in fields:
        if f not in transformers:
            transformers[f] = config.DEFAULT_TRANSFORMER.get(f, None)
    return tuple(transformers[f] for f in fields)


def _variants_dtype(fields, dtypes, arities, filter_ids, flatten_filter,
                    info_types):
    """Utility function to build a numpy dtype for a variants array,
    given user arguments and information available from VCF header."""
    dtype = list()
    for f, n, vcf_type in zip(fields, arities, info_types):
        if f == 'FILTER' and flatten_filter:
            # split FILTER into multiple boolean fields
            for flt in filter_ids:
                nm = 'FILTER_' + flt
                dtype.append((nm, 'b1'))
        elif f == 'FILTER' and not flatten_filter:
            # represent FILTER as a structured field
            t = [(flt, 'b1') for flt in filter_ids]
            dtype.append((f, t))
        else:
            if dtypes is not None and f in dtypes:
                # user overrides default dtype
                t = dtypes[f]
            elif f in config.STANDARD_VARIANT_FIELDS:
                t = config.DEFAULT_VARIANT_DTYPE[f]
            elif f in config.DEFAULT_INFO_DTYPE:
                # known INFO field
                t = config.DEFAULT_INFO_DTYPE[f]
            else:
                t = config.DEFAULT_TYPE_MAP[vcf_type]
            # deal with arity
            if n == 1:
                dtype.append((f, t))
            else:
                dtype.append((f, t, (n,)))
    return dtype


def _fromiter(it, dtype, count, progress, log):
    """Utility function to load an array from an iterator."""
    if progress > 0:
        it = _iter_withprogress(it, progress, log)
    if count is not None:
        a = np.fromiter(it, dtype=dtype, count=count)
    else:
        a = np.fromiter(it, dtype=dtype)
    return a


def _iter_withprogress(iterable, progress, log):
    """Utility function to load an array from an iterator, reporting progress
    as we go."""
    before_all = time.time()
    before = before_all
    n = 0
    for i, o in enumerate(iterable):
        yield o
        n = i+1
        if n % progress == 0:
            after = time.time()
            log('%s rows in %.2fs; batch in %.2fs (%d rows/s)'
                % (n, after-before_all, after-before, progress/(after-before)))
            before = after
    after_all = time.time()
    log('%s rows in %.2fs (%d rows/s)'
        % (n, after_all-before_all, n/(after_all-before_all)))


def calldata(vcf_fn, region=None, samples=None, ploidy=2, fields=None,
             exclude_fields=None, dtypes=None, arities=None, fills=None,
             vcf_types=None, count=None, progress=0, logstream=None,
             condition=None, slice_args=None, verbose=True, cache=False,
             cachedir=None, skip_cached=False, compress_cache=False,
             truncate=True):
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
    compress_cache: bool, optional
        If True, compress the cache file.
    truncate: bool, optional
        If True (default) only include variants whose start position is within
        the given region. If False, use default tabix behaviour.

    Examples
    --------

    >>> from vcfnp import calldata, view2d
    >>> c = calldata('fixture/sample.vcf')
    >>> c
    array([ ((True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, False, [0, 1], 0, 0, b'0/1', [3, 3])),
           ((True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, False, [0, 1], 0, 0, b'0/1', [3, 3])),
           ((True, True, [0, 0], 1, 48, b'0|0', [51, 51]), (True, True, [1, 0], 8, 48, b'1|0', [51, 51]), (True, False, [1, 1], 5, 43, b'1/1', [0, 0])),
           ((True, True, [0, 0], 3, 49, b'0|0', [58, 50]), (True, True, [0, 1], 5, 3, b'0|1', [65, 3]), (True, False, [0, 0], 3, 41, b'0/0', [0, 0])),
           ((True, True, [1, 2], 6, 21, b'1|2', [23, 27]), (True, True, [2, 1], 0, 2, b'2|1', [18, 2]), (True, False, [2, 2], 4, 35, b'2/2', [0, 0])),
           ((True, True, [0, 0], 0, 54, b'0|0', [56, 60]), (True, True, [0, 0], 4, 48, b'0|0', [51, 51]), (True, False, [0, 0], 2, 61, b'0/0', [0, 0])),
           ((True, False, [0, 1], 4, 0, b'0/1', [0, 0]), (True, False, [0, 2], 2, 17, b'0/2', [0, 0]), (False, False, [-1, -1], 3, 40, b'./.', [0, 0])),
           ((True, False, [0, 0], 0, 0, b'0/0', [0, 0]), (True, True, [0, 0], 0, 0, b'0|0', [0, 0]), (False, False, [-1, -1], 0, 0, b'./.', [0, 0])),
           ((True, False, [0, -1], 0, 0, b'0', [0, 0]), (True, False, [0, 1], 0, 0, b'0/1', [0, 0]), (True, True, [0, 2], 0, 0, b'0|2', [0, 0]))],
          dtype=[('NA00001', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00002', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00003', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])])
    >>> c['NA00001']
    array([(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
           (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
           (True, True, [0, 0], 1, 48, b'0|0', [51, 51]),
           (True, True, [0, 0], 3, 49, b'0|0', [58, 50]),
           (True, True, [1, 2], 6, 21, b'1|2', [23, 27]),
           (True, True, [0, 0], 0, 54, b'0|0', [56, 60]),
           (True, False, [0, 1], 4, 0, b'0/1', [0, 0]),
           (True, False, [0, 0], 0, 0, b'0/0', [0, 0]),
           (True, False, [0, -1], 0, 0, b'0', [0, 0])],
          dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
    >>> c2d = view2d(c)
    >>> c2d
    array([[(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 1, 48, b'0|0', [51, 51]),
            (True, True, [1, 0], 8, 48, b'1|0', [51, 51]),
            (True, False, [1, 1], 5, 43, b'1/1', [0, 0])],
           [(True, True, [0, 0], 3, 49, b'0|0', [58, 50]),
            (True, True, [0, 1], 5, 3, b'0|1', [65, 3]),
            (True, False, [0, 0], 3, 41, b'0/0', [0, 0])],
           [(True, True, [1, 2], 6, 21, b'1|2', [23, 27]),
            (True, True, [2, 1], 0, 2, b'2|1', [18, 2]),
            (True, False, [2, 2], 4, 35, b'2/2', [0, 0])],
           [(True, True, [0, 0], 0, 54, b'0|0', [56, 60]),
            (True, True, [0, 0], 4, 48, b'0|0', [51, 51]),
            (True, False, [0, 0], 2, 61, b'0/0', [0, 0])],
           [(True, False, [0, 1], 4, 0, b'0/1', [0, 0]),
            (True, False, [0, 2], 2, 17, b'0/2', [0, 0]),
            (False, False, [-1, -1], 3, 40, b'./.', [0, 0])],
           [(True, False, [0, 0], 0, 0, b'0/0', [0, 0]),
            (True, True, [0, 0], 0, 0, b'0|0', [0, 0]),
            (False, False, [-1, -1], 0, 0, b'./.', [0, 0])],
           [(True, False, [0, -1], 0, 0, b'0', [0, 0]),
            (True, False, [0, 1], 0, 0, b'0/1', [0, 0]),
            (True, True, [0, 2], 0, 0, b'0|2', [0, 0])]],
          dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
    >>> c2d['genotype']
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
    >>> c2d['genotype'][3, :]
    array([[0, 0],
           [0, 1],
           [0, 0]], dtype=int8)

    """  # flake8: noqa

    loader = _CalldataLoader(vcf_fn, region=region, samples=samples,
                             ploidy=ploidy, fields=fields,
                             exclude_fields=exclude_fields, dtypes=dtypes,
                             arities=arities, fills=fills, vcf_types=vcf_types,
                             count=count, progress=progress,
                             logstream=logstream, condition=condition,
                             slice_args=slice_args, verbose=verbose,
                             cache=cache, cachedir=cachedir,
                             skip_cached=skip_cached,
                             compress_cache=compress_cache,
                             truncate=truncate)
    arr = loader.load()
    return arr


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
        all_samples = vcf.sample_names

        # determine which samples to extract
        samples = self.samples
        if samples is None:
            samples = all_samples
        else:
            # guard against unknown samples requested by user
            for s in samples:
                assert s in all_samples, 'unknown sample: %s' % s
        samples = tuple(samples)
        debug(samples)

        # determine which fields to extract
        fields = _calldata_fields(self.fields, self.exclude_fields, format_ids)

        # support for working around VCFs with bad FORMAT headers
        vcf_types = self.vcf_types
        for f in fields:
            if (f not in config.STANDARD_CALLDATA_FIELDS and
                    f not in format_ids):
                # fall back to unary string; can be overridden with
                # vcf_types, dtypes and arities args
                format_types[f] = TYPE_STRING
                format_counts[f] = 1
            if vcf_types is not None and f in vcf_types:
                # override type declared in VCF header
                format_types[f] = config.TYPESTRING2KEY[vcf_types[f]]

        # conveniences
        format_types = tuple(format_types[f] if f in format_types else -1
                             for f in fields)
        format_counts = tuple(format_counts[f] if f in format_counts else -1
                              for f in fields)

        # determine expected number of values for each field
        ploidy = self.ploidy
        arities = _calldata_arities(fields, self.arities, format_counts,
                                    ploidy)

        # determine fill values to use where number of values is less than
        # expectation
        fills = _calldata_fills(fields, self.fills, format_types, ploidy)

        # construct a numpy dtype for structured array
        dtype = _calldata_dtype(fields, self.dtypes, format_types, arities,
                                samples, ploidy)

        # set up iterator
        condition = self.condition
        if condition is not None:
            condition = np.asarray(condition).astype('uint8')
        region = self.region
        truncate = self.truncate
        it = itercalldata(vcf_fns=vcf_fns, region=region, samples=samples,
                          ploidy=ploidy, fields=fields, arities=arities,
                          fills=fills, format_types=format_types,
                          condition=condition, truncate=truncate)

        # slice iterator
        slice_args = self.slice_args
        if slice_args:
            it = islice(it, *slice_args)

        # build an array from the iterator
        arr = _fromiter(it, dtype, self.count, self.progress, log)

        return arr


def calldata_2d(vcf_fn, region=None, samples=None, ploidy=2, fields=None,
                exclude_fields=None, dtypes=None, arities=None, fills=None,
                vcf_types=None, count=None, progress=0, logstream=None,
                condition=None, slice_args=None, verbose=True, cache=False,
                cachedir=None, skip_cached=False, compress_cache=False,
                truncate=True):
    """
    Load a numpy 2-dimensional structured array with data from the sample
    columns of a VCF file. Convenience function, equivalent to calldata()
    followed by view2d(), except that if caching is enabled, files will be
    cached as 2D.

    Parameters
    ----------
    vcf_fn: string or list
        Name of the VCF file or list of file names.
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'.
    samples: sequence of strings
        Samples to load.
    ploidy: int
        Sample ploidy.
    fields: list or array-like
        List of fields to extract from the VCF.
    exclude_fields: list or array-like
        Fields to exclude from extraction.
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred
        ones.
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
    compress_cache: bool, optional
        If True, compress the cache file.
    truncate: bool, optional
        If True (default) only include variants whose start position is within
        the given region. If False, use default tabix behaviour.

    Examples
    --------

    >>> from vcfnp import calldata_2d
    >>> c2d = calldata_2d('fixture/sample.vcf')
    >>> c2d
    array([[(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 1, 48, b'0|0', [51, 51]),
            (True, True, [1, 0], 8, 48, b'1|0', [51, 51]),
            (True, False, [1, 1], 5, 43, b'1/1', [0, 0])],
           [(True, True, [0, 0], 3, 49, b'0|0', [58, 50]),
            (True, True, [0, 1], 5, 3, b'0|1', [65, 3]),
            (True, False, [0, 0], 3, 41, b'0/0', [0, 0])],
           [(True, True, [1, 2], 6, 21, b'1|2', [23, 27]),
            (True, True, [2, 1], 0, 2, b'2|1', [18, 2]),
            (True, False, [2, 2], 4, 35, b'2/2', [0, 0])],
           [(True, True, [0, 0], 0, 54, b'0|0', [56, 60]),
            (True, True, [0, 0], 4, 48, b'0|0', [51, 51]),
            (True, False, [0, 0], 2, 61, b'0/0', [0, 0])],
           [(True, False, [0, 1], 4, 0, b'0/1', [0, 0]),
            (True, False, [0, 2], 2, 17, b'0/2', [0, 0]),
            (False, False, [-1, -1], 3, 40, b'./.', [0, 0])],
           [(True, False, [0, 0], 0, 0, b'0/0', [0, 0]),
            (True, True, [0, 0], 0, 0, b'0|0', [0, 0]),
            (False, False, [-1, -1], 0, 0, b'./.', [0, 0])],
           [(True, False, [0, -1], 0, 0, b'0', [0, 0]),
            (True, False, [0, 1], 0, 0, b'0/1', [0, 0]),
            (True, True, [0, 2], 0, 0, b'0|2', [0, 0])]],
          dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
    >>> c2d['GT']
    array([[b'0|0', b'0|0', b'0/1'],
           [b'0|0', b'0|0', b'0/1'],
           [b'0|0', b'1|0', b'1/1'],
           [b'0|0', b'0|1', b'0/0'],
           [b'1|2', b'2|1', b'2/2'],
           [b'0|0', b'0|0', b'0/0'],
           [b'0/1', b'0/2', b'./.'],
           [b'0/0', b'0|0', b'./.'],
           [b'0', b'0/1', b'0|2']],
          dtype='|S3')
    >>> c2d['genotype']
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
    >>> c2d['genotype'][3, :]
    array([[0, 0],
           [0, 1],
           [0, 0]], dtype=int8)

    """  # flake8: noqa

    loader = _Calldata2DLoader(vcf_fn, region=region, samples=samples,
                               ploidy=ploidy, fields=fields,
                               exclude_fields=exclude_fields, dtypes=dtypes,
                               arities=arities, fills=fills,
                               vcf_types=vcf_types, count=count,
                               progress=progress, logstream=logstream,
                               condition=condition, slice_args=slice_args,
                               verbose=verbose, cache=cache, cachedir=cachedir,
                               skip_cached=skip_cached,
                               compress_cache=compress_cache,
                               truncate=truncate)
    arr = loader.load()
    return arr


class _Calldata2DLoader(_CalldataLoader):

    array_type = 'calldata_2d'

    def build(self):
        arr = super(_Calldata2DLoader, self).build()
        return view2d(arr)


def _calldata_fields(fields, exclude_fields, format_ids):
    """Utility function to determine which calldata (i.e., FORMAT) fields to
    extract."""
    if fields is None:
        # no fields specified by user
        # default to all standard fields plus all FORMAT fields in VCF header
        fields = config.STANDARD_CALLDATA_FIELDS + format_ids
    else:
        # fields specified by user
        for f in fields:
            # check if field is standard or defined in VCF header
            if (f not in config.STANDARD_CALLDATA_FIELDS and
                    f not in format_ids):
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
            elif f in config.DEFAULT_CALLDATA_ARITY:
                arities[f] = config.DEFAULT_CALLDATA_ARITY[f]
            elif vcf_count == NUMBER_ALLELE:
                # default to 2 (biallelic)
                arities[f] = 2
            elif vcf_count == NUMBER_GENOTYPE:
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
            elif f in config.DEFAULT_CALLDATA_FILL:
                fills[f] = config.DEFAULT_CALLDATA_FILL[f]
            else:
                fills[f] = config.DEFAULT_FILL_MAP[vcf_type]
    return tuple(fills[f] for f in fields)


def _calldata_dtype(fields, dtypes, format_types, arities, samples, ploidy):

    # construct a numpy dtype for structured array cells
    cell_dtype = list()
    for f, vcf_type, n in zip(fields, format_types, arities):
        if dtypes is not None and f in dtypes:
            t = dtypes[f]
        elif f == 'GT':
            t = 'a%d' % ((ploidy*2)-1)
        elif f in config.DEFAULT_CALLDATA_DTYPE:
            # known field
            t = config.DEFAULT_CALLDATA_DTYPE[f]
        else:
            t = config.DEFAULT_TYPE_MAP[vcf_type]
        if n == 1:
            cell_dtype.append((f, t))
        else:
            cell_dtype.append((f, t, (n,)))

    # construct a numpy dtype for structured array
    dtype = [(s, cell_dtype) for s in samples]
    return dtype


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
    >>> a = calldata('fixture/sample.vcf')
    >>> a
    array([ ((True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, False, [0, 1], 0, 0, b'0/1', [3, 3])),
           ((True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, True, [0, 0], 0, 0, b'0|0', [10, 10]), (True, False, [0, 1], 0, 0, b'0/1', [3, 3])),
           ((True, True, [0, 0], 1, 48, b'0|0', [51, 51]), (True, True, [1, 0], 8, 48, b'1|0', [51, 51]), (True, False, [1, 1], 5, 43, b'1/1', [0, 0])),
           ((True, True, [0, 0], 3, 49, b'0|0', [58, 50]), (True, True, [0, 1], 5, 3, b'0|1', [65, 3]), (True, False, [0, 0], 3, 41, b'0/0', [0, 0])),
           ((True, True, [1, 2], 6, 21, b'1|2', [23, 27]), (True, True, [2, 1], 0, 2, b'2|1', [18, 2]), (True, False, [2, 2], 4, 35, b'2/2', [0, 0])),
           ((True, True, [0, 0], 0, 54, b'0|0', [56, 60]), (True, True, [0, 0], 4, 48, b'0|0', [51, 51]), (True, False, [0, 0], 2, 61, b'0/0', [0, 0])),
           ((True, False, [0, 1], 4, 0, b'0/1', [0, 0]), (True, False, [0, 2], 2, 17, b'0/2', [0, 0]), (False, False, [-1, -1], 3, 40, b'./.', [0, 0])),
           ((True, False, [0, 0], 0, 0, b'0/0', [0, 0]), (True, True, [0, 0], 0, 0, b'0|0', [0, 0]), (False, False, [-1, -1], 0, 0, b'./.', [0, 0])),
           ((True, False, [0, -1], 0, 0, b'0', [0, 0]), (True, False, [0, 1], 0, 0, b'0/1', [0, 0]), (True, True, [0, 2], 0, 0, b'0|2', [0, 0]))],
          dtype=[('NA00001', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00002', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))]), ('NA00003', [('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])])
    >>> from vcfnp import view2d
    >>> b = view2d(a)
    >>> b
    array([[(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, True, [0, 0], 0, 0, b'0|0', [10, 10]),
            (True, False, [0, 1], 0, 0, b'0/1', [3, 3])],
           [(True, True, [0, 0], 1, 48, b'0|0', [51, 51]),
            (True, True, [1, 0], 8, 48, b'1|0', [51, 51]),
            (True, False, [1, 1], 5, 43, b'1/1', [0, 0])],
           [(True, True, [0, 0], 3, 49, b'0|0', [58, 50]),
            (True, True, [0, 1], 5, 3, b'0|1', [65, 3]),
            (True, False, [0, 0], 3, 41, b'0/0', [0, 0])],
           [(True, True, [1, 2], 6, 21, b'1|2', [23, 27]),
            (True, True, [2, 1], 0, 2, b'2|1', [18, 2]),
            (True, False, [2, 2], 4, 35, b'2/2', [0, 0])],
           [(True, True, [0, 0], 0, 54, b'0|0', [56, 60]),
            (True, True, [0, 0], 4, 48, b'0|0', [51, 51]),
            (True, False, [0, 0], 2, 61, b'0/0', [0, 0])],
           [(True, False, [0, 1], 4, 0, b'0/1', [0, 0]),
            (True, False, [0, 2], 2, 17, b'0/2', [0, 0]),
            (False, False, [-1, -1], 3, 40, b'./.', [0, 0])],
           [(True, False, [0, 0], 0, 0, b'0/0', [0, 0]),
            (True, True, [0, 0], 0, 0, b'0|0', [0, 0]),
            (False, False, [-1, -1], 0, 0, b'./.', [0, 0])],
           [(True, False, [0, -1], 0, 0, b'0', [0, 0]),
            (True, False, [0, 1], 0, 0, b'0/1', [0, 0]),
            (True, True, [0, 2], 0, 0, b'0|2', [0, 0])]],
          dtype=[('is_called', '?'), ('is_phased', '?'), ('genotype', 'i1', (2,)), ('DP', '<u2'), ('GQ', 'u1'), ('GT', 'S3'), ('HQ', '<i4', (2,))])
    >>> b['GT']
    array([[b'0|0', b'0|0', b'0/1'],
           [b'0|0', b'0|0', b'0/1'],
           [b'0|0', b'1|0', b'1/1'],
           [b'0|0', b'0|1', b'0/0'],
           [b'1|2', b'2|1', b'2/2'],
           [b'0|0', b'0|0', b'0/0'],
           [b'0/1', b'0/2', b'./.'],
           [b'0/0', b'0|0', b'./.'],
           [b'0', b'0/1', b'0|2']],
          dtype='|S3')
    >>> b['genotype']
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

    """  # flake8: noqa

    rows = a.size
    cols = len(a.dtype)
    dtype = a.dtype[0]
    b = a.view(dtype).reshape(rows, cols)
    return b
