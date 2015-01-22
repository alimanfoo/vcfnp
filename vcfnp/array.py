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
from vcfnp.vcflib import PyVariantCallFile, FIELD_BOOL, FIELD_FLOAT, \
    FIELD_INTEGER, FIELD_STRING, FIELD_UNKNOWN, ALLELE_NUMBER
from vcfnp.compat import string_types
from vcfnp.array_ext import itervariants


logger = logging.getLogger(__name__)
debug = lambda msg: logger.debug('%s: %s' % (inspect.stack()[0][3], msg))


# default configuration
CACHEDIR_SUFFIX = '.vcfnp_cache'

# these are the standard fields in the variants array
STANDARD_VARIANT_FIELDS = (
    'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
    'num_alleles', 'is_snp', 'svlen'
)


TYPESTRING2KEY = {
    'Float': FIELD_FLOAT,
    'Integer': FIELD_INTEGER,
    'String': FIELD_STRING,
    'Flag': FIELD_BOOL,
}


# default dtypes for the variants array fields
DEFAULT_VARIANT_DTYPE = {
    'CHROM': 'a12',
    'POS': 'i4',
    'ID': 'a12',
    'REF': 'a12',
    'ALT': 'a12',
    'QUAL': 'f4',
    'num_alleles': 'u1',
    'is_snp': 'b1',
    'svlen': 'i4',
}


# default arities for the variants array fields
DEFAULT_VARIANT_ARITY = {
    'CHROM': 1,
    'POS': 1,
    'ID': 1,
    'REF': 1,
    'ALT': 1,  # default assume biallelic (1 alt allele)
    'QUAL': 1,
    'num_alleles': 1,
    'is_snp': 1,
    'svlen': 1,  # default assume biallelic
}


# default fill values for the variants fields if values are missing
DEFAULT_VARIANT_FILL = {
    'CHROM': '',
    'POS': 0,
    'ID': '',
    'REF': '',
    'ALT': '',
    'QUAL': 0,
    'num_alleles': 0,
    'is_snp': False,
    'svlen': 0,
}


# default mapping from VCF field types to numpy dtypes
DEFAULT_TYPE_MAP = {
    FIELD_FLOAT: 'f4',
    FIELD_INTEGER: 'i4',
    FIELD_STRING: 'a12',
    FIELD_BOOL: 'b1',
    FIELD_UNKNOWN: 'a12'  # leave as string
}


# default mapping from VCF field types to fill values for missing values
DEFAULT_FILL_MAP = {
    FIELD_FLOAT: 0.,
    FIELD_INTEGER: 0,
    FIELD_STRING: '',
    FIELD_BOOL: False,
    FIELD_UNKNOWN: ''
}


# default dtypes for some known INFO fields where lower precision is
# acceptable in most cases
DEFAULT_INFO_DTYPE = {
    'ABHet': 'f2',
    'ABHom': 'f2',
    'AC': 'u2',
    'AF': 'f2',
    'AN': 'u2',
    'BaseQRankSum': 'f2',
    'ClippingRankSum': 'f2',
    'Dels': 'f2',
    'FS': 'f2',
    'HRun': 'u1',
    'HaplotypeScore': 'f2',
    'InbreedingCoeff': 'f2',
    'VariantType': 'a12',
    'MLEAC': 'u2',
    'MLEAF': 'f2',
    'MQ': 'f2',
    'MQ0Fraction': 'f2',
    'MQRankSum': 'f2',
    'OND': 'f2',
    'QD': 'f2',
    'RPA': 'u2',
    'RU': 'a12',
    'ReadPosRankSum': 'f2',
}


DEFAULT_TRANSFORMER = dict()


STANDARD_CALLDATA_FIELDS = ('is_called', 'is_phased', 'genotype')


DEFAULT_CALLDATA_DTYPE = {
    'is_called': 'b1',
    'is_phased': 'b1',
    'genotype': 'i1',
    # set some lower precision defaults for known FORMAT fields
    'AD': 'u2',
    'DP': 'u2',
    'GQ': 'u1',
    'MLPSAC': 'u1',
    'MLPSAF': 'f2',
    'MQ0': 'u2',
    'PL': 'u2',
}


DEFAULT_CALLDATA_FILL = {
    'is_called': False,
    'is_phased': False,
    'genotype': -1,
}


DEFAULT_CALLDATA_ARITY = {
    'is_called': 1,
    'is_phased': 1,
    # N.B., set genotype arity to ploidy
    'AD': 2,  # default to biallelic
}


def variants(vcf_fn, region=None, fields=None, exclude_fields=None, dtypes=None,
             arities=None, fills=None, transformers=None, vcf_types=None,
             count=None, progress=0, logstream=None, condition=None,
             slice_args=None, flatten_filter=False, verbose=True, cache=True,
             cachedir=None, skip_cached=False):
    """
    Load an numpy structured array with data from the fixed fields of a VCF file
    (including INFO).

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
        Dictionary cotaining dtypes to use instead of the default inferred ones.
    arities: dict or dict-like, optional
        Dictinoary containing field:integer mappings used to override the number
        of values to expect.
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

    Examples
    --------

        >>> from vcfnp import variants
        >>> v = variants('fixture/sample.vcf')
        >>> v
        array([ ('19', 111, '.', 'A', 'C', 9.600000381469727, (False, False, False), 2, True, 0, '.', 0, 0.0, 0, False, 0, False, 0),
               ('19', 112, '.', 'A', 'G', 10.0, (False, False, False), 2, True, 0, '.', 0, 0.0, 0, False, 0, False, 0),
               ('20', 14370, 'rs6054257', 'G', 'A', 29.0, (False, False, True), 2, True, 0, '.', 0, 0.5, 0, True, 14, True, 3),
               ('20', 17330, '.', 'T', 'A', 3.0, (True, False, False), 2, True, 0, '.', 0, 0.017000000923871994, 0, False, 11, False, 3),
               ('20', 1110696, 'rs6040355', 'A', 'G', 67.0, (False, False, True), 3, True, 0, 'T', 0, 0.3330000042915344, 0, True, 10, False, 2),
               ('20', 1230237, '.', 'T', '.', 47.0, (False, False, True), 2, False, 0, 'T', 0, 0.0, 0, False, 13, False, 3),
               ('20', 1234567, 'microsat1', 'G', 'GA', 50.0, (False, False, True), 3, False, 1, 'G', 3, 0.0, 6, False, 9, False, 3),
               ('20', 1235237, '.', 'T', '.', 0.0, (False, False, False), 2, False, 0, '.', 0, 0.0, 0, False, 0, False, 0),
               ('X', 10, 'rsTest', 'AC', 'A', 10.0, (False, False, True), 3, False, -1, '.', 0, 0.0, 0, False, 0, False, 0)],
              dtype=[('CHROM', 'S12'), ('POS', '<i4'), ('ID', 'S12'), ('REF', 'S12'), ('ALT', 'S12'), ('QUAL', '<f4'), ('FILTER', [('q10', '?'), ('s50', '?'), ('PASS', '?')]), ('num_alleles', 'u1'), ('is_snp', '?'), ('svlen', '<i4'), ('AA', 'S12'), ('AC', '<u2'), ('AF', '<f4'), ('AN', '<u2'), ('DB', '?'), ('DP', '<i4'), ('H2', '?'), ('NS', '<i4')])
        >>> v['QUAL']
        array([  9.60000038,  10.        ,  29.        ,   3.        ,
                67.        ,  47.        ,  50.        ,   0.        ,  10.        ], dtype=float32)
        >>> v['FILTER']['PASS']
        array([False, False,  True, False,  True,  True,  True, False,  True], dtype=bool)
        >>> v['AF']
        array([ 0.   ,  0.   ,  0.5  ,  0.017,  0.333,  0.   ,  0.   ,  0.   ,  0.   ], dtype=float32)

    """

    loader = _VariantsLoader(vcf_fn, region=region, fields=fields,
                             exclude_fields=exclude_fields, dtypes=dtypes,
                             arities=arities, fills=fills,
                             transformers=transformers, vcf_types=vcf_types,
                             count=count, progress=progress,
                             logstream=logstream, condition=condition,
                             slice_args=slice_args,
                             flatten_filter=flatten_filter, verbose=verbose,
                             cache=cache, cachedir=cachedir,
                             skip_cached=skip_cached)
    return loader.load()


class _ArrayLoader(object):
    """Abstract class providing support for loading an array optionally via a
    cache layer."""

    # to be overridden in subclass
    array_type = None

    def __init__(self, vcf_fn, logstream, verbose, **kwargs):
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
        if cache:
            log('caching is enabled')
            cache_fn, is_cached = _get_cache(vcf_fn, array_type=array_type,
                                             region=region, cachedir=cachedir,
                                             log=log)
            if not is_cached:
                log('building array')
                arr = self.build()
                log('saving to cache file', cache_fn)
                np.save(cache_fn, arr)
            elif skip_cached:
                log('skipping load from cache file', cache_fn)
                arr = None
            else:
                log('loading from cache file', cache_fn)
                arr = np.load(cache_fn)

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
            raise Exception('file not found: %s' % fn)
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


def _mk_cache_fn(vcf_fn, array_type, region=None, cachedir=None):
    """Utility function to construct a filename for a cache file, given a VCF
    file name (where the original data came from) and other parameters."""
    if cachedir is None:
        # use the VCF file name as the base for a directory name
        cachedir = vcf_fn + CACHEDIR_SUFFIX
    if not os.path.exists(cachedir):
        # ensure cache dir exists
        os.makedirs(cachedir)
    else:
        assert os.path.isdir(cachedir), \
            'unexpected error, cache directory is not a directory: %r' \
            % cachedir
    if region is None:
        # loading the whole genome (i.e., all variants)
        cache_fn = os.path.join(cachedir, '%s.npy' % array_type)
    else:
        # loading a specific region
        region = region.replace(':', '_').replace('-', '_')
        cache_fn = os.path.join(cachedir, '%s.%s.npy' % (array_type, region))
    return cache_fn


def _get_cache(vcf_fn, array_type, region, cachedir, log):
    """Utility function to obtain a cache file name and determine whether or
    not a fresh cache file is available."""

    # guard condition
    if isinstance(vcf_fn, (list, tuple)):
        raise Exception(
            'caching only supported when loading from a single VCF file'
        )

    # create cache file name
    cache_fn = _mk_cache_fn(vcf_fn, array_type=array_type, region=region,
                            cachedir=cachedir)

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
        parse_info = any([f not in STANDARD_VARIANT_FIELDS for f in fields])

        # support for working around VCFs with bad INFO headers
        vcf_types = self.vcf_types
        for f in fields:
            if f not in STANDARD_VARIANT_FIELDS and f not in info_ids:
                # fall back to unary string; can be overridden with
                # vcf_types, dtypes and arities args
                info_types[f] = FIELD_STRING
                info_counts[f] = 1
            if vcf_types is not None and f in vcf_types:
                # override type declared in VCF header
                info_types[f] = TYPESTRING2KEY[vcf_types[f]]

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
        condition = self.condition
        it = itervariants(vcf_fns, region=region, fields=fields,
                          arities=arities, fills=fills,
                          info_types=info_types, transformers=transformers,
                          filter_ids=filter_ids, flatten_filter=flatten_filter,
                          parse_info=parse_info, condition=condition)

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
        fields = STANDARD_VARIANT_FIELDS + info_ids
    else:
        # fields have been specified
        for f in fields:
            # check for non-standard fields not declared in INFO header
            if f not in STANDARD_VARIANT_FIELDS and f not in info_ids:
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
            if f in STANDARD_VARIANT_FIELDS:
                arities[f] = DEFAULT_VARIANT_ARITY[f]
            elif vcf_count == ALLELE_NUMBER:
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
            if f in STANDARD_VARIANT_FIELDS:
                fills[f] = DEFAULT_VARIANT_FILL[f]
            else:
                fills[f] = DEFAULT_FILL_MAP[vcf_type]
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
            transformers[f] = DEFAULT_TRANSFORMER.get(f, None)
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
            elif f in STANDARD_VARIANT_FIELDS:
                t = DEFAULT_VARIANT_DTYPE[f]
            elif f in DEFAULT_INFO_DTYPE:
                # known INFO field
                t = DEFAULT_INFO_DTYPE[f]
            else:
                t = DEFAULT_TYPE_MAP[vcf_type]
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
