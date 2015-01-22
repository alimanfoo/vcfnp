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
from vcflib cimport (PyVariantCallFile, VariantCallFile, Variant,
                     VariantFieldType, FIELD_FLOAT, FIELD_INTEGER,
                     FIELD_STRING, FIELD_BOOL, FIELD_UNKNOWN,
                     ALLELE_NUMBER, GENOTYPE_NUMBER)
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport atoi, atol, atof
from cython.operator cimport dereference as deref
import time
from itertools import islice
import os
from datetime import datetime
import logging


logger = logging.getLogger(__name__)
import inspect
debug = lambda msg: logger.debug('%s: %s' % (inspect.stack()[0][3], msg))


# PY2/3 compatibility
PY2 = sys.version_info[0] == 2
if PY2:
    text_type = unicode
    binary_type = str
    string_types = basestring,
else:
    text_type = str
    binary_type = bytes
    string_types = str,


cdef size_t npos = -1


cdef extern from "split.h":
    # split a string on a single delimiter character (delim)
    vector[string]& split(const string &s, char delim, vector[string] &elems)
    vector[string]  split(const string &s, char delim)
    # split a string on any character found in the string of delimiters (delims)
    vector[string]& split(const string &s, const string& delims, vector[string] &elems)
    vector[string]  split(const string &s, const string& delims)


TYPESTRING2KEY = {
    'Float': FIELD_FLOAT,
    'Integer': FIELD_INTEGER,
    'String': FIELD_STRING,
    'Flag': FIELD_BOOL,
}

# these are the standard fields in the variants array
STANDARD_VARIANT_FIELDS = (
    'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
    'num_alleles', 'is_snp', 'svlen'
)


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
    FIELD_UNKNOWN: 'a12' # leave as string
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


cdef char SEMICOLON = b';'
cdef string DOT = b'.'
cdef string GT_DELIMS = b'/|'
cdef string FIELD_NAME_CHROM = b'CHROM'
cdef string FIELD_NAME_POS = b'POS'
cdef string FIELD_NAME_ID = b'ID'
cdef string FIELD_NAME_REF = b'REF'
cdef string FIELD_NAME_ALT = b'ALT'
cdef string FIELD_NAME_QUAL = b'QUAL'
cdef string FIELD_NAME_FILTER = b'FILTER'
cdef string FIELD_NAME_INFO = b'INFO'
cdef string FIELD_NAME_NUM_ALLELES = b'num_alleles'
cdef string FIELD_NAME_IS_SNP = b'is_snp'
cdef string FIELD_NAME_SVLEN = b'svlen'
cdef string FIELD_NAME_IS_CALLED = b'is_called'
cdef string FIELD_NAME_IS_PHASED = b'is_phased'
cdef string FIELD_NAME_GENOTYPE = b'genotype'
cdef string FIELD_NAME_GT = b'GT'


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
    return tuple(fields)


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


CACHEDIR_SUFFIX = '.vcfnp_cache'


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


def _get_cache_fn(vcf_fn, array_type, region, cachedir, log):
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
        debug(log)
        array_type = self.array_type
        vcf_fn = self.vcf_fn
        region = self.region
        cache = self.cache
        cachedir = self.cachedir
        skip_cached = self.skip_cached
        if cache:
            log('caching is enabled')
            cache_fn, is_cached = _get_cache_fn(vcf_fn, array_type=array_type,
                                                region=region,
                                                cachedir=cachedir, log=log)
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

        # zip up field information for convenience
        fieldspec = list(zip(fields, arities, fills, info_types, transformers))

        # determine dtype to use
        flatten_filter = self.flatten_filter
        dtype = _variants_dtype(fields, self.dtypes, arities, filter_ids,
                                flatten_filter, info_types)

        # set up iterator
        region = self.region
        condition = self.condition
        if condition is not None:
            it = _itervariants_with_condition(vcf_fns, region, fieldspec,
                                              filter_ids, flatten_filter,
                                              parse_info, condition)
        else:
            it = _itervariants(vcf_fns, region, fieldspec, filter_ids,
                               flatten_filter, parse_info)

        # slice iterator
        slice_args = self.slice_args
        if slice_args:
            it = islice(it, *slice_args)

        # load array
        arr = _fromiter(it, dtype, self.count, self.progress, log)

        return arr


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
        Name of the VCF file or list of file names
    region: string, optional
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'
    fields: list or array-like, optional
        List of fields to extract from the VCF
    exclude_fields: list or array-like, optional
        Fields to exclude from extraction
    dtypes: dict or dict-like, optional
        Dictionary cotaining dtypes to use instead of the default inferred ones
    arities: dict or dict-like, optional
        Dictinoary containing field:integer mappings used to override the number
        of values to expect
    fills: dict or dict-like, optional
        Dictionary containing field:fillvalue mappings used to override the
        defaults used for missing values
    transformers: dict or dict-like, optional
        Dictionary containing field:function mappings used to preprocess
        any values prior to loading into array
    vcf_types: dict or dict-like, optional
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header (e.g., MQ0Fraction declared
        as Integer)
    count: int, optional
        Attempt to extract a specific number of records
    progress: int, optional
        If greater than 0, log progress
    logstream: file or file-like object, optional
        Stream to use for logging progress
    condition: array, optional
        Boolean array defining which rows to load
    slice_args: tuple or list, optional
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every
        10th row from the first 1000
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
        >>> V = variants('fixture/sample.vcf')
        >>> V
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
        >>> V['QUAL']
        array([  9.60000038,  10.        ,  29.        ,   3.        ,
                67.        ,  47.        ,  50.        ,   0.        ,  10.        ], dtype=float32)
        >>> V['FILTER']['PASS']
        array([False, False,  True, False,  True,  True,  True, False,  True], dtype=bool)
        >>> V['AF']
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


def _fromiter(it, dtype, count, long progress, log):
    """Utility function to load an array from an iterator."""
    if progress > 0:
        it = _iter_withprogress(it, progress, log)
    if count is not None:
        a = np.fromiter(it, dtype=dtype, count=count)
    else:
        a = np.fromiter(it, dtype=dtype)
    return a


def _iter_withprogress(iterable, long progress, log):
    """Utility function to load an array from an iterator, reporting progress
    as we go."""
    cdef long i, n
    before_all = time.time()
    before = before_all
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


def _itervariants(vcf_fns, region, list fieldspec, tuple filter_ids,
                  bint flatten_filter, bint parse_info):
    """Iterate over variants from a VCF file, and generate a tuple for each
    variant suitable for loading into a numpy array."""

    # statically typed variables
    cdef VariantCallFile *variant_file
    cdef Variant *variant

    # work through multiple VCFs if provided
    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        # set whether INFO field needs to be parsed
        variant_file.parseInfo = parse_info
        # set whether samples fields need to be parsed
        variant_file.parseSamples = False
        if region is not None:
            # set genome region to extract variants from
            region_set = variant_file.setRegion(region)
            if not region_set:
                raise StopIteration
        variant = new Variant(deref(variant_file))

        # iterate over variants
        while _get_next_variant(variant_file, variant):
            yield _mkvrow(variant, fieldspec, filter_ids, flatten_filter)

        # clean up
        del variant_file
        del variant


def _itervariants_with_condition(vcf_fns, region, list fieldspec,
                                 tuple filter_ids, bint flatten_filter,
                                 parse_info, condition):
    """Utility function to iterate over variants and generate a tuple for each
    variant suitable for loading into a numpy array, yielding only those
    variants for which the corresponding item in condition is True."""

    # statically typed variables
    cdef VariantCallFile *variant_file
    cdef Variant *variant
    cdef long i = 0
    cdef long n = len(condition)

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

        while i < n and _get_next_variant(variant_file, variant):
            if condition[i]:
                yield _mkvrow(variant, fieldspec, filter_ids, flatten_filter)
            i += 1

        del variant_file
        del variant


cdef inline bool _get_next_variant(VariantCallFile *variant_file,
                                   Variant *variant):
    # break this out into a separate function so we can profile it
    return variant_file.getNextVariant(deref(variant))


cdef inline object _mkvrow(Variant *variant, list fieldspec, tuple filter_ids,
                           bint flatten_filter):
    """Make a row of variant data."""
    out = list()
    for f, arity, fill, vcf_type, transformer in fieldspec:
        val = _mkvval(variant, f, arity, fill, vcf_type, transformer,
                      filter_ids)
        if f == 'FILTER' and flatten_filter:
            out.extend(val)
        else:
            out.append(val)
    return tuple(out)


cdef inline object _mkvval(Variant *variant, string field, int arity,
                           object fill, int vcf_type, transformer,
                           tuple filter_ids):
    if field == FIELD_NAME_CHROM:
        out = variant.sequenceName
    elif field == FIELD_NAME_POS:
        out = variant.position
    elif field == FIELD_NAME_ID:
        out = variant.id
    elif field == FIELD_NAME_REF:
        out = variant.ref
    elif field == FIELD_NAME_ALT:
        out = _mkaltval(variant, arity, fill)
    elif field == FIELD_NAME_QUAL:
        out = variant.quality
    elif field == FIELD_NAME_FILTER:
        out = _mkfilterval(variant, filter_ids)
    elif field == FIELD_NAME_NUM_ALLELES:
        out = <int>(variant.alt.size() + 1)
    elif field == FIELD_NAME_IS_SNP:
        out = _is_snp(variant)
    elif field == FIELD_NAME_SVLEN:
        out = _svlen(variant, arity, fill)
    elif transformer is not None:
        out = transformer(variant.info[field])
    elif vcf_type == FIELD_BOOL:
        # ignore arity, this is a flag
        out = (variant.infoFlags.count(field) > 0)
    else:
        out = _mkval(variant.info[field], arity, fill, vcf_type)
    return out


cdef inline object _mkaltval(Variant *variant, int arity, object fill):
    if arity == 1:
        if variant.alt.size() == 0:
            out = fill
        else:
            out = variant.alt.at(0)
    elif variant.alt.size() == arity:
        out = variant.alt
        out = tuple(out)
    elif variant.alt.size() > arity:
        out = variant.alt
        out = tuple(out[:arity])
    else:
        out = variant.alt
        out += [fill] * (arity-variant.alt.size())
        out = tuple(out)
    return out


cdef inline object _mkfilterval(Variant *variant, tuple filter_ids):
    filters = <list>split(variant.filter, SEMICOLON)
    out = [(f in filters) for f in filter_ids]
    out = tuple(out)
    return out


cdef inline object _is_snp(Variant *variant):
    cdef int i
    cdef bytes alt
    if variant.ref.size() > 1:
        return False
    for i in range(variant.alt.size()):
        alt = variant.alt.at(i)
        if alt not in {'A', 'C', 'G', 'T'}:
            return False
    return True


cdef inline object _svlen(Variant *variant, int arity, object fill):
    if arity == 1:
        return _svlen_single(variant.ref, variant.alt, fill)
    else:
        return _svlen_multi(variant.ref, variant.alt, arity, fill)


cdef inline object _svlen_single(string ref, vector[string]& alt, object fill):
    if alt.size() > 0:
        return <int>(alt.at(0).size() - ref.size())
    return fill


cdef inline object _svlen_multi(string ref, vector[string]& alt, int arity,
                                object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < alt.size():
            out.append(<int>(alt.at(i).size() - ref.size()))
        else:
            out.append(fill)
    return out


def _warn_duplicates(fields):
    visited = set()
    for f in fields:
        if f in visited:
            print('WARNING: duplicate definition in header: %s' % f,
                  file=sys.stderr)
        visited.add(f)


cdef inline object _mkval(vector[string]& string_vals, int arity, object fill,
                          int vcf_type):
    if vcf_type == FIELD_FLOAT:
        out = _mkval_double(string_vals, arity, fill)
    elif vcf_type == FIELD_INTEGER:
        out = _mkval_long(string_vals, arity, fill)
    else:
        # make strings by default
        out = _mkval_string(string_vals, arity, fill)
    return out


cdef inline object _mkval_string(vector[string]& string_vals, int arity,
                                 object fill):
    if arity == 1:
        if string_vals.size() > 0:
            return string_vals.at(0)
        else:
            return fill
    else:
        return _mkval_string_multi(string_vals, arity, fill)


cdef inline object _mkval_string_multi(vector[string]& string_vals, int arity,
                                       object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(string_vals.at(i))
        else:
            out.append(fill)
    return out


cdef inline object _mkval_double(vector[string]& string_vals, int arity,
                                 object fill):
    if arity == 1:
        out = _mkval_double_single(string_vals, fill)
    else:
        out = _mkval_double_multi(string_vals, arity, fill)
    return out


cdef inline object _mkval_double_single(vector[string]& string_vals,
                                        object fill):
    cdef double v
    if string_vals.size() > 0:
        return atof(string_vals.at(0).c_str())
    return fill


cdef inline object _mkval_double_multi(vector[string]& string_vals, int arity,
                                       object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atof(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


cdef inline object _mkval_long(vector[string]& string_vals, int arity,
                               object fill):
    if arity == 1:
        out = _mkval_long_single(string_vals, fill)
    else:
        out = _mkval_long_multi(string_vals, arity, fill)
    return out


cdef inline object _mkval_long_single(vector[string]& string_vals, object fill):
    if string_vals.size() > 0:
        return atol(string_vals.at(0).c_str())
    return fill


cdef inline object _mkval_long_multi(vector[string]& string_vals, int arity,
                                     object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atol(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


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
