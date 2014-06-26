# cython: profile = False
# cython: boundscheck = False
# cython: wraparound = False
# cython: embedsignature = True


"""
Utility functions to extract data from a VCF file and load into a numpy array.

"""


# 1.10.1
# 1.10.2
# 1.11
# 1.11.1
# 1.11.2
__version__ = '1.11.3'


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

# these are the possible fields in the variants array
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


# default dtypes for some known INFO fields where lower precision is acceptable in most cases
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


cdef char SEMICOLON = ';'
cdef string DOT = '.'
cdef string GT_DELIMS = '/|'
cdef string FIELD_NAME_CHROM = 'CHROM'
cdef string FIELD_NAME_POS = 'POS'
cdef string FIELD_NAME_ID = 'ID'
cdef string FIELD_NAME_REF = 'REF'
cdef string FIELD_NAME_ALT = 'ALT'
cdef string FIELD_NAME_QUAL = 'QUAL'
cdef string FIELD_NAME_FILTER = 'FILTER'
cdef string FIELD_NAME_INFO = 'INFO'
cdef string FIELD_NAME_NUM_ALLELES = 'num_alleles'
cdef string FIELD_NAME_IS_SNP = 'is_snp'
cdef string FIELD_NAME_SVLEN = 'svlen'
cdef string FIELD_NAME_IS_CALLED = 'is_called'
cdef string FIELD_NAME_IS_PHASED = 'is_phased'
cdef string FIELD_NAME_GENOTYPE = 'genotype'
cdef string FIELD_NAME_GT = 'GT'


def _variants_fields(fields, exclude_fields, info_ids):
    if fields is None:
        fields = STANDARD_VARIANT_FIELDS + info_ids
    else:
        for f in fields:
            if f not in STANDARD_VARIANT_FIELDS and f not in info_ids:
                # support extracting INFO even if not declared in header, but warn...
                print >>sys.stderr, 'WARNING: no INFO definition found for field %s' % f
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]
    return tuple(fields)


def _variants_arities(fields, arities, info_counts):
    if arities is None:
        arities = dict()
    for f, vcf_count in zip(fields, info_counts):
        if f == 'FILTER':
            arities[f] = 1 # one value
        elif f not in arities:
            if f in STANDARD_VARIANT_FIELDS:
                arities[f] = DEFAULT_VARIANT_ARITY[f]
            elif vcf_count == ALLELE_NUMBER:
                # default to 1 (biallelic)
                arities[f] = 1
            elif vcf_count <= 0:
                # catch any other cases of non-specific arity
                arities[f] = 1
            else:
                arities[f] = vcf_count
    arities = tuple(arities[f] for f in fields)
    return arities


def _variants_fills(fields, fills, info_types):
    if fills is None:
        fills = dict()
    for f, vcf_type in zip(fields, info_types):
        if f == 'FILTER':
            fills[f] = False
        elif f not in fills:
            if f in STANDARD_VARIANT_FIELDS:
                fills[f] = DEFAULT_VARIANT_FILL[f]
            else:
                fills[f] = DEFAULT_FILL_MAP[vcf_type]
    fills = tuple(fills[f] for f in fields)
    return fills


def _info_transformers(fields, transformers):
    if transformers is None:
        transformers = dict()
    for f in fields:
        if f not in transformers:
            transformers[f] = DEFAULT_TRANSFORMER.get(f, None)
    return tuple(transformers[f] for f in fields)


def _variants_dtype(fields, dtypes, arities, filter_ids, flatten_filter, info_types):

    dtype = list()
    for f, n, vcf_type in zip(fields, arities, info_types):
        if f == 'FILTER' and flatten_filter:
            # represent FILTER as multiple boolean fields
            for flt in filter_ids:
                nm = 'FILTER_' + flt
                dtype.append((nm, 'b1'))
        elif f == 'FILTER' and not flatten_filter:
            # represent FILTER as a structured datatype
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
            if n == 1:
                dtype.append((f, t))
            else:
                dtype.append((f, t, (n,)))

    return dtype


def _filenames_from_arg(filename):
    if isinstance(filename, basestring):
        filenames = [filename]
    elif isinstance(filename, (list, tuple)):
        filenames = filename
    else:
        raise Exception('filename argument must be basestring, list or tuple')
    for fn in filenames:
        if not os.path.exists(fn):
            raise Exception('file not found: %s' % fn)
    return filenames


def _setup_variants(filename,
                    region,
                    fields,
                    exclude_fields,
                    arities,
                    fills,
                    transformers,
                    vcf_types,
                    flatten_filter):

    filenames = _filenames_from_arg(filename)

    # extract definitions from VCF header
    vcf = PyVariantCallFile(filenames[0])
    # FILTER definitions
    filter_ids = vcf.filterIds
    _warn_duplicates(filter_ids)
    filter_ids = sorted(set(filter_ids))
    if 'PASS' not in filter_ids:
        filter_ids.append('PASS')
    filter_ids = tuple(filter_ids)
    # INFO definitions
    _warn_duplicates(vcf.infoIds)
    info_ids = tuple(sorted(set(vcf.infoIds)))
    info_types = vcf.infoTypes
    info_counts = vcf.infoCounts

    # determine fields to extract
    fields = _variants_fields(fields, exclude_fields, info_ids)

    # determine if we need to parse the INFO field
    parse_info = any([f not in STANDARD_VARIANT_FIELDS for f in fields])

    # support for working around VCFs with bad INFO headers
    for f in fields:
        if f not in STANDARD_VARIANT_FIELDS and f not in info_ids:
            # fall back to unary string; can be overridden with vcf_types, dtypes and arities args
            info_types[f] = FIELD_STRING
            info_counts[f] = 1
        if vcf_types is not None and f in vcf_types:
            # override type declared in VCF header
            info_types[f] = TYPESTRING2KEY[vcf_types[f]]

    info_types = tuple(info_types[f] if f in info_types else -1 for f in fields)
    info_counts = tuple(info_counts[f] if f in info_counts else -1 for f in fields)

    # determine expected number of values for each field
    arities = _variants_arities(fields, arities, info_counts)

    # determine fill values to use where number of values is less than expectation
    fills = _variants_fills(fields, fills, info_types)

    # initialise INFO field transformers
    transformers = _info_transformers(fields, transformers)

    return filenames, region, fields, arities, fills, info_types, transformers, parse_info, filter_ids, flatten_filter


def log(logstream, *msg):
    print >>logstream, '[vcfnp] ' + str(datetime.now()) + ' :: ' + ' '.join([str(m) for m in msg])
    sys.stderr.flush()


def variants(filename,
             region=None,
             fields=None,
             exclude_fields=None,
             dtypes=None,
             arities=None,
             fills=None,
             transformers=None,
             vcf_types=None,
             count=None,
             progress=0,
             logstream=sys.stderr,
             condition=None,
             slice=None,
             flatten_filter=False,
             verbose=False,
             cache=False,
             cachedir=None,
             skip_cached=False,
             ):
    """
    Load an numpy structured array with data from the fixed fields of a VCF file
    (including INFO).

    Parameters
    ----------

    filename: string or list
        Name of the VCF file or list of file names
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'
    fields: list or array-like
        List of fields to extract from the VCF
    exclude_fields: list or array-like
        Fields to exclude from extraction
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred ones
    arities: dict or dict-like
        Dictinoary containing field:integer mappings used to override the number
        of values to expect
    fills: dict or dict-like
        Dictionary containing field:fillvalue mappings used to override the
        defaults used for missing values
    transformers: dict or dict-like
        Dictionary containing field:function mappings used to preprocess
        any values prior to loading into array
    vcf_types: dict or dict-like
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header (e.g., MQ0Fraction declared
        as Integer)
    count: int
        Attempt to extract a specific number of records
    progress: int
        If greater than 0, log parsing progress
    logstream: file or file-like object
        Stream to use for logging progress
    condition: array
        Boolean array defining which rows to load
    slice: tuple or list
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every 10th row from the first 1000
    flatten_filter: bool
        Return FILTER as multiple boolean fields, e.g., FILTER_PASS, FILTER_LowQuality, etc.
    verbose: bool
        Log more messages.
    cache: bool
        If True, save the resulting numpy array to disk, and load from the cache if present rather than rebuilding
        from the VCF.
    cachedir: string
        Manually specify the directory to use to store cache files.
    skip_cached: bool
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

    if cache:

        if isinstance(filename, (list, tuple)):
            raise Exception('caching only supported when loading from a single VCF file')

        cache_fn = _mk_cache_fn(filename, array_type='variants', region=region, cachedir=cachedir)
        if not os.path.exists(cache_fn) or os.path.getmtime(filename) > os.path.getmtime(cache_fn):
            if verbose:
                log(logstream, 'no cache file found or cache out of date')
            A = _build_variants(filename,
                                region=region,
                                fields=fields,
                                exclude_fields=exclude_fields,
                                dtypes=dtypes,
                                arities=arities,
                                fills=fills,
                                transformers=transformers,
                                vcf_types=vcf_types,
                                count=count,
                                progress=progress,
                                logstream=logstream,
                                condition=condition,
                                slice=slice,
                                flatten_filter=flatten_filter,
                                verbose=verbose)
            if verbose:
                log(logstream, 'saving to cache', cache_fn)
            np.save(cache_fn, A)
            return A
        else:
            if skip_cached:
                if verbose:
                    log(logstream, 'skipping from cache', cache_fn)
                return None
            else:
                if verbose:
                    log(logstream, 'loading from cache', cache_fn)
                A = np.load(cache_fn)
                return A

    else:

        A = _build_variants(filename,
                            region=region,
                            fields=fields,
                            exclude_fields=exclude_fields,
                            dtypes=dtypes,
                            arities=arities,
                            fills=fills,
                            transformers=transformers,
                            vcf_types=vcf_types,
                            count=count,
                            progress=progress,
                            logstream=logstream,
                            condition=condition,
                            slice=slice,
                            flatten_filter=flatten_filter,
                            verbose=verbose)
        return A


cachedir_suffix = '.vcfnp_cache'


def _mk_cache_fn(vcf_fn, array_type, region=None, cachedir=None):
    if cachedir is None:
        cachedir = vcf_fn + cachedir_suffix
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    else:
        assert os.path.isdir(cachedir), 'unexpected error, cache directory is not a directory: %s' % cachedir
    if region is None:
        cache_fn = os.path.join(cachedir, '%s.npy' % array_type)
    else:
        region = region.replace(':', '_').replace('-', '_')
        cache_fn = os.path.join(cachedir, '%s.%s.npy' % (array_type, region))
    return cache_fn


def _build_variants(filename,
                    region=None,
                    fields=None,
                    exclude_fields=None,
                    dtypes=None,
                    arities=None,
                    fills=None,
                    transformers=None,
                    vcf_types=None,
                    count=None,
                    progress=0,
                    logstream=sys.stderr,
                    condition=None,
                    slice=None,
                    flatten_filter=False,
                    verbose=False,
                    cache=False,
                    ):

    if verbose:
        log(logstream, 'loading variants from', filename)

    filenames, region, fields, arities, fills, infoTypes, transformers, parseInfo, filterIds, flatten_filter = _setup_variants(filename,
                                                                                                                               region,
                                                                                                                               fields,
                                                                                                                               exclude_fields,
                                                                                                                               arities,
                                                                                                                               fills,
                                                                                                                               transformers,
                                                                                                                               vcf_types,
                                                                                                                               flatten_filter)

    # zip up field parameters
    fieldspec = zip(fields, arities, fills, infoTypes, transformers)

    # create a numpy dtype
    dtype = _variants_dtype(fields, dtypes, arities, filterIds, flatten_filter, infoTypes)

    # set up iterator
    if condition is not None:
        it = itervariants_with_condition(filenames, region, fieldspec, filterIds, flatten_filter, parseInfo, condition)
    else:
        it = itervariants(filenames, region, fieldspec, filterIds, flatten_filter, parseInfo)

    # slice?
    if slice:
        it = islice(it, *slice)

    # build an array from the iterator
    return _fromiter(it, dtype, count, progress, logstream)


def _fromiter(it, dtype, count, long progress=0, logstream=sys.stderr):
    if progress > 0:
        it = _iter_withprogress(it, progress, logstream)
    if count is not None:
        a = np.fromiter(it, dtype=dtype, count=count)
    else:
        a = np.fromiter(it, dtype=dtype)
    return a


def _iter_withprogress(iterable, long progress, logstream):
    cdef long i, n
    before_all = time.time()
    before = before_all
    for i, o in enumerate(iterable):
        yield o
        n = i+1
        if n % progress == 0:
            after = time.time()
            log(logstream, '%s rows in %.2fs; batch in %.2fs (%d rows/s)' % (n, after-before_all, after-before, progress/(after-before)))
            before = after
    after_all = time.time()
    log(logstream, '%s rows in %.2fs (%d rows/s)' % (n, after_all-before_all, n/(after_all-before_all)))


def itervariants(filenames,
                 region,
                 list fieldspec,
                 tuple filterIds,
                 bint flatten_filter,
                 parseInfo):
    cdef VariantCallFile *variantFile
    cdef Variant *var

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseInfo = parseInfo
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while _get_next_variant(variantFile, var):
            yield _mkvrow(var, fieldspec, filterIds, flatten_filter)

        del variantFile
        del var


def itervariants_with_condition(filenames,
                                region,
                                list fieldspec,
                                tuple filter_ids,
                                bint flatten_filter,
                                parse_info,
                                condition,
                                ):
    cdef VariantCallFile *variantFile
    cdef Variant *var
    cdef long i = 0
    cdef long n = len(condition)

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseInfo = parse_info
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while i < n and _get_next_variant(variantFile, var):
            if condition[i]:
                yield _mkvrow(var, fieldspec, filter_ids, flatten_filter)
            i += 1

        del variantFile
        del var


cdef inline bool _get_next_variant(VariantCallFile *variantFile, Variant *var):
    # break this out into a separate function so we can profile it
    return variantFile.getNextVariant(deref(var))


cdef inline object _mkvrow(Variant *var,
                            list fieldspec,
                            tuple filter_ids,
                            bint flatten_filter):
    out = list()
    for f, arity, fill, vcf_type, transformer in fieldspec:
        val = _mkvval(var, f, arity, fill, vcf_type, transformer, filter_ids)
        if f == 'FILTER' and flatten_filter:
            out.extend(val)
        else:
            out.append(val)
    return tuple(out)


cdef inline object _mkvval(Variant *var, string field, int arity, object fill, int vcf_type, transformer, tuple filter_ids):
    if field == FIELD_NAME_CHROM:
        out = var.sequenceName
    elif field == FIELD_NAME_POS:
        out = var.position
    elif field == FIELD_NAME_ID:
        out = var.id
    elif field == FIELD_NAME_REF:
        out = var.ref
    elif field == FIELD_NAME_ALT:
        out = _mkaltval(var, arity, fill)
    elif field == FIELD_NAME_QUAL:
        out = var.quality
    elif field == FIELD_NAME_FILTER:
        out = _mkfilterval(var, filter_ids)
    elif field == FIELD_NAME_NUM_ALLELES:
        out = <int>(var.alt.size() + 1)
    elif field == FIELD_NAME_IS_SNP:
        out = _is_snp(var)
    elif field == FIELD_NAME_SVLEN:
        out = _svlen(var, arity, fill)
    elif transformer is not None:
        out = transformer(var.info[field])
    elif vcf_type == FIELD_BOOL:
        # ignore arity, this is a flag
        out = (var.infoFlags.count(field) > 0)
    else:
        out = _mkval(var.info[field], arity, fill, vcf_type)
    return out


cdef inline object _mkaltval(Variant *var, int arity, object fill):
    if arity == 1:
        if var.alt.size() == 0:
            out = fill
        else:
            out = var.alt.at(0)
    elif var.alt.size() == arity:
        out = var.alt
        out = tuple(out)
    elif var.alt.size() > arity:
        out = var.alt
        out = tuple(out[:arity])
    else:
        out = var.alt
        out += [fill] * (arity-var.alt.size())
        out = tuple(out)
    return out


cdef inline object _mkfilterval(Variant *var, tuple filter_ids):
    filters = <list>split(var.filter, SEMICOLON)
    out = [(id in filters) for id in filter_ids]
    out = tuple(out)
    return out


cdef inline object _is_snp(Variant *var):
    cdef int i
    cdef bytes alt
    if var.ref.size() > 1:
        return False
    for i in range(var.alt.size()):
        alt = var.alt.at(i)
        if alt not in {'A', 'C', 'G', 'T'}:
            return False
    return True


cdef inline object _svlen(Variant *var, int arity, object fill):
    if arity == 1:
        return _svlen_single(var.ref, var.alt, fill)
    else:
        return _svlen_multi(var.ref, var.alt, arity, fill)


cdef inline object _svlen_single(string ref, vector[string]& alt, object fill):
    if alt.size() > 0:
        return <int>(alt.at(0).size() - ref.size())
    return fill


cdef inline object _svlen_multi(string ref, vector[string]& alt, int arity, object fill):
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
            print >>sys.stderr, 'WARNING: duplicate definition in header: %s' % f
        visited.add(f)


cdef inline object _mkval(vector[string]& string_vals, int arity, object fill, int vcf_type):
    if vcf_type == FIELD_FLOAT:
        out = _mkval_double(string_vals, arity, fill)
    elif vcf_type == FIELD_INTEGER:
        out = _mkval_long(string_vals, arity, fill)
    else:
        # make strings by default
        out = _mkval_string(string_vals, arity, fill)
    return out


cdef inline object _mkval_string(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        if string_vals.size() > 0:
            return string_vals.at(0)
        else:
            return fill
    else:
        return _mkval_string_multi(string_vals, arity, fill)


cdef inline object _mkval_string_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(string_vals.at(i))
        else:
            out.append(fill)
    return out


cdef inline object _mkval_double(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        out = _mkval_double_single(string_vals, fill)
    else:
        out = _mkval_double_multi(string_vals, arity, fill)
    return out


cdef inline object _mkval_double_single(vector[string]& string_vals, object fill):
    cdef double v
    if string_vals.size() > 0:
        return atof(string_vals.at(0).c_str())
    return fill


cdef inline object _mkval_double_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atof(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


cdef inline object _mkval_long(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        out = _mkval_long_single(string_vals, fill)
    else:
        out = _mkval_long_multi(string_vals, arity, fill)
    return out


cdef inline object _mkval_long_single(vector[string]& string_vals, object fill):
    if string_vals.size() > 0:
        return atol(string_vals.at(0).c_str())
    return fill


cdef inline object _mkval_long_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atol(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


def _calldata_fields(fields, exclude_fields, formatIds):
    if fields is None:
        fields = STANDARD_CALLDATA_FIELDS + formatIds
    else:
        for f in fields:
            if f not in STANDARD_CALLDATA_FIELDS and f not in formatIds:
                # support extracting FORMAT even if not declared in header, but warn...
                print >>sys.stderr, 'WARNING: no definition found for field %s' % f
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]
    return tuple(fields)


def _calldata_arities(fields, arities, formatCounts, ploidy):
    if arities is None:
        arities = dict()
    for f, vcf_count in zip(fields, formatCounts):
        if f not in arities:
            if f == 'genotype':
                arities[f] = ploidy
            elif f in DEFAULT_CALLDATA_ARITY:
                arities[f] = DEFAULT_CALLDATA_ARITY[f]
            elif vcf_count == ALLELE_NUMBER:
                # default to 2 (biallelic)
                arities[f] = 2
            elif vcf_count == GENOTYPE_NUMBER:
                # arity = (n + p - 1) choose p (n is number of alleles; p is ploidy)
                # default to biallelic (n = 2)
                arities[f] = ploidy + 1
            elif vcf_count <= 0:
                # catch any other cases of non-specific arity
                arities[f] = 1
            else:
                arities[f] = vcf_count
    return tuple(arities[f] for f in fields)


def _calldata_fills(fields, fills, formatTypes, ploidy):
    if fills is None:
        fills = dict()
    for f, vcf_type in zip(fields, formatTypes):
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


def calldata(filename,
             region=None,
             samples=None,
             ploidy=2,
             fields=None,
             exclude_fields=None,
             dtypes=None,
             arities=None,
             fills=None,
             vcf_types=None,
             count=None,
             progress=0,
             logstream=sys.stderr,
             condition=None,
             slice=None,
             verbose=False,
             cache=False,
             cachedir=None,
             skip_cached=False,
             ):
    """
    Load a numpy 1-dimensional structured array with data from the sample columns of a VCF
    file.

    Parameters
    ----------

    filename: string or list
        Name of the VCF file or list of file names
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'
    fields: list or array-like
        List of fields to extract from the VCF
    exclude_fields: list or array-like
        Fields to exclude from extraction
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred ones
    arities: dict or dict-like
        Override the amount of values to expect
    fills: dict or dict-like
        Dictionary containing field:fillvalue mappings used to override the
        default fill in values in VCF fields
    vcf_types: dict or dict-like
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header
    count: int
        Attempt to extract a specific number of records
    progress: int
        If greater than 0, log parsing progress
    logstream: file or file-like object
        Stream to use for logging progress
    condition: array
        Boolean array defining which rows to load
    slice: tuple or list
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every 10th row from the first 1000
    verbose: bool
        Log more messages.
    cache: bool
        If True, save the resulting numpy array to disk, and load from the cache if present rather than rebuilding
        from the VCF.
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

    if cache:

        if isinstance(filename, (list, tuple)):
            raise Exception('caching only supported when loading from a single VCF file')

        cache_fn = _mk_cache_fn(filename, array_type='calldata', region=region, cachedir=cachedir)
        if not os.path.exists(cache_fn) or os.path.getmtime(filename) > os.path.getmtime(cache_fn):
            if verbose:
                log(logstream, 'no cache file found or cache out of date')
            A = _build_calldata(filename,
                                region=region,
                                samples=samples,
                                ploidy=ploidy,
                                fields=fields,
                                exclude_fields=exclude_fields,
                                dtypes=dtypes,
                                arities=arities,
                                fills=fills,
                                vcf_types=vcf_types,
                                count=count,
                                progress=progress,
                                logstream=logstream,
                                condition=condition,
                                slice=slice,
                                verbose=verbose)
            if verbose:
                log(logstream, 'saving to cache', cache_fn)
            np.save(cache_fn, A)
            return A
        else:
            if skip_cached:
                if verbose:
                    log(logstream, 'skipping from cache', cache_fn)
                return None
            else:
                if verbose:
                    log(logstream, 'loading from cache', cache_fn)
                A = np.load(cache_fn)
                return A

    else:

        A = _build_calldata(filename,
                            region=region,
                            samples=samples,
                            ploidy=ploidy,
                            fields=fields,
                            exclude_fields=exclude_fields,
                            dtypes=dtypes,
                            arities=arities,
                            fills=fills,
                            vcf_types=vcf_types,
                            count=count,
                            progress=progress,
                            logstream=logstream,
                            condition=condition,
                            slice=slice,
                            verbose=verbose)
        return A


def _build_calldata(filename,
                    region=None,
                    samples=None,
                    ploidy=2,
                    fields=None,
                    exclude_fields=None,
                    dtypes=None,
                    arities=None,
                    fills=None,
                    vcf_types=None,
                    count=None,
                    progress=0,
                    logstream=sys.stderr,
                    condition=None,
                    slice=None,
                    verbose=False,
                    ):

    if verbose:
        log(logstream, 'loading calldata from', filename)

    filenames = _filenames_from_arg(filename)

    # extract definitions from VCF header
    vcf = PyVariantCallFile(filenames[0])
    _warn_duplicates(vcf.formatIds)
    format_ids = tuple(sorted(set(vcf.formatIds)))
    format_types = vcf.formatTypes
    format_counts = vcf.formatCounts
    all_samples = vcf.sampleNames

    # determine which samples to extract
    if samples is None:
        samples = all_samples
    else:
        for s in samples:
            assert s in all_samples, 'unknown sample: %s' % s
    samples = tuple(samples)

    # determine fields to extract
    fields = _calldata_fields(fields, exclude_fields, format_ids)

    # support for working around VCFs with bad FORMAT headers
    for f in fields:
        if f not in STANDARD_CALLDATA_FIELDS and f not in format_ids:
            # fall back to unary string; can be overridden with vcf_types, dtypes and arities args
            format_types[f] = FIELD_STRING
            format_counts[f] = 1
        if vcf_types is not None and f in vcf_types:
            # override type declared in VCF header
            format_types[f] = TYPESTRING2KEY[vcf_types[f]]

    format_types = tuple(format_types[f] if f in format_types else -1 for f in fields)
    format_counts = tuple(format_counts[f] if f in format_counts else -1 for f in fields)

    # determine expected number of values for each field
    arities = _calldata_arities(fields, arities, format_counts, ploidy)

    # determine fill values to use where number of values is less than expectation
    fills = _calldata_fills(fields, fills, format_types, ploidy)

    # construct a numpy dtype for structured array
    dtype = _calldata_dtype(fields, dtypes, format_types, arities, samples, ploidy)

    # zip up field parameters
    fieldspec = zip(fields, arities, fills, format_types)

    # set up iterator
    if condition is not None:
        it = itercalldata_with_condition(filenames, region, samples, ploidy, fieldspec, condition)
    else:
        it = itercalldata(filenames, region, samples, ploidy, fieldspec)

    # slice?
    if slice:
        it = islice(it, *slice)

    # build an array from the iterator
    return _fromiter(it, dtype, count, progress, logstream)


def calldata_2d(filename,
                region=None,
                samples=None,
                ploidy=2,
                fields=None,
                exclude_fields=None,
                dtypes=None,
                arities=None,
                fills=None,
                vcf_types=None,
                count=None,
                progress=0,
                logstream=sys.stderr,
                condition=None,
                slice=None,
                verbose=False,
                cache=False,
                cachedir=None,
                skip_cached=False,
               ):
    """
    Load a numpy 2-dimensional structured array with data from the sample columns of a VCF
    file. Convenience function, equivalent to calldata() followed by view2d().

    Parameters
    ----------

    filename: string or list
        Name of the VCF file or list of file names
    region: string
        Region to extract, e.g., 'chr1' or 'chr1:0-100000'
    fields: list or array-like
        List of fields to extract from the VCF
    exclude_fields: list or array-like
        Fields to exclude from extraction
    dtypes: dict or dict-like
        Dictionary cotaining dtypes to use instead of the default inferred ones
    arities: dict or dict-like
        Override the amount of values to expect
    fills: dict or dict-like
        Dictionary containing field:fillvalue mappings used to override the
        default fill in values in VCF fields
    vcf_types: dict or dict-like
        Dictionary containing field:string mappings used to override any
        bogus type declarations in the VCF header
    count: int
        Attempt to extract a specific number of records
    progress: int
        If greater than 0, log parsing progress
    logstream: file or file-like object
        Stream to use for logging progress
    condition: array
        Boolean array defining which rows to load
    slice: tuple or list
        Slice of the underlying iterator, e.g., (0, 1000, 10) takes every 10th row from the first 1000
    verbose: bool
        Log more messages.
    cache: bool
        If True, save the resulting numpy array to disk, and load from the cache if present rather than rebuilding
        from the VCF.
    cachedir: string
        Manually specify the directory to use to store cache files.
    skip_cached: bool
        If True and cache file is fresh, do not load and return None.

    """

    if cache:

        if isinstance(filename, (list, tuple)):
            raise Exception('caching only supported when loading from a single VCF file')

        cache_fn = _mk_cache_fn(filename, array_type='calldata_2d', region=region, cachedir=cachedir)
        if not os.path.exists(cache_fn) or os.path.getmtime(filename) > os.path.getmtime(cache_fn):
            if verbose:
                log(logstream, 'no cache file found or cache out of date')
            A = _build_calldata_2d(filename,
                                   region=region,
                                   samples=samples,
                                   ploidy=ploidy,
                                   fields=fields,
                                   exclude_fields=exclude_fields,
                                   dtypes=dtypes,
                                   arities=arities,
                                   fills=fills,
                                   vcf_types=vcf_types,
                                   count=count,
                                   progress=progress,
                                   logstream=logstream,
                                   condition=condition,
                                   slice=slice,
                                   verbose=verbose)
            if verbose:
                log(logstream, 'saving to cache', cache_fn)
            np.save(cache_fn, A)
            return A
        else:
            if skip_cached:
                if verbose:
                    log(logstream, 'skipping from cache', cache_fn)
                return None
            else:
                if verbose:
                    log(logstream, 'loading from cache', cache_fn)
                A = np.load(cache_fn)
                return A

    else:

        A = _build_calldata_2d(filename,
                               region=region,
                               samples=samples,
                               ploidy=ploidy,
                               fields=fields,
                               exclude_fields=exclude_fields,
                               dtypes=dtypes,
                               arities=arities,
                               fills=fills,
                               vcf_types=vcf_types,
                               count=count,
                               progress=progress,
                               logstream=logstream,
                               condition=condition,
                               slice=slice,
                               verbose=verbose)
        return A


def _build_calldata_2d(*args, **kwargs):
    C = calldata(*args, **kwargs)
    C2d = view2d(C)
    return C2d


def itercalldata(filenames,
                  region,
                  tuple samples,
                  int ploidy,
                  list fieldspec):
    cdef VariantCallFile *variantFile
    cdef Variant *var

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseInfo = False
        variantFile.parseSamples = True
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while _get_next_variant(variantFile, var):
            yield _mkcrow(var, samples, ploidy, fieldspec)

        del variantFile
        del var


def itercalldata_with_condition(filenames,
                                 region,
                                 tuple samples,
                                 int ploidy,
                                 list fieldspec,
                                 condition,
                                 ):
    cdef VariantCallFile *variantFile
    cdef Variant *var
    cdef long i = 0
    cdef long n = len(condition)

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseInfo = False
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while i < n:
            # only worth parsing samples if we know we want the variant
            if condition[i]:
                variantFile.parseSamples = True
                if not _get_next_variant(variantFile, var):
                    break
                yield _mkcrow(var, samples, ploidy, fieldspec)
            else:
                variantFile.parseSamples = False
                if not _get_next_variant(variantFile, var):
                    break
            i += 1

        del variantFile
        del var


cdef inline object _mkcrow(Variant *var,
                             tuple samples,
                             int ploidy,
                             list fieldspec):
    out = [_mkcvals(var, s, ploidy, fieldspec) for s in samples]
    return tuple(out)


cdef inline object _mkcvals(Variant *var,
                            string sample,
                            int ploidy,
                            list fieldspec):
    out = [_mkcval(var.samples[sample], ploidy, f, arity, fill, format_type)
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


cdef inline object _genotype(map[string, vector[string]]& sample_data, int ploidy):
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
            match_eff_main = _prog_eff_main.match(vals[0]) # ignore all but first effect
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

        filenames = _filenames_from_arg(self.filename)

        # extract definitions from VCF header
        vcf = PyVariantCallFile(filenames[0])

        # FILTER definitions
        filter_ids = vcf.filterIds
        _warn_duplicates(filter_ids)
        filter_ids = sorted(set(filter_ids))
        if 'PASS' not in filter_ids:
            filter_ids.append('PASS')
        filter_ids = tuple(filter_ids)

        # INFO definitions
        _warn_duplicates(vcf.infoIds)
        info_ids = tuple(sorted(set(vcf.infoIds)))
        info_types = vcf.infoTypes

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
        info_types = tuple(info_types[f] if f in info_types else -1 for f in fields)

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
            elif self.arities is not None and f in self.arities and self.arities[f] > 1:
                for i in range(1, self.arities[f] + 1):
                    header.append(f + '_' + str(i))
            elif f in self.flatten and self.flatten[f] is not None:
                fflds, _ = self.flatten[f]
                header.extend(fflds)
            else:
                header.append(f)
        header = tuple(header)
        # make data rows
        data = itervariantstable(filenames=filenames,
                                 region=self.region,
                                 fields=fields,
                                 arities=arities,
                                 info_types=info_types,
                                 parse_info=parse_info,
                                 filter_ids=filter_ids,
                                 flatten_filter=self.flatten_filter,
                                 fill=self.fill,
                                 flatten=self.flatten)
        return chain((header,), data)


def itervariantstable(filenames, region, fields, arities, info_types, parse_info, filter_ids, flatten_filter, fill, flatten):
    cdef VariantCallFile *variantFile
    cdef Variant *var

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseInfo = parse_info
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while _get_next_variant(variantFile, var):
            yield _mkvtblrow(var, fields, arities, info_types, filter_ids, flatten_filter, fill, flatten)

        del variantFile
        del var


cdef inline object _mkvtblrow(Variant *var,
                              fields,
                              arities,
                              info_types,
                              filter_ids,
                              flatten_filter,
                              fill,
                              flatten):
    out = list()
    cdef string field
    for field, arity, vcf_type in zip(fields, arities, info_types):
        if field == FIELD_NAME_CHROM:
            out.append(var.sequenceName)
        elif field == FIELD_NAME_POS:
            out.append(var.position)
        elif field == FIELD_NAME_ID:
            out.append(var.id)
        elif field == FIELD_NAME_REF:
            out.append(var.ref)
        elif field == FIELD_NAME_ALT:
            if arity is not None:
                vals = _mktblval_multi(var.alt, arity, fill)
                out.extend(vals)
            elif var.alt.size() == 0:
                out.append(fill)
            else:
                val = ','.join(var.alt)
                out.append(val)
        elif field == FIELD_NAME_QUAL:
            out.append(var.quality)
        elif field == FIELD_NAME_FILTER:
            if flatten_filter:
                out.extend(_mkfilterval(var, filter_ids))
            elif var.filter == DOT:
                out.append(fill)
            else:
                out.append(var.filter)
        elif field == FIELD_NAME_NUM_ALLELES:
            out.append(var.alt.size() + 1)
        elif field == FIELD_NAME_IS_SNP:
            out.append(_is_snp(var))
        else:
            if vcf_type == FIELD_BOOL:
                # ignore arity, this is a flag
                val = (var.infoFlags.count(field) > 0)
                out.append(val)
            else:
                if arity is not None:
                    vals = _mktblval_multi(var.info[field], arity, fill)
                    out.extend(vals)
                elif str(field) in flatten and flatten[str(field)] is not None:
                    _, t = flatten[str(field)]
                    vals = t(var.info[field])
                    out.extend(vals)
                elif var.info[field].size() == 0:
                    out.append(fill)
                else:
                    out.append(','.join(var.info[field]))
    return tuple(out)


cdef inline object _mktblval_multi(vector[string]& string_vals, int arity, object fill):
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
            match_eff_main = _prog_eff_main.match(vals[0])  # ignore all but first effect
            eff = [match_eff_main.group(1)] + match_eff_main.group(2).split('|')
            eff = [fill if v == '' else v for v in eff[:11]]
            return eff
    return _flatten


DEFAULT_FLATTEN = {
    'EFF': (EFF_FIELDS, flatten_eff),
}