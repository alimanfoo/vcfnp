# cython: profile = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: embedsignature = True


"""
Utility functions to extract data from a VCF file and load into a numpy array.

"""


__version__ = '0.16'


import sys
import re
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
from libc.stdlib cimport atoi, atof
from cython.operator cimport dereference as deref
import time
from itertools import islice
import os


cdef size_t npos = -1


cdef extern from "split.h":
    # split a string on a single delimiter character (delim)
    vector[string]& split(const string &s, char delim, vector[string] &elems)
    vector[string]  split(const string &s, char delim)
    # split a string on any character found in the string of delimiters (delims)
    vector[string]& split(const string &s, const string& delims, vector[string] &elems)
    vector[string]  split(const string &s, const string& delims)


cdef extern from "convert.h":
    bool convert(const string& s, int& r)
    bool convert(const string& s, float& r)


TYPESTRING2KEY = {
                  'Float': FIELD_FLOAT,
                  'Integer': FIELD_INTEGER,
                  'String': FIELD_STRING,
                  'Flag': FIELD_BOOL,
                  }

# these are the possible fields in the variants array
VARIANT_FIELDS = ('CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                  'num_alleles', 'is_snp', 'svlen')


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
DEFAULT_VARIANT_FILL = {'CHROM': '',
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
DEFAULT_TYPE_MAP = {FIELD_FLOAT: 'f4',
                    FIELD_INTEGER: 'i4',
                    FIELD_STRING: 'a12',
                    FIELD_BOOL: 'b1',
                    FIELD_UNKNOWN: 'a12' # leave as string
                    }


# default mapping from VCF field types to fill values for missing values
DEFAULT_FILL_MAP = {FIELD_FLOAT: 0.,
                    FIELD_INTEGER: 0,
                    FIELD_STRING: '.',
                    FIELD_BOOL: False,
                    FIELD_UNKNOWN: ''
                    }


# default dtypes for some known INFO fields where we know that lower
# precision is acceptable
DEFAULT_INFO_DTYPE = {
                     'AC': 'u2',
                     'AN': 'u2',
                     'HRun': 'u2',
                     'MLEAC': 'u2',
                     'MQ': 'f2',
                     'QD': 'f2',
                     'RPA': 'u2',
                     }

CALLDATA_FIELDS = ('is_called', 'is_phased', 'genotype')

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



def variants(filename,
             region=None,
             fields=None,
             exclude_fields=None,
             dtypes=None,
             arities=None,
             fills=None,
             count=None,
             progress=0,
             logstream=sys.stderr,
             condition=None,
             slice=None
             ):
    """
    Load an numpy structured array with data from the fixed fields of a VCF file
    (excluding INFO).

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

    Examples
    --------

        >>> from vcfnp import variants
        >>> a = variants('sample.vcf')
        >>> a
        array([ ('19', 111, '.', 'A', 'C', 9.600000381469727, (False, False, False), 2, True),
               ('19', 112, '.', 'A', 'G', 10.0, (False, False, False), 2, True),
               ('20', 14370, 'rs6054257', 'G', 'A', 29.0, (True, False, False), 2, True),
               ('20', 17330, '.', 'T', 'A', 3.0, (False, True, False), 2, True),
               ('20', 1110696, 'rs6040355', 'A', 'G', 67.0, (True, False, False), 3, True),
               ('20', 1230237, '.', 'T', '.', 47.0, (True, False, False), 2, False),
               ('20', 1234567, 'microsat1', 'G', 'GA', 50.0, (True, False, False), 3, False),
               ('20', 1235237, '.', 'T', '.', 0.0, (False, False, False), 2, False),
               ('X', 10, 'rsTest', 'AC', 'A', 10.0, (True, False, False), 3, False)],
              dtype=[('CHROM', '|S12'), ('POS', '<i4'), ('ID', '|S12'), ('REF', '|S12'), ('ALT', '|S12'), ('QUAL', '<f4'), ('FILTER', [('PASS', '|b1'), ('q10', '|b1'), ('s50', '|b1')]), ('num_alleles', '|u1'), ('is_snp', '|b1')])
        >>> a['QUAL']
        array([  9.60000038,  10.        ,  29.        ,   3.        ,
                67.        ,  47.        ,  50.        ,   0.        ,  10.        ], dtype=float32)
        >>> a['FILTER']['PASS']
        array([False, False,  True, False,  True,  True,  True, False,  True], dtype=bool)

    """

    if isinstance(filename, basestring):
        filenames = [filename]
    else:
        filenames = filename

    for fn in filenames:
        if not os.path.exists(fn):
            raise Exception('file not found: %s' % fn)

    # determine fields to extract
    if fields is None:
        fields = VARIANT_FIELDS
    else:
        for f in fields:
            assert f in VARIANT_FIELDS, 'unknown field: %s' % f

    # exclude fields
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]

    # determine a numpy dtype for each field
    if dtypes is None:
        dtypes = dict()
    for f in fields:
        if f == 'FILTER':
            filterIds = PyVariantCallFile(filenames[0]).filterIds
            warn_duplicates(filterIds)
            t = [(flt, 'b1') for flt in sorted(set(filterIds))]
            if 'PASS' not in filterIds:
                t += [('PASS', 'b1')]
            dtypes[f] = t
        elif f not in dtypes:
            dtypes[f] = DEFAULT_VARIANT_DTYPE[f]

    # determine expected number of values for each field
    if arities is None:
        arities = dict()
    for f in fields:
        if f == 'FILTER':
            arities[f] = 1 # one structured value
        elif f not in arities:
            arities[f] = DEFAULT_VARIANT_ARITY[f]

    # determine fill values to use where number of values is less than expectation
    if fills is None:
        fills = dict()
    for f in fields:
        if f == 'FILTER':
            fills[f] = False
        elif f not in fills:
            fills[f] = DEFAULT_VARIANT_FILL[f]

    # convert to tuples for faster iteration
    fields = tuple(fields)
    dtypes = tuple(dtypes[f] for f in fields)
    arities = tuple(arities[f] for f in fields)
    fills = tuple(fills[f] for f in fields)

    # construct a numpy dtype for structured array
    dtype = list()
    for f, t, n in zip(fields, dtypes, arities):
        if n == 1:
            dtype.append((f, t))
        else:
            dtype.append((f, t, (n,)))

    # set up iterator
    if condition is not None:
        it = itervariants_with_condition(filenames, region, fields, arities, fills, condition)
    else:
        it = itervariants(filenames, region, fields, arities, fills)

    # slice?
    if slice:
        it = islice(it, *slice)

    # build an array from the iterator
    return _fromiter(it, dtype, count, progress, logstream)



def _fromiter(it, dtype, count, int progress=0, logstream=sys.stderr):
    if progress > 0:
        it = _iter_withprogress(it, progress, logstream)
    if count is not None:
        a = np.fromiter(it, dtype=dtype, count=count)
    else:
        a = np.fromiter(it, dtype=dtype)
    return a



def _iter_withprogress(iterable, int progress, logstream):
    cdef int i, n
    before_all = time.time()
    before = before_all
    for i, o in enumerate(iterable):
        yield o
        n = i+1
        if n % progress == 0:
            after = time.time()
            print >>logstream, '%s rows in %.2fs; batch in %.2fs (%d rows/s)' % (n, after-before_all, after-before, progress/(after-before))
            before = after
    after_all = time.time()
    print >>logstream, '%s rows in %.2fs (%d rows/s)' % (n, after_all-before_all, n/(after_all-before_all))



def itervariants(filenames,
                 region,
                 tuple fields,
                 tuple arities,
                 tuple fills):
    cdef VariantCallFile *variantFile
    cdef Variant *var
#    cdef vector[string] filterIds

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
        filterIds = <list>variantFile.filterIds()
        filterIds = sorted(set(filterIds))
        if 'PASS' not in filterIds:
            filterIds += ['PASS']
        filterIds = tuple(filterIds)

        while _get_next_variant(variantFile, var):
            yield _mkvvals(var, fields, arities, fills, filterIds)

        del variantFile
        del var


def itervariants_with_condition(filenames,
                                 region,
                                 tuple fields,
                                 tuple arities,
                                 tuple fills,
                                 condition):
    cdef VariantCallFile *variantFile
    cdef Variant *var
#    cdef vector[string] filterIds
    cdef int i = 0
    cdef int n = len(condition)

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
        filterIds = <list>variantFile.filterIds()
        filterIds = sorted(set(filterIds))
        if 'PASS' not in filterIds:
            filterIds += ['PASS']
        filterIds = tuple(filterIds)

        while i < n and _get_next_variant(variantFile, var):
            if condition[i]:
                yield _mkvvals(var, fields, arities, fills, filterIds)
            i += 1

        del variantFile
        del var


cdef inline bool _get_next_variant(VariantCallFile *variantFile, Variant *var):
    # break this out into a separate function so we can profile it
    return variantFile.getNextVariant(deref(var))



cdef inline object _mkvvals(Variant *var,
                            tuple fields,
                            tuple arities,
                            tuple fills,
                            tuple filterIds):
    out = tuple([_mkvval(var, f, arity, fill, filterIds) for (f, arity, fill) in zip(fields, arities, fills)])
    return out



cdef inline object _mkvval(Variant *var, string field, int arity, object fill, tuple filterIds):
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
        out = _mkfilterval(var, filterIds)
    elif field == FIELD_NAME_NUM_ALLELES:
        out = var.alt.size() + 1
    elif field == FIELD_NAME_IS_SNP:
        out = _is_snp(var)
    elif field == FIELD_NAME_SVLEN:
        out = _svlen(var, arity, fill)
    else:
        out = 0 # TODO review this
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



cdef inline object _mkfilterval(Variant *var, tuple filterIds):
    filters = <list>split(var.filter, SEMICOLON)
    out = [(id in filters) for id in filterIds]
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


cdef inline object _svlen(Variant *var, int arity, int fill):
    cdef int i
    cdef bytes alt
    if arity == 1:
        return _svlen_single(var.ref, var.alt, fill)
    else:
        return _svlen_multi(var.ref, var.alt, arity, fill)


cdef inline int _svlen_single(string ref, vector[string]& alt, int fill):
    if alt.size() > 0:
        return alt.at(0).size() - ref.size()
    return fill


cdef inline vector[int] _svlen_multi(string ref, vector[string]& alt, int arity, int fill):
    cdef int i
    cdef vector[int] out
    for i in range(arity):
        if i < alt.size():
            out.push_back(alt.at(i).size() - ref.size())
        else:
            out.push_back(fill)
    return out


def info(filename,
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
         ):
    """
    Load a numpy structured array with data from the INFO field of a VCF file.

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

    Examples
    --------

        >>> from vcfnp import info
        >>> a = info('sample.vcf')
        >>> a
        array([(0, 0, 0, 0, 0.0, '.', False, False),
               (0, 0, 0, 0, 0.0, '.', False, False),
               (3, 0, 0, 14, 0.5, '.', True, True),
               (3, 0, 0, 11, 0.017000000923871994, '.', False, False),
               (2, 0, 0, 10, 0.3330000042915344, 'T', True, False),
               (3, 0, 0, 13, 0.0, 'T', False, False),
               (3, 6, 3, 9, 0.0, 'G', False, False),
               (0, 0, 0, 0, 0.0, '.', False, False),
               (0, 0, 0, 0, 0.0, '.', False, False)],
              dtype=[('NS', '<i4'), ('AN', '<u2'), ('AC', '<u2'), ('DP', '<i4'), ('AF', '<f4'), ('AA', '|S12'), ('DB', '|b1'), ('H2', '|b1')])

    """

    if isinstance(filename, basestring):
        filenames = [filename]
    else:
        filenames = filename

    for fn in filenames:
        if not os.path.exists(fn):
            raise Exception('file not found: %s' % fn)

    vcf = PyVariantCallFile(filenames[0])
    # warn about duplicate field definitions
    warn_duplicates(vcf.infoIds)
    infoIds = sorted(set(vcf.infoIds))
    infoTypes = vcf.infoTypes
    infoCounts = vcf.infoCounts

    # determine INFO fields to extract
    if fields is None:
        fields = infoIds  # extract all INFO fields
    else:
        for f in fields:
            if f not in infoIds:
                print >>sys.stderr, 'WARNING: no definition found for field %s' % f
                # fall back to unary string, can be overridden with vcf_types, dtypes and arities args
                infoTypes[f] = FIELD_STRING
                infoCounts[f] = 1

    # exclude fields
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]

    # override vcf types
    if vcf_types is not None:
        for f in vcf_types:
            infoTypes[f] = TYPESTRING2KEY[vcf_types[f]]

    # determine a numpy dtype for each field
    if dtypes is None:
        dtypes = dict()
    for f in fields:
        if f not in dtypes:
            if f in DEFAULT_INFO_DTYPE:
                # known INFO field
                dtypes[f] = DEFAULT_INFO_DTYPE[f]
            else:
                vcf_type = infoTypes[f]
                dtypes[f] = DEFAULT_TYPE_MAP[vcf_type]

    # determine expected number of values for each field
    if arities is None:
        arities = dict()
    for f in fields:
        if f not in arities:
            vcf_count = infoCounts[f]
            if vcf_count == ALLELE_NUMBER:
                # default to 1 (biallelic)
                arities[f] = 1
            elif vcf_count <= 0:
                # catch any other cases of non-specific arity
                arities[f] = 1
            else:
                arities[f] = vcf_count

    # determine fill values to use where number of values is less than expectation
    if fills is None:
        fills = dict()
    for f in fields:
        if f not in fills:
            vcf_type = infoTypes[f]
            fills[f] = DEFAULT_FILL_MAP[vcf_type]

    if transformers is None:
        transformers = dict()
    for f in fields:
        if f not in transformers:
            transformers[f] = None

    # convert to tuples for faster iteration
    fields = tuple(fields)
    dtypes = tuple(dtypes[f] for f in fields)
    arities = tuple(arities[f] for f in fields)
    fills = tuple(fills[f] for f in fields)
    infoTypes = tuple(infoTypes[f] for f in fields)
    transformers = tuple(transformers[f] for f in fields)

    # construct a numpy dtype for structured array
    dtype = list()
    for f, t, n in zip(fields, dtypes, arities):
        if n == 1:
            dtype.append((f, t))
        else:
            dtype.append((f, t, (n,)))

    # set up iterator
    if condition is not None:
        it = iterinfo_with_condition(filenames, region, fields, arities, fills, infoTypes, transformers, condition)
    else:
        it = iterinfo(filenames, region, fields, arities, fills, infoTypes, transformers)

    # slice?
    if slice:
        it = islice(it, *slice)

    # build an array from the iterator
    return _fromiter(it, dtype, count, progress, logstream)


def warn_duplicates(fields):
    visited = set()
    for f in fields:
        if f in visited:
            print >>sys.stderr, 'WARNING: duplicate definition in header: %s' % f
        visited.add(f)


def iterinfo(filenames,
             region,
             tuple fields,
             tuple arities,
             tuple fills,
             tuple infoTypes,
             tuple transformers):
    cdef VariantCallFile *variantFile
    cdef Variant *var

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while _get_next_variant(variantFile, var):
            yield _mkivals(var, fields, arities, fills, infoTypes, transformers)

        del variantFile
        del var


def iterinfo_with_condition(filenames,
                             region,
                             tuple fields,
                             tuple arities,
                             tuple fills,
                             tuple infoTypes,
                             tuple transformers,
                             condition):
    cdef VariantCallFile *variantFile
    cdef Variant *var
    cdef int i = 0
    cdef int n = len(condition)

    for current_filename in filenames:
        variantFile = new VariantCallFile()
        variantFile.open(current_filename)
        variantFile.parseSamples = False
        if region is not None:
            region_set = variantFile.setRegion(region)
            if not region_set:
                raise StopIteration
        var = new Variant(deref(variantFile))

        while i < n and _get_next_variant(variantFile, var):
            if condition[i]:
                yield _mkivals(var, fields, arities, fills, infoTypes, transformers)
            i += 1

        del variantFile
        del var


cdef inline object _mkivals(Variant *var,
                            tuple fields,
                            tuple arities,
                            tuple fills,
                            tuple infoTypes,
                            tuple transformers):
    out = [_mkival(var, f, arity, fill, infoType, transformer) for (f, arity, fill, infoType, transformer) in zip(fields, arities, fills, infoTypes, transformers)]
    return tuple(out)



cdef inline object _mkival(Variant *var, string field, int arity, object fill, int vcf_type, transformer):
    if transformer is not None:
        out = transformer(var.info[field])
    elif vcf_type == FIELD_BOOL:
        # ignore arity, this is a flag
        out = (var.infoFlags.count(field) > 0)
    else:
        out = _mkval(var.info[field], arity, fill, vcf_type)
    return out



cdef inline object _mkval(vector[string]& string_vals, int arity, object fill, int vcf_type):
    if vcf_type == FIELD_FLOAT:
        out = _mkval_float(string_vals, arity, fill)
    elif vcf_type == FIELD_INTEGER:
        out = _mkval_int(string_vals, arity, fill)
    else:
        # make strings by default
        out = _mkval_string(string_vals, arity, fill)
    return out



cdef inline object _mkval_string(vector[string]& string_vals, int arity, string fill):
    if arity == 1:
        if string_vals.size() > 0:
            return string_vals.at(0)
        else:
            return fill
    else:
        return _mkval_string_multi(string_vals, arity, fill)



cdef inline vector[string] _mkval_string_multi(vector[string]& string_vals, int arity, string fill):
    cdef int i
    cdef vector[string] v
    for i in range(arity):
        if i < string_vals.size():
            v.push_back(string_vals.at(i))
        else:
            v.push_back(fill)
    return v



cdef inline object _mkval_float(vector[string]& string_vals, int arity, float fill):
    if arity == 1:
        out = _mkval_float_single(string_vals, fill)
    else:
        out = _mkval_float_multi(string_vals, arity, fill)
    return out



cdef inline float _mkval_float_single(vector[string]& string_vals, float fill):
    cdef float v
    if string_vals.size() > 0:
        return atof(string_vals.at(0).c_str())
    return fill
#    cdef float v = fill
#    if string_vals.size() > 0:
#        convert(string_vals.at(0), v)
#    return v



cdef inline vector[float] _mkval_float_multi(vector[string]& string_vals, int arity, float fill):
    cdef int i
    cdef vector[float] out
    for i in range(arity):
        if i < string_vals.size():
            out.push_back(atof(string_vals.at(i).c_str()))
        else:
            out.push_back(fill)
    return out
#cdef inline vector[float] _mkval_float_multi(vector[string]& string_vals, int arity, float fill):
#    cdef int i
#    cdef float v
#    cdef vector[float] out
#    for i in range(arity):
#        v = fill
#        if i < string_vals.size():
#            convert(string_vals.at(i), v)
#        out.push_back(v)
#    return out



cdef inline object _mkval_int(vector[string]& string_vals, int arity, int fill):
    if arity == 1:
        out = _mkval_int_single(string_vals, fill)
    else:
        out = _mkval_int_multi(string_vals, arity, fill)
    return out



cdef inline int _mkval_int_single(vector[string]& string_vals, int fill):
    cdef int v
    if string_vals.size() > 0:
        return atoi(string_vals.at(0).c_str())
    return fill
#    cdef int v = fill
#    if string_vals.size() > 0:
#        convert(string_vals.at(0), v)
#    return v



cdef inline vector[int] _mkval_int_multi(vector[string]& string_vals, int arity, int fill):
    cdef int i
    cdef vector[int] out
    for i in range(arity):
        if i < string_vals.size():
            out.push_back(atoi(string_vals.at(i).c_str()))
        else:
            out.push_back(fill)
    return out
#cdef inline vector[int] _mkval_int_multi(vector[string]& string_vals, int arity, int fill):
#    cdef int i
#    cdef int v
#    cdef vector[int] out
#    for i in range(arity):
#        v = fill
#        if i < string_vals.size():
#            convert(string_vals.at(i), v)
#        out.push_back(v)
#    return out



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
             ):
    """
    Load a numpy structured array with data from the sample columns of a VCF
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

    Examples
    --------

        >>> from vcfnp import samples
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
        >>> a['NA00001']
        array([(True, True, [0, 0], '0|0', 0, 0, [10, 10]),
               (True, True, [0, 0], '0|0', 0, 0, [10, 10]),
               (True, True, [0, 0], '0|0', 48, 1, [51, 51]),
               (True, True, [0, 0], '0|0', 49, 3, [58, 50]),
               (True, True, [1, 2], '1|2', 21, 6, [23, 27]),
               (True, True, [0, 0], '0|0', 54, 0, [56, 60]),
               (True, False, [0, 1], '0/1', 0, 4, [0, 0]),
               (True, False, [0, 0], '0/0', 0, 0, [0, 0]),
               (True, False, [0, -1], '0', 0, 0, [0, 0])],
              dtype=[('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))])

    """

    if isinstance(filename, basestring):
        filenames = [filename]
    else:
        filenames = filename

    for fn in filenames:
        if not os.path.exists(fn):
            raise Exception('file not found: %s' % fn)

    vcf = PyVariantCallFile(filenames[0])
    # warn about duplicate field definitions
    warn_duplicates(vcf.formatIds)
    formatIds = sorted(set(vcf.formatIds))
    formatTypes = vcf.formatTypes
    formatCounts = vcf.formatCounts
    all_samples = vcf.sampleNames

    if samples is None:
        samples = all_samples
    else:
        for s in samples:
            assert s in all_samples, 'unknown sample: %s' % s

    # determine fields to extract
    if fields is None:
        fields = list(CALLDATA_FIELDS) + formatIds
    else:
        for f in fields:
            if f not in CALLDATA_FIELDS and f not in formatIds:
                print >>sys.stderr, 'WARNING: no definition found for field %s' % f
                # fall back to unary string, can be overridden with vcf_types, dtypes and arities args
                formatTypes[f] = FIELD_STRING
                formatCounts[f] = 1

    # exclude fields
    if exclude_fields is not None:
        fields = [f for f in fields if f not in exclude_fields]

    # override vcf types
    if vcf_types is not None:
        for f in vcf_types:
            formatTypes[f] = TYPESTRING2KEY[vcf_types[f]]

    # determine a numpy dtype for each field
    if dtypes is None:
        dtypes = dict()
    for f in fields:
        if f not in dtypes:
            if f == 'GT':
                dtypes[f] = 'a%d' % ((ploidy*2)-1)
            elif f in DEFAULT_CALLDATA_DTYPE:
                # known field
                dtypes[f] = DEFAULT_CALLDATA_DTYPE[f]
            else:
                vcf_type = formatTypes[f]
                dtypes[f] = DEFAULT_TYPE_MAP[vcf_type]

    # determine expected number of values for each field
    if arities is None:
        arities = dict()
    for f in fields:
        if f not in arities:
            if f == 'genotype':
                arities[f] = ploidy
            elif f in DEFAULT_CALLDATA_ARITY:
                arities[f] = DEFAULT_CALLDATA_ARITY[f]
            else:
                vcf_count = formatCounts[f]
                if vcf_count == ALLELE_NUMBER:
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

    # determine fill values to use where number of values is less than expectation
    if fills is None:
        fills = dict()
    for f in fields:
        if f not in fills:
            if f == 'GT':
                fills[f] = '/'.join(['.'] * ploidy)
            elif f in DEFAULT_CALLDATA_FILL:
                fills[f] = DEFAULT_CALLDATA_FILL[f]
            else:
                vcf_type = formatTypes[f]
                fills[f] = DEFAULT_FILL_MAP[vcf_type]

    # convert to tuples for faster iteration
    samples = tuple(samples)
    fields = tuple(fields)
    formatTypes = tuple(formatTypes[f] if f in formatTypes else -1 for f in fields)
    dtypes = tuple(dtypes[f] for f in fields)
    arities = tuple(arities[f] for f in fields)
    fills = tuple(fills[f] for f in fields)

    # construct a numpy dtype for structured array cells
    cell_dtype = list()
    for f, t, n in zip(fields, dtypes, arities):
        if n == 1:
            cell_dtype.append((f, t))
        else:
            cell_dtype.append((f, t, (n,)))
    # construct a numpy dtype for structured array
    dtype = [(s, cell_dtype) for s in samples]

    # set up iterator
    if condition is not None:
        it = itercalldata_with_condition(filenames, region, samples, ploidy, fields, formatTypes, arities, fills, condition)
    else:
        it = itercalldata(filenames, region, samples, ploidy, fields, formatTypes, arities, fills)

    # slice?
    if slice:
        it = islice(it, *slice)

    # build an array from the iterator
    return _fromiter(it, dtype, count, progress, logstream)



def itercalldata(filenames,
                  region,
                  tuple samples,
                  int ploidy,
                  tuple fields,
                  tuple formatTypes,
                  tuple arities,
                  tuple fills):
    cdef VariantCallFile *variantFile
    cdef Variant *var

    fieldspec = zip(fields, arities, fills, formatTypes)

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
            yield _mkssvals(var, samples, ploidy, fieldspec)

        del variantFile
        del var


def itercalldata_with_condition(filenames,
                                 region,
                                 tuple samples,
                                 int ploidy,
                                 tuple fields,
                                 tuple formatTypes,
                                 tuple arities,
                                 tuple fills,
                                 condition,
                                 ):
    cdef VariantCallFile *variantFile
    cdef Variant *var
    cdef int i = 0
    cdef int n = len(condition)

    fieldspec = zip(fields, arities, fills, formatTypes)

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
            # only both parsing samples if we know we want the variant
            if condition[i]:
                variantFile.parseSamples = True
                if not _get_next_variant(variantFile, var):
                    break
                yield _mkssvals(var, samples, ploidy, fieldspec)
            else:
                variantFile.parseSamples = False
                if not _get_next_variant(variantFile, var):
                    break
            i += 1

        del variantFile
        del var


cdef inline object _mkssvals(Variant *var,
                             tuple samples,
                             int ploidy,
                             list fieldspec):
    out = [_mksvals(var, s, ploidy, fieldspec) for s in samples]
    return tuple(out)



cdef inline object _mksvals(Variant *var,
                            string sample,
                            int ploidy,
                            list fieldspec):
    out = [_mksval(var.samples[sample], ploidy, f, arity, fill, formatType)
           for (f, arity, fill, formatType) in fieldspec]
    return tuple(out)


# alternative implementations, no speed difference...
#
#
# cdef inline object _mkssvals(Variant *var,
#                              tuple samples,
#                              int ploidy,
#                              list fieldspec):
#     cdef int i
#     out = list()
#     for i in range(len(samples)):
#         s = samples[i]
#         out.append(_mksvals(var, s, ploidy, fieldspec))
#     return tuple(out)
#
#
# cdef inline object _mksvals(Variant *var,
#                             string sample,
#                             int ploidy,
#                             list fieldspec):
#     cdef int i
#     out = list()
#     for i in range(len(fieldspec)):
#         f, arity, fill, formatType = fieldspec[i]
#         out.append(_mksval(var.samples[sample], ploidy, f, arity, fill, formatType))
#     return tuple(out)


cdef inline object _mksval(map[string, vector[string]]& sample_data,
                           int ploidy,
                           string field,
                           int arity,
                           object fill,
                           int formatType):
    if field == FIELD_NAME_IS_CALLED:
        return _is_called(sample_data)
    elif field == FIELD_NAME_IS_PHASED:
        return _is_phased(sample_data)
    elif field == FIELD_NAME_GENOTYPE:
        return _genotype(sample_data, ploidy)
    else:
        return _mkval(sample_data[field], arity, fill, formatType)



cdef inline bool _is_called(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return (gts.at(0).find('.') == npos)


cdef inline bool _is_phased(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return (gts.at(0).find('|') != npos)


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
#cdef inline object _genotype(map[string, vector[string]]& sample_data, int ploidy):
#    cdef vector[string] *gts
#    cdef vector[int] alleles
#    cdef vector[string] allele_strings
#    cdef int i
#    cdef int allele
#    gts = &sample_data[FIELD_NAME_GT]
#    if gts.size() == 0:
#        if ploidy == 1:
#            return -1
#        else:
#            return (-1,) * ploidy
#    else:
#        split(gts.at(0), GT_DELIMS, allele_strings)
#        if ploidy == 1:
#            allele = -1
#            if allele_strings.size() > 0:
#                convert(allele_strings.at(0), allele)
#            return allele
#        else:
#            for i in range(ploidy):
#                allele = -1
#                if i < allele_strings.size():
#                    convert(allele_strings.at(i), allele)
#                alleles.push_back(allele)
#            return tuple(alleles)


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


EFF_DEFAULT_FILLS = ('.', '.', '.', '.', '.', -1, '.', '.', -1, '.', -1)


def eff_default_transformer(fills=EFF_DEFAULT_FILLS):
    """
    Return a simple transformer function for parsing EFF annotations. N.B.,
    ignores all but the first effect.

    """
    prog_eff_main = re.compile(r'([^(]+)\(([^)]+)\)')
    def _transformer(vals):
        if len(vals) == 0:
            return fills
        else:
            match_eff_main = prog_eff_main.match(vals[0]) # ignore all but first effect
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




