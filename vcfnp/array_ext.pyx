from __future__ import print_function, absolute_import, division


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport atoi, atol, atof
from cython.operator cimport dereference as deref


from vcfnp.compat import b as _b
from vcfnp.vcflib cimport VariantCallFile, Variant, VariantFieldType, \
    VariantFieldNumber


cdef size_t npos = -1


cdef extern from "split.h":
    # split a string on a single delimiter character (delim)
    vector[string]& split(const string &s, char delim, vector[string] &elems)
    vector[string]  split(const string &s, char delim)
    # split a string on any character found in the string of delimiters (delims)
    vector[string]& split(const string &s, const string& delims, vector[string] &elems)
    vector[string]  split(const string &s, const string& delims)


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


def itervariants(vcf_fns, region, fields, arities, fills, info_types,
                 transformers, filter_ids, flatten_filter, parse_info,
                 condition):
    """Iterate over variants from a VCF file, and generate a tuple for each
    variant suitable for loading into a numpy array."""

    # force to bytes
    vcf_fns = _b(vcf_fns)
    region = _b(region)
    fields = _b(fields)
    filter_ids = _b(filter_ids)

    # zip up field information for convenience
    fieldspec = list(zip(fields, arities, fills, info_types, transformers))

    if condition is None:
        return _itervariants(vcf_fns=vcf_fns, region=region,
                             fieldspec=fieldspec, filter_ids=filter_ids,
                             flatten_filter=flatten_filter,
                             parse_info=parse_info)
    else:
        return _itervariants_with_condition(vcf_fns=vcf_fns, region=region,
                                            fieldspec=fieldspec,
                                            filter_ids=filter_ids,
                                            flatten_filter=flatten_filter,
                                            parse_info=parse_info,
                                            condition=condition)


def _itervariants(vcf_fns, region, fieldspec, filter_ids, flatten_filter,
                  parse_info):

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


def _itervariants_with_condition(vcf_fns, region, fieldspec, filter_ids,
                                 flatten_filter, parse_info, condition):

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
    elif vcf_type == VariantFieldType.FIELD_BOOL:
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


cdef inline object _mkval(vector[string]& string_vals, int arity, object fill,
                          int vcf_type):
    if vcf_type == VariantFieldType.FIELD_FLOAT:
        out = _mkval_double(string_vals, arity, fill)
    elif vcf_type == VariantFieldType.FIELD_INTEGER:
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
