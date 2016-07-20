# cython: profile = False
# cython: boundscheck = True
# cython: wraparound = False
# cython: embedsignature = True
from __future__ import print_function, absolute_import, division


from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport atoi, atol, atof
# noinspection PyUnresolvedReferences
from cython.operator cimport dereference as deref
import logging


from vcfnp.compat cimport b as _b, s as _s
from vcfnp.vcflib cimport VariantCallFile, Variant, FIELD_STRING, \
    FIELD_INTEGER, FIELD_BOOL, FIELD_FLOAT, FIELD_UNKNOWN, ALLELE_NUMBER, \
    GENOTYPE_NUMBER


logger = logging.getLogger(__name__)
debug = logger.debug


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
cdef string FIELD_NAME_GENOTYPE_AC = b'genotype_ac'
cdef string FIELD_NAME_PLOIDY = b'ploidy'


def itervariants(vcf_fns, region, fields, arities, fills, info_types,
                 transformers, filter_ids, flatten_filter, parse_info,
                 condition, truncate):
    """Iterate over variants from a VCF file, and generate a tuple for each
    variant suitable for loading into a numpy array."""

    # force to bytes
    vcf_fns = _b(tuple(vcf_fns))
    fields = _b(tuple(fields))
    arities = tuple(arities)
    fills = tuple(fills)
    info_types = tuple(info_types)
    transformers = tuple(transformers)
    filter_ids = _b(tuple(filter_ids))

    # zip up field information for convenience
    fieldspec = tuple(zip(fields, arities, fills, info_types, transformers))

    if condition is None:
        return _itervariants(vcf_fns=vcf_fns, region=region,
                             fieldspec=fieldspec, filter_ids=filter_ids,
                             flatten_filter=flatten_filter,
                             parse_info=parse_info, truncate=truncate)
    else:
        return _itervariants_with_condition(vcf_fns=vcf_fns, region=region,
                                            fieldspec=fieldspec,
                                            filter_ids=filter_ids,
                                            flatten_filter=flatten_filter,
                                            parse_info=parse_info,
                                            condition=condition,
                                            truncate=truncate)


def _itervariants(vcf_fns, region, fieldspec, filter_ids, flatten_filter,
                  parse_info, truncate):

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
        region_start, region_stop = None, None
        if region is not None:
            # set genome region to extract variants from
            region_set = variant_file.setRegion(_b(region))
            if not region_set:
                raise StopIteration
            if ':' in region:
                _, region_start_stop = region.split(':')
                region_start, region_stop = [int(v) for v in
                                             region_start_stop.split('-')]
        variant = new Variant(deref(variant_file))

        # iterate over variants
        while _get_next_variant(variant_file, variant):
            if region_start is not None and truncate and \
                    variant.position < region_start:
                continue
            yield _mkvrow(variant, fieldspec, filter_ids, flatten_filter)

        # clean up
        del variant_file
        del variant


def _itervariants_with_condition(vcf_fns, region, fieldspec, filter_ids,
                                 flatten_filter, parse_info, condition,
                                 truncate):

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
        region_start, region_stop = None, None
        if region is not None:
            region_set = variant_file.setRegion(_b(region))
            if not region_set:
                raise StopIteration
            if ':' in region:
                _, region_start_stop = region.split(':')
                region_start, region_stop = [int(v) for v in
                                             region_start_stop.split('-')]
        variant = new Variant(deref(variant_file))

        while i < n and _get_next_variant(variant_file, variant):
            if region_start is not None and truncate and \
                    variant.position < region_start:
                i += 1
                continue
            if condition[i]:
                yield _mkvrow(variant, fieldspec, filter_ids, flatten_filter)
            i += 1

        del variant_file
        del variant


cdef _get_next_variant(VariantCallFile *variant_file, Variant *variant):
    # break this out into a separate function so we can profile it
    return variant_file.getNextVariant(deref(variant))


cdef _mkvrow(Variant *variant, tuple fieldspec, tuple filter_ids,
             bint flatten_filter):
    """Make a row of variant data."""
    out = list()
    for f, arity, fill, vcf_type, transformer in fieldspec:
        val = _mkvval(variant, f, arity, fill, vcf_type, transformer,
                      filter_ids)
        if (f == b'FILTER') and flatten_filter:
            out.extend(val)
        else:
            out.append(val)
    return tuple(out)


cdef _mkvval(Variant *variant, string field, int arity, object fill,
             int vcf_type, transformer, tuple filter_ids):
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


cdef _mkaltval(Variant *variant, int arity, object fill):
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


cdef _mkfilterval(Variant *variant, tuple filter_ids):
    filters = <list>split(variant.filter, SEMICOLON)
    out = [(f in filters) for f in filter_ids]
    out = tuple(out)
    return out


cdef _is_snp(Variant *variant):
    cdef int i
    cdef bytes alt
    if variant.ref.size() > 1:
        return False
    for i in range(variant.alt.size()):
        alt = variant.alt.at(i)
        if alt not in {b'A', b'C', b'G', b'T', b'*'}:
            return False
    return True


cdef _svlen(Variant *variant, int arity, object fill):
    if arity == 1:
        return _svlen_single(variant.ref, variant.alt, fill)
    else:
        return _svlen_multi(variant.ref, variant.alt, arity, fill)


cdef _svlen_single(string ref, vector[string]& alt, object fill):
    if alt.size() > 0:
        return <int>(alt.at(0).size() - ref.size())
    return fill


cdef _svlen_multi(string ref, vector[string]& alt, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < alt.size():
            out.append(<int>(alt.at(i).size() - ref.size()))
        else:
            out.append(fill)
    return out


cdef _mkval(vector[string]& string_vals, int arity, object fill, int vcf_type):
    if vcf_type == FIELD_FLOAT:
        out = _mkval_double(string_vals, arity, fill)
    elif vcf_type == FIELD_INTEGER:
        out = _mkval_long(string_vals, arity, fill)
    else:
        # make strings by default
        out = _mkval_string(string_vals, arity, fill)
    return out


cdef _mkval_string(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        if string_vals.size() > 0:
            return string_vals.at(0)
        else:
            return fill
    else:
        return _mkval_string_multi(string_vals, arity, fill)


cdef _mkval_string_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(string_vals.at(i))
        else:
            out.append(fill)
    return out


cdef _mkval_double(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        out = _mkval_double_single(string_vals, fill)
    else:
        out = _mkval_double_multi(string_vals, arity, fill)
    return out


cdef _mkval_double_single(vector[string]& string_vals, object fill):
    cdef double v
    if string_vals.size() > 0:
        return atof(string_vals.at(0).c_str())
    return fill


cdef _mkval_double_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atof(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


cdef _mkval_long(vector[string]& string_vals, int arity, object fill):
    if arity == 1:
        out = _mkval_long_single(string_vals, fill)
    else:
        out = _mkval_long_multi(string_vals, arity, fill)
    return out


cdef _mkval_long_single(vector[string]& string_vals, object fill):
    if string_vals.size() > 0:
        return atol(string_vals.at(0).c_str())
    return fill


cdef _mkval_long_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(atol(string_vals.at(i).c_str()))
        else:
            out.append(fill)
    return out


def itercalldata(vcf_fns, region, samples, ploidy, fields, arities, fills,
                 format_types, condition, truncate):
    """Iterate over call data (genotypes, etc.) returning tuples suitable for
    loading into a numpy structured array."""

    # force bytes
    vcf_fns = _b(tuple(vcf_fns))
    region = region
    samples = _b(tuple(samples))
    fields = _b(tuple(fields))
    arities = tuple(arities)
    fills = tuple(fills)
    format_types = tuple(format_types)

    # zip up field parameters
    fieldspec = tuple(zip(fields, arities, fills, format_types))

    if condition is None:
        return _itercalldata(vcf_fns, region, samples, ploidy, fieldspec,
                             truncate)
    else:
        return _itercalldata_with_condition(vcf_fns, region, samples, ploidy,
                                            fieldspec, condition, truncate)



def _itercalldata(vcf_fns, region, samples, ploidy, fieldspec, truncate):
    cdef VariantCallFile *variant_file
    cdef Variant *variant

    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        variant_file.parseInfo = False
        variant_file.parseSamples = True
        region_start, region_stop = None, None
        if region is not None:
            region_set = variant_file.setRegion(_b(region))
            if not region_set:
                raise StopIteration
            if ':' in region:
                _, region_start_stop = region.split(':')
                region_start, region_stop = [int(v) for v in
                                             region_start_stop.split('-')]
        variant = new Variant(deref(variant_file))

        while _get_next_variant(variant_file, variant):
            if region_start is not None and truncate and \
                    variant.position < region_start:
                continue
            yield _mkcrow(variant, samples, ploidy, fieldspec)

        del variant_file
        del variant


def _itercalldata_with_condition(vcf_fns, region, samples, ploidy, fieldspec,
                                 condition, truncate):
    cdef VariantCallFile *variant_file
    cdef Variant *variant
    cdef long i = 0
    cdef long n = len(condition)

    for vcf_fn in vcf_fns:
        variant_file = new VariantCallFile()
        variant_file.open(vcf_fn)
        variant_file.parseInfo = False
        variant_file.parseSamples = False
        region_start, region_stop = None, None
        if region is not None:
            region_set = variant_file.setRegion(_b(region))
            if not region_set:
                raise StopIteration
            if ':' in region:
                _, region_start_stop = region.split(':')
                region_start, region_stop = [int(v) for v in
                                             region_start_stop.split('-')]
        variant = new Variant(deref(variant_file))

        while i < n:
            # only worth parsing samples if we know we want the variant
            if condition[i]:
                variant_file.parseSamples = True
                if not _get_next_variant(variant_file, variant):
                    break
                if region_start is not None and truncate and \
                        variant.position < region_start:
                    continue
                yield _mkcrow(variant, samples, ploidy, fieldspec)
            else:
                variant_file.parseSamples = False
                if not _get_next_variant(variant_file, variant):
                    break
            i += 1

        del variant_file
        del variant


cdef _mkcrow(Variant *variant, tuple samples, int ploidy, tuple fieldspec):
    out = [_mkcvals(variant, s, ploidy, fieldspec) for s in samples]
    return tuple(out)


cdef _mkcvals(Variant *variant, string sample, int ploidy, tuple fieldspec):
    out = [_mkcval(variant.samples[sample], ploidy, f, arity, fill, format_type)
           for (f, arity, fill, format_type) in fieldspec]
    return tuple(out)


cdef _mkcval(map[string, vector[string]]& sample_data, int ploidy,
             string field, int arity, object fill, int format_type):
    if field == FIELD_NAME_IS_CALLED:
        return _is_called(sample_data)
    elif field == FIELD_NAME_IS_PHASED:
        return _is_phased(sample_data)
    elif field == FIELD_NAME_GENOTYPE:
        return _genotype(sample_data, ploidy, fill)
    elif field == FIELD_NAME_GENOTYPE_AC:
        return _genotype_ac(sample_data, arity, fill)
    elif field == FIELD_NAME_PLOIDY:
        return _ploidy(sample_data, fill)
    else:
        return _mkval(sample_data[field], arity, fill, format_type)


cdef _is_called(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return gts.at(0).find(b'.') == npos


cdef _is_phased(map[string, vector[string]]& sample_data):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return False
    else:
        return gts.at(0).find(b'|') != npos


cdef _genotype(map[string, vector[string]]& sample_data, int ploidy, fill):
    cdef vector[string] *gts
    cdef vector[int] alleles
    cdef vector[string] allele_strings
    cdef int i
    cdef int allele
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        if ploidy == 1:
            return fill
        else:
            return (fill,) * ploidy
    else:
        split(gts.at(0), GT_DELIMS, allele_strings)
        if ploidy == 1:
            if allele_strings.size() > 0:
                s = allele_strings.at(0)
                if s == b'.':
                    return -1
                else:
                    return atoi(allele_strings.at(0).c_str())
            else:
                return -1
        else:
            for i in range(ploidy):
                if i < allele_strings.size():
                    s = allele_strings.at(i)
                    if s == b'.':
                        alleles.push_back(-1)
                    else:
                        alleles.push_back(atoi(allele_strings.at(i).c_str()))
                else:
                    alleles.push_back(-1)
            return tuple(alleles)


cdef _ploidy(map[string, vector[string]]& sample_data, fill):
    cdef vector[string] *gts
    gts = &sample_data[FIELD_NAME_GT]
    cdef vector[string] allele_strings
    if gts.size() == 0:
        return fill
    else:
        split(gts.at(0), GT_DELIMS, allele_strings)
        return allele_strings.size()


cdef _genotype_ac(map[string, vector[string]]& sample_data, int arity, fill):
    cdef vector[string] *gts
    cdef int i
    cdef vector[string] allele_strings
    gts = &sample_data[FIELD_NAME_GT]
    if gts.size() == 0:
        return (fill,) * arity
    else:
        gac = [0] * arity
        split(gts.at(0), GT_DELIMS, allele_strings)
        for i in range(allele_strings.size()):
            s = allele_strings.at(i)
            if s != b'.':
                allele = atoi(s.c_str())
                if allele < arity:
                    gac[allele] += 1
        return tuple(gac)


def itervariantstable(vcf_fns, region, fields, arities, info_types, parse_info,
                      filter_ids, flatten_filter, fill, flatteners):

    # force bytes
    vcf_fns = _b(tuple(vcf_fns))
    region = _b(region)
    fields = _b(tuple(fields))
    arities = tuple(arities)
    info_types = tuple(info_types)
    filter_ids = _b(tuple(filter_ids))
    fill = _b(fill)
    flatteners = tuple(flatteners)
    debug(flatteners)

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
                             flatten_filter, fill, flatteners)

        del variant_file
        del variant


cdef _mkvtblrow(Variant *variant, tuple fields, tuple arities,
                tuple info_types, tuple filter_ids, bint flatten_filter,
                object fill, tuple flatteners):
    out = list()
    cdef string field
    cdef string flt
    for field, arity, vcf_type, flattener in zip(fields, arities, info_types,
                                                 flatteners):
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
                val = b','.join(variant.alt)
                out.append(val)
        elif field == FIELD_NAME_QUAL:
            out.append(variant.quality)
        elif field == FIELD_NAME_FILTER:
            flt = variant.filter
            if flatten_filter:
                out.extend(_mkfilterval(variant, filter_ids))
            elif flt == b'.':
                out.append(fill)
            else:
                out.append(flt)
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
                elif flattener is not None:
                    _, t = flattener
                    vals = t(variant.info[field])
                    out.extend(vals)
                elif variant.info[field].size() == 0:
                    out.append(fill)
                else:
                    out.append(b','.join(variant.info[field]))
    # force back to str
    return _s(tuple(out))


cdef _mktblval_multi(vector[string]& string_vals, int arity, object fill):
    cdef int i
    out = list()
    for i in range(arity):
        if i < string_vals.size():
            out.append(string_vals.at(i))
        else:
            out.append(fill)
    return out
