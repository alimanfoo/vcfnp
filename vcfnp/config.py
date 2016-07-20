# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from vcfnp.vcflib import TYPE_BOOL, TYPE_FLOAT, TYPE_INTEGER, TYPE_STRING, \
    TYPE_UNKNOWN


# default configuration
CACHEDIR_SUFFIX = '.vcfnp_cache'


# these are the standard fields in the variants array
STANDARD_VARIANT_FIELDS = (
    'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
    'num_alleles', 'is_snp', 'svlen'
)


TYPESTRING2KEY = {
    'Float': TYPE_FLOAT,
    'Integer': TYPE_INTEGER,
    'String': TYPE_STRING,
    'Flag': TYPE_BOOL,
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
    'CHROM': b'',
    'POS': 0,
    'ID': b'',
    'REF': b'',
    'ALT': b'',
    'QUAL': 0,
    'num_alleles': 0,
    'is_snp': False,
    'svlen': 0,
}


# default mapping from VCF field types to numpy dtypes
DEFAULT_TYPE_MAP = {
    TYPE_FLOAT: 'f4',
    TYPE_INTEGER: 'i4',
    TYPE_STRING: 'a12',
    TYPE_BOOL: 'b1',
    TYPE_UNKNOWN: 'a12'  # leave as string
}


# default mapping from VCF field types to fill values for missing values
DEFAULT_FILL_MAP = {
    TYPE_FLOAT: 0.,
    TYPE_INTEGER: 0,
    TYPE_STRING: b'',
    TYPE_BOOL: False,
    TYPE_UNKNOWN: b''
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
    'genotype_ac': 'i1',
    'ploidy': 'i1',
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
    'genotype_ac': -1,
    'ploidy': -1,
}


DEFAULT_CALLDATA_ARITY = {
    'is_called': 1,
    'is_phased': 1,
    'genotype_ac': 2,  # biallelic
    'ploidy': 1,
    # N.B., set genotype arity to ploidy
    'AD': 2,  # default to biallelic
}


DEFAULT_FLATTEN = dict()
