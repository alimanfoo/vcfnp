# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# standard library dependencies
import re
import logging


# internal dependencies
import vcfnp.config as config


logger = logging.getLogger(__name__)
debug = logger.debug


EFF_DEFAULT_DTYPE = [
    ('Effect', 'a33'),
    ('Effect_Impact', 'a8'),
    ('Functional_Class', 'a8'),
    ('Codon_Change', 'a7'),  # N.B., will lose information for indels
    ('Amino_Acid_Change', 'a8'),  # N.B., will lose information for indels
    ('Amino_Acid_Length', 'i4'),
    ('Gene_Name', 'a14'),  # N.B., may be too short for some species
    ('Transcript_BioType', 'a20'),
    ('Gene_Coding', 'i1'),
    ('Transcript_ID', 'a20'),
    ('Exon', 'i1')
]


ANN_DEFAULT_DTYPE = [
    ('Allele', 'a12'),
    ('Annotation', 'a34'),
    ('Annotation_Impact', 'a8'),
    ('Gene_Name', 'a14'),
    ('Gene_ID', 'a14'),
    ('Feature_Type', 'a20'),
    ('Feature_ID', 'a14'),
    ('Transcript_BioType', 'a20'),
    ('Rank', 'i1'),
    ('HGVS_c', 'a12'),
    ('HGVS_p', 'a14'),
    ('cDNA_pos', 'i4'),
    ('cDNA_length', 'i4'),
    ('CDS_pos', 'i4'),
    ('CDS_length', 'i4'),
    ('AA_pos', 'i4'),
    ('AA_length', 'i4'),
    ('Distance', 'i4')
]

config.DEFAULT_INFO_DTYPE['EFF'] = EFF_DEFAULT_DTYPE
config.DEFAULT_VARIANT_ARITY['EFF'] = 1
config.DEFAULT_INFO_DTYPE['ANN'] = ANN_DEFAULT_DTYPE
config.DEFAULT_VARIANT_ARITY['ANN'] = 1


EFF_DEFAULT_FILLS = (b'.', b'.', b'.', b'.', b'.', -1, b'.', b'.', -1, b'.',
                     -1)
ANN_DEFAULT_FILLS = (b'.', b'.', b'.', b'.', b'.', b'.', b'.', b'.', -1, b'.',
                     b'.', -1, -1, -1, -1, -1, -1, -1)


_prog_eff_main = re.compile(b'([^(]+)\(([^)]+)\)')


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
            if match_eff_main is None:
                logging.warning(
                    'match_eff_main is None: vals={}'.format(str(vals[0]))
                )
                return fills
            eff = [match_eff_main.group(1)] \
                + match_eff_main.group(2).split(b'|')
            result = tuple(
                fill if v == b''
                else int(v) if i == 5 or i == 10
                else (1 if v == b'CODING' else 0) if i == 8
                else v
                for i, (v, fill) in enumerate(list(zip(eff, fills))[:11])
            )
            return result
    return _transformer


def _ann_split2(b):
    x, _, y = b.partition(b'/')
    return [x, y]


def ann_default_transformer(fills=ANN_DEFAULT_FILLS):
    """
    Return a simple transformer function for parsing ANN annotations. N.B.,
    ignores all but the first effect.

    """
    def _transformer(vals):
        if len(vals) == 0:
            return fills
        else:
            # ignore all but first effect
            ann = vals[0].split(b'|')
            ann = ann[:11] + _ann_split2(ann[11]) + _ann_split2(ann[12]) + \
                _ann_split2(ann[13]) + ann[14:]
            result = tuple(
                fill if v == b''
                else int(v.partition(b'/')[0]) if i == 8
                else int(v) if 11 <= i < 18
                else v
                for i, (v, fill) in enumerate(list(zip(ann, fills))[:18])
            )
            return result
    return _transformer


config.DEFAULT_TRANSFORMER['EFF'] = eff_default_transformer()
config.DEFAULT_TRANSFORMER['ANN'] = ann_default_transformer()


EFF_FIELDS = tuple(n for n, _ in EFF_DEFAULT_DTYPE)
# noinspection PyRedeclaration
ANN_FIELDS = tuple(n for n, _ in ANN_DEFAULT_DTYPE)


def flatten_eff(fill=b'.'):
    def _flatten(vals):
        if len(vals) == 0:
            return [fill] * 11
        else:
            # ignore all but first effect
            match_eff_main = _prog_eff_main.match(vals[0])
            eff = [match_eff_main.group(1)] \
                + match_eff_main.group(2).split(b'|')
            eff = [fill if v == b'' else v for v in eff[:11]]
            return eff
    return _flatten


def flatten_ann(fill=b'.'):
    def _flatten(vals):
        if len(vals) == 0:
            return [fill] * 18
        else:
            # ignore all but first effect
            ann = vals[0].split(b'|')
            ann = ann[:11] + _ann_split2(ann[11]) + _ann_split2(ann[12]) + \
                _ann_split2(ann[13]) + ann[14:]
            ann = [fill if v == b'' else v for v in ann[:18]]
            return ann
    return _flatten


config.DEFAULT_FLATTEN['EFF'] = (EFF_FIELDS, flatten_eff)
config.DEFAULT_FLATTEN['ANN'] = (ANN_FIELDS, flatten_ann)
