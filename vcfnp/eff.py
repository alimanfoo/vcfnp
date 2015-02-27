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
    ('Amino_Acid_Change', 'a6'),  # N.B., will lose information for indels
    ('Amino_Acid_Length', 'i4'),
    ('Gene_Name', 'a14'),  # N.B., may be too short for some species
    ('Transcript_BioType', 'a20'),
    ('Gene_Coding', 'i1'),
    ('Transcript_ID', 'a20'),
    ('Exon', 'i1')
]


# 'Allele | Annotation | Annotation_Impact | Gene_Name | Gene_ID | Feature_Type | Feature_ID | Transcript_BioType | Rank | HGVS.c | HGVS.p | cDNA.pos / cDNA.length | CDS.pos / CDS.length | AA.pos / AA.length | Distance | ERRORS / WARNINGS / INFO'
ANN_DEFAULT_DTYPE = [
    ('Allele', 'a12'),
    ('Annotation', 'a34'),
]

config.DEFAULT_INFO_DTYPE['EFF'] = EFF_DEFAULT_DTYPE
config.DEFAULT_VARIANT_ARITY['EFF'] = 1


EFF_DEFAULT_FILLS = (b'.', b'.', b'.', b'.', b'.', -1, b'.', b'.', -1, b'.', -1)


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


config.DEFAULT_TRANSFORMER['EFF'] = eff_default_transformer()


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


config.DEFAULT_FLATTEN['EFF'] = (EFF_FIELDS, flatten_eff)
