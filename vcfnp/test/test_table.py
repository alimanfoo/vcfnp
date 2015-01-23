# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_
import logging


from vcfnp.table import VariantsTable
import vcfnp.eff


logger = logging.getLogger(__name__)
debug = logger.debug


def test_tabulate_variants():
    vcf_fn = 'fixture/sample.vcf'

    fields = ('CHROM', 'POS', 'REF', 'ALT', 'FILTER', 'AF')
    tbl = list(VariantsTable(vcf_fn, fields=fields))
    debug(tbl)
    eq_(fields, tbl[0])
    eq_(('19', 111, 'A', 'C', '.', '.'), tbl[1])
    eq_(('20', 17330, 'T', 'A', 'q10', '0.017'), tbl[4])
    eq_(('20', 1110696, 'A', 'G,T', 'PASS', '0.333,0.667'), tbl[5])


def test_tabulate_variants_flatten_filter():
    vcf_fn = 'fixture/sample.vcf'

    fields = ('CHROM', 'POS', 'REF', 'ALT', 'FILTER')
    tbl = list(VariantsTable(vcf_fn, fields=fields, flatten_filter=True))
    expect_fields = ('CHROM', 'POS', 'REF', 'ALT', 'FILTER_q10', 'FILTER_s50',
                     'FILTER_PASS')
    eq_(expect_fields, tbl[0])
    eq_(('19', 111, 'A', 'C', False, False, False), tbl[1])
    eq_(('20', 17330, 'T', 'A', True, False, False), tbl[4])
    eq_(('20', 1110696, 'A', 'G,T', False, False, True), tbl[5])


def test_tabulate_variants_explicit_arity():
    vcf_fn = 'fixture/sample.vcf'

    fields = ('CHROM', 'POS', 'REF', 'ALT', 'AF')
    tbl = list(VariantsTable(vcf_fn, fields=fields,
                             arities={'ALT': 2, 'AF': 2}))
    expect_fields = ('CHROM', 'POS', 'REF', 'ALT_1', 'ALT_2', 'AF_1', 'AF_2')
    eq_(expect_fields, tbl[0])
    eq_(('19', 111, 'A', 'C', '.', '.', '.'), tbl[1])
    eq_(('20', 17330, 'T', 'A', '.', '0.017', '.'), tbl[4])
    eq_(('20', 1110696, 'A', 'G', 'T', '0.333', '0.667'), tbl[5])


def test_tabulate_variants_fill():
    vcf_fn = 'fixture/sample.vcf'

    fields = ('CHROM', 'POS', 'REF', 'ALT', 'FILTER', 'AF')
    tbl = list(VariantsTable(vcf_fn, fields=fields, fill=None))
    debug(tbl)
    eq_(fields, tbl[0])
    eq_(('19', 111, 'A', 'C', None, None), tbl[1])
    eq_(('20', 17330, 'T', 'A', 'q10', '0.017'), tbl[4])
    eq_(('20', 1110696, 'A', 'G,T', 'PASS', '0.333,0.667'), tbl[5])


def test_tabulate_variants_flatten_eff():
    vcf_fn = 'fixture/test12.vcf'

    # test without flattening
    fields = ('CHROM', 'POS', 'REF', 'ALT', 'EFF')
    tbl = list(VariantsTable(vcf_fn, fields=fields, flatteners={'EFF': None}))
    debug(tbl)
    eq_(fields, tbl[0])
    eq_(('1', 889455, 'G', 'A',
         'STOP_GAINED(HIGH|NONSENSE|Cag/Tag|Q236*|749|NOC2L||CODING|'
         'NM_015658|)'), tbl[1])
    eq_(('1', 897062, 'C', 'T',
         'NON_SYNONYMOUS_CODING(MODERATE|MISSENSE|gTt/gGt|V155G||PF3D7_0108900'
         '|||rna_PF3D7_0108900-1|1|1|WARNING_TRANSCRIPT_MULTIPLE_STOP_CODONS'
         ')'), tbl[2])
    eq_(('1', 897063, 'C', 'T', '.'), tbl[3])

    # test with explicit flattening
    fields = ('CHROM', 'POS', 'REF', 'ALT', 'EFF')
    tbl = list(
        VariantsTable(
            vcf_fn,
            fields=fields,
            flatteners={'EFF': (vcfnp.eff.EFF_FIELDS,
                                vcfnp.eff.flatten_eff('NA'))}
        )
    )
    debug(tbl)
    eq_(fields[:4] + ('Effect', 'Effect_Impact', 'Functional_Class',
                      'Codon_Change', 'Amino_Acid_Change', 'Amino_Acid_Length',
                      'Gene_Name', 'Transcript_BioType', 'Gene_Coding',
                      'Transcript_ID', 'Exon'),
        tbl[0])
    eq_(('1', 889455, 'G', 'A', 'STOP_GAINED', 'HIGH', 'NONSENSE',
         'Cag/Tag', 'Q236*', '749', 'NOC2L', 'NA', 'CODING', 'NM_015658',
         'NA'), tbl[1])
    eq_(('1', 897062, 'C', 'T', 'NON_SYNONYMOUS_CODING', 'MODERATE',
         'MISSENSE', 'gTt/gGt', 'V155G', 'NA', 'PF3D7_0108900', 'NA',
         'NA', 'rna_PF3D7_0108900-1', '1'), tbl[2])
    eq_(('1', 897063, 'C', 'T', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
         'NA', 'NA', 'NA', 'NA', 'NA'), tbl[3])

    # test with default flattening
    fields = ('CHROM', 'POS', 'REF', 'ALT', 'EFF')
    tbl = list(VariantsTable(vcf_fn, fields=fields, fill=None))
    debug(tbl)
    eq_(fields[:4] + ('Effect', 'Effect_Impact', 'Functional_Class',
                      'Codon_Change', 'Amino_Acid_Change', 'Amino_Acid_Length',
                      'Gene_Name', 'Transcript_BioType', 'Gene_Coding',
                      'Transcript_ID', 'Exon'),
        tbl[0])
    eq_(('1', 889455, 'G', 'A', 'STOP_GAINED', 'HIGH', 'NONSENSE',
         'Cag/Tag', 'Q236*', '749', 'NOC2L', None, 'CODING', 'NM_015658',
         None), tbl[1])
    eq_(('1', 897062, 'C', 'T', 'NON_SYNONYMOUS_CODING', 'MODERATE',
         'MISSENSE', 'gTt/gGt', 'V155G', None, 'PF3D7_0108900', None, None,
         'rna_PF3D7_0108900-1', '1'), tbl[2])
    eq_(('1', 897063, 'C', 'T', None, None, None, None, None, None, None,
         None, None, None, None), tbl[3])
