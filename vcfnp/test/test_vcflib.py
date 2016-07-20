from __future__ import print_function, division, absolute_import


from vcfnp.vcflib import PyVariantCallFile, TYPE_INTEGER, TYPE_FLOAT, \
    TYPE_STRING, TYPE_BOOL
from nose.tools import eq_


def test_len():
    vcf = PyVariantCallFile('fixture/sample.vcf')
    eq_(9, len(vcf))


def test_metadata():
    vcf = PyVariantCallFile('fixture/sample.vcf')
    eq_(['NS', 'AN', 'AC', 'DP', 'AF', 'AA', 'DB', 'H2'], vcf.info_ids)
    eq_(['GT', 'GQ', 'DP', 'HQ'], vcf.format_ids)
    eq_(['s50', 'q10'], vcf.filter_ids)
    eq_(TYPE_INTEGER, vcf.info_types['NS'])
    eq_(TYPE_FLOAT, vcf.info_types['AF'])
    eq_(TYPE_STRING, vcf.info_types['AA'])
    eq_(TYPE_BOOL, vcf.info_types['DB'])
    eq_(TYPE_STRING, vcf.format_types['GT'])
    eq_(TYPE_INTEGER, vcf.format_types['GQ'])
    eq_(1, vcf.info_counts['AA'])
    eq_(2, vcf.format_counts['HQ'])
    eq_(['NA00001', 'NA00002', 'NA00003'], vcf.sample_names)


def test_fixed_fields():
    vcf = PyVariantCallFile('fixture/sample.vcf')
    v = next(iter(vcf))  # first variant
    eq_('19', v.CHROM)
    eq_(111, v.POS)
    eq_('.', v.ID)
    eq_('A', v.REF)
    eq_(['C'], v.ALT)
    eq_(9.6, v.QUAL)
    eq_(['.'], v.FILTER)  # split in C++


def test_info():
    vcf = PyVariantCallFile('fixture/sample.vcf')
    v = list(iter(vcf))[4]  # fifth variant
    expect = {'AA': ['T'], 'NS': ['2'], 'DP': ['10'], 'AF': ['0.333', '0.667']}
    eq_(expect, v.INFO)


def test_samples():
    vcf = PyVariantCallFile('fixture/sample.vcf')
    v = list(iter(vcf))[2]  # third variant
    expect = {'GT': ['0|0'], 'HQ': ['51', '51'], 'GQ': ['48'], 'DP': ['1']}
    eq_(expect, v.samples['NA00001'])


def test_region():
    vcf = PyVariantCallFile('fixture/sample.vcf.gz')
    vcf.set_region('20')
    eq_(6, len(vcf))
    vcf.set_region('20:1000000-2000000')
    eq_(4, len(vcf))
    vcf.set_region('20', 1000000, 2000000)
    eq_(4, len(vcf))
