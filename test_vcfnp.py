"""
Some simple unit tests for the vcfnp extension.

"""


from vcfnp import variants, info, calldata, EFF_DEFAULT_DTYPE, eff_default_transformer
from nose.tools import eq_, assert_almost_equal
import re
import numpy as np


def test_variants():
    a = variants('fixture/sample.vcf', arities={'ALT': 2})
    print repr(a)
    eq_(9, len(a))
    eq_('19', a[0]['CHROM'])
    eq_(111, a[0]['POS'])
    eq_('rs6054257', a[2]['ID'])
    eq_('A', a[0]['REF'])
    eq_('ATG', a[8]['ALT'][1])
    eq_(10.0, a[1]['QUAL'])
    eq_(True, a[2]['FILTER']['PASS'])
    eq_(False, a[3]['FILTER']['PASS'])
    eq_(True, a[3]['FILTER']['q10'])
    eq_(2, a[0]['num_alleles'])
    eq_(False, a[5]['is_snp'])
    
#array([ ('19', 111, '.', 'A', ['C', ''], 9.600000381469727, (False, False, False), 2, True),
#       ('19', 112, '.', 'A', ['G', ''], 10.0, (False, False, False), 2, True),
#       ('20', 14370, 'rs6054257', 'G', ['A', ''], 29.0, (True, False, False), 2, True),
#       ('20', 17330, '.', 'T', ['A', ''], 3.0, (False, True, False), 2, True),
#       ('20', 1110696, 'rs6040355', 'A', ['G', 'T'], 67.0, (True, False, False), 3, True),
#       ('20', 1230237, '.', 'T', ['.', ''], 47.0, (True, False, False), 2, False),
#       ('20', 1234567, 'microsat1', 'G', ['GA', 'GAC'], 50.0, (True, False, False), 3, False),
#       ('20', 1235237, '.', 'T', ['.', ''], 0.0, (False, False, False), 2, False),
#       ('X', 10, 'rsTest', 'AC', ['A', 'ATG'], 10.0, (True, False, False), 3, False)], 
#      dtype=[('CHROM', '|S12'), ('POS', '<i4'), ('ID', '|S12'), ('REF', '|S12'), ('ALT', '|S12', (2,)), ('QUAL', '<f4'), ('FILTER', [('PASS', '|b1'), ('q10', '|b1'), ('s50', '|b1')]), ('num_alleles', '|u1'), ('is_snp', '|b1')])


def test_variants_region():
    a = variants('fixture/sample.vcf.gz', region='20')
    eq_(6, len(a))
    
    
def test_variants_count():
    a = variants('fixture/sample.vcf', count=3)
    eq_(3, len(a))
    
    
def test_variants_exclude_fields():
    a = variants('fixture/sample.vcf', exclude_fields=['ID', 'FILTER'])
    assert 'CHROM' in a.dtype.names
    assert 'ID' not in a.dtype.names
    assert 'FILTER' not in a.dtype.names
    
    
def test_variants_slice():
    a = variants('fixture/sample.vcf.gz')
    eq_('rs6054257', a['ID'][2])
    a = variants('fixture/sample.vcf.gz', slice=(0, None, 2))
    eq_('rs6054257', a['ID'][1])
    
    
def test_info():
    a = info('fixture/sample.vcf', arities={'AC': 2})
    print repr(a)
    eq_(3, a[2]['NS'])
    eq_(.5, a[2]['AF'])
    eq_(True, a[2]['DB'])
    eq_((3, 1), tuple(a[6]['AC']))

#array([(0, 0, [0, 0], 0, 0.0, '', False, False),
#       (0, 0, [0, 0], 0, 0.0, '', False, False),
#       (3, 0, [0, 0], 14, 0.5, '', True, True),
#       (3, 0, [0, 0], 11, 0.017000000923871994, '', False, False),
#       (2, 0, [0, 0], 10, 0.3330000042915344, 'T', True, False),
#       (3, 0, [0, 0], 13, 0.0, 'T', False, False),
#       (3, 6, [3, 1], 9, 0.0, 'G', False, False),
#       (0, 0, [0, 0], 0, 0.0, '', False, False),
#       (0, 0, [0, 0], 0, 0.0, '', False, False)], 
#      dtype=[('NS', '<i4'), ('AN', '<i4'), ('AC', '<i4', (2,)), ('DP', '<i4'), ('AF', '<f4'), ('AA', '|S12'), ('DB', '|b1'), ('H2', '|b1')])


def test_calldata():
    a = calldata('fixture/sample.vcf')
    print repr(a)
    eq_('0|0', a[0]['NA00001']['GT'])
    eq_(True, a[0]['NA00001']['is_called'])
    eq_(True, a[0]['NA00001']['is_phased'])
    eq_((0, 0), tuple(a[0]['NA00001']['genotype']))
    eq_((-1, -1), tuple(a[6]['NA00003']['genotype']))
    eq_((-1, -1), tuple(a[7]['NA00003']['genotype']))
    eq_((10, 10), tuple(a[0]['NA00001']['HQ']))

#>>> a['NA00001']
#array([(True, True, [0, 0], '0|0', 0, 0, [10, 10]),
#       (True, True, [0, 0], '0|0', 0, 0, [10, 10]),
#       (True, True, [0, 0], '0|0', 48, 1, [51, 51]),
#       (True, True, [0, 0], '0|0', 49, 3, [58, 50]),
#       (True, True, [1, 2], '1|2', 21, 6, [23, 27]),
#       (True, True, [0, 0], '0|0', 54, 0, [56, 60]),
#       (True, False, [0, 1], '0/1', 0, 4, [0, 0]),
#       (True, False, [0, 0], '0/0', 0, 0, [0, 0]),
#       (True, False, [0, -1], '0', 0, 0, [0, 0])], 
#      dtype=[('is_called', '|b1'), ('is_phased', '|b1'), ('genotype', '|i1', (2,)), ('GT', '|S3'), ('GQ', '|u1'), ('DP', '<u2'), ('HQ', '<i4', (2,))])
    
    
def test_condition():
    V = variants('fixture/sample.vcf')
    eq_(9, len(V))
    C = calldata('fixture/sample.vcf', condition=V['FILTER']['PASS'])
    eq_(5, len(C))
    I = info('fixture/sample.vcf', condition=V['FILTER']['PASS'])
    eq_(5, len(I))
    Vf = variants('fixture/sample.vcf', condition=V['FILTER']['PASS'])
    eq_(5, len(Vf))
    

def test_variable_calldata():
    C = calldata('fixture/test1.vcf')
    eq_((1, 0), tuple(C['test2']['AD'][0]))
    eq_((1, 0), tuple(C['test2']['AD'][1]))
    eq_((1, 0), tuple(C['test2']['AD'][2]))
    eq_('.', C['test2']['GT'][0])
    eq_('0', C['test2']['GT'][1])
    eq_('1', C['test2']['GT'][2])
    
    
def test_missing_calldata():
    C = calldata('fixture/test1.vcf')
    eq_('.', C['test3']['GT'][2])
    eq_((-1, -1), tuple(C['test3']['genotype'][2]))
    eq_('./.', C['test4']['GT'][2])
    eq_((-1, -1), tuple(C['test4']['genotype'][2]))


def test_override_vcf_types():
    I = info('fixture/test4.vcf')
    eq_(0, I['MQ0Fraction'][2])
    I = info('fixture/test4.vcf', vcf_types={'MQ0Fraction': 'Float'})
    assert_almost_equal(0.03, I['MQ0Fraction'][2])


def test_info_transformers():
    I = info('fixture/test12.vcf',
             dtypes={'EFF': EFF_DEFAULT_DTYPE},
             arities={'EFF': 1},
             transformers={'EFF': eff_default_transformer()})

    eq_('STOP_GAINED', I['EFF']['Effect'][0])
    eq_('HIGH', I['EFF']['Effect_Impact'][0])
    eq_('NONSENSE', I['EFF']['Functional_Class'][0])
    eq_('Cag/Tag', I['EFF']['Codon_Change'][0])
    eq_('Q236*', I['EFF']['Amino_Acid_Change'][0])
    eq_(749, I['EFF']['Amino_Acid_Length'][0])
    eq_('NOC2L', I['EFF']['Gene_Name'][0])
    eq_('.', I['EFF']['Transcript_BioType'][0])
    eq_(1, I['EFF']['Gene_Coding'][0])
    eq_('NM_015658', I['EFF']['Transcript_ID'][0])
    eq_(-1, I['EFF']['Exon'][0])

    eq_('NON_SYNONYMOUS_CODING', I['EFF']['Effect'][1])
    eq_('MODERATE', I['EFF']['Effect_Impact'][1])
    eq_('MISSENSE', I['EFF']['Functional_Class'][1])
    eq_('gTt/gGt', I['EFF']['Codon_Change'][1])
    eq_('V155G', I['EFF']['Amino_Acid_Change'][1])
    eq_(-1, I['EFF']['Amino_Acid_Length'][1])
    eq_('PF3D7_0108900', I['EFF']['Gene_Name'][1])
    eq_('.', I['EFF']['Transcript_BioType'][1])
    eq_(-1, I['EFF']['Gene_Coding'][1])
    eq_('rna_PF3D7_0108900-1', I['EFF']['Transcript_ID'][1])
    eq_(1, I['EFF']['Exon'][1])

    eq_('.', I['EFF']['Effect'][2])
    eq_('.', I['EFF']['Effect_Impact'][2])
    eq_('.', I['EFF']['Functional_Class'][2])
    eq_('.', I['EFF']['Codon_Change'][2])
    eq_('.', I['EFF']['Amino_Acid_Change'][2])
    eq_(-1, I['EFF']['Amino_Acid_Length'][2])
    eq_('.', I['EFF']['Gene_Name'][2])
    eq_('.', I['EFF']['Transcript_BioType'][2])
    eq_(-1, I['EFF']['Gene_Coding'][2])
    eq_('.', I['EFF']['Transcript_ID'][2])
    eq_(-1, I['EFF']['Exon'][2])


def test_svlen():
    V = variants('fixture/test13.vcf').view(np.recarray)
    assert hasattr(V, 'svlen')
    eq_(0, V.svlen[0])
    eq_(1, V.svlen[1])
    eq_(-1, V.svlen[2])
    eq_(3, V.svlen[3])
    eq_(3, V.svlen[4])
    V = variants('fixture/test13.vcf', arities={'svlen': 2}).view(np.recarray)
    assert hasattr(V, 'svlen')
    eq_((3, 0), tuple(V.svlen[3]))
    eq_((3, -2), tuple(V.svlen[4]))


def test_duplicate_field_definitions():
    I = info('fixture/test10.vcf')
    # should not raise, but print useful message to stderr
    C = calldata('fixture/test10.vcf')
    # should not raise, but print useful message to stderr


def test_missing_info_definition():
    # INFO field DP not declared in VCF header
    I = info('fixture/test14.vcf', fields=['DP'])
    eq_('14', I[2]['DP'])  # default is string
    I = info('fixture/test14.vcf', fields=['DP'], vcf_types={'DP':'Integer'})
    eq_(14, I[2]['DP'])
    # what about a field which isn't present at all?
    I = info('fixture/test14.vcf', fields=['FOO'])
    eq_('.', I[2]['FOO'])  # default missing value for string field


def test_missing_format_definition():
    # FORMAT field DP not declared in VCF header
    C = calldata('fixture/test14.vcf', fields=['DP'], vcf_types={'DP':'Integer'})
    eq_(1, C[2]['NA00001']['DP'])

