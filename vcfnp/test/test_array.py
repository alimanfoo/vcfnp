from __future__ import print_function, division, absolute_import


import os
from nose.tools import eq_, assert_almost_equal, assert_raises
import numpy as np
from numpy.testing import assert_array_equal
import logging


import vcfnp
from vcfnp.array import variants, calldata, calldata_2d


logger = logging.getLogger(__name__)
debug = logger.debug


def test_variants():
    a = variants('fixture/sample.vcf', arities={'ALT': 2, 'AC': 2})
    debug(repr(a))
    eq_(9, len(a))

    eq_(b'19', a[0]['CHROM'])
    eq_(111, a[0]['POS'])
    eq_(b'rs6054257', a[2]['ID'])
    eq_(b'A', a[0]['REF'])
    eq_(b'ATG', a[8]['ALT'][1])
    eq_(10.0, a[1]['QUAL'])
    eq_(True, a[2]['FILTER']['PASS'])
    eq_(False, a[3]['FILTER']['PASS'])
    eq_(True, a[3]['FILTER']['q10'])
    eq_(2, a[0]['num_alleles'])
    eq_(False, a[5]['is_snp'])

    # INFO fields
    eq_(3, a[2]['NS'])
    eq_(.5, a[2]['AF'])
    eq_(True, a[2]['DB'])
    eq_((3, 1), tuple(a[6]['AC']))


def test_variants_flatten_filter():
    a = variants('fixture/sample.vcf', flatten_filter=True)
    debug(a)
    debug(a.dtype)
    eq_(True, a[2]['FILTER_PASS'])
    eq_(False, a[3]['FILTER_PASS'])
    eq_(True, a[3]['FILTER_q10'])


def test_variants_region():
    a = variants('fixture/sample.vcf.gz', region='20')
    eq_(6, len(a))


def test_variants_region_empty():
    a = variants('fixture/sample.vcf.gz', region='18')
    eq_(0, len(a))
    a = variants('fixture/sample.vcf.gz', region='19:113-200')
    eq_(0, len(a))


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
    eq_(b'rs6054257', a['ID'][2])
    a = variants('fixture/sample.vcf.gz', slice_args=(0, None, 2))
    eq_(b'rs6054257', a['ID'][1])


def test_calldata():
    a = calldata('fixture/sample.vcf')
    debug(repr(a))
    eq_(b'0|0', a[0]['NA00001']['GT'])
    eq_(True, a[0]['NA00001']['is_called'])
    eq_(True, a[0]['NA00001']['is_phased'])
    eq_((0, 0), tuple(a[0]['NA00001']['genotype']))
    eq_((-1, -1), tuple(a[6]['NA00003']['genotype']))
    eq_((-1, -1), tuple(a[7]['NA00003']['genotype']))
    eq_((10, 10), tuple(a[0]['NA00001']['HQ']))


def test_genotype_ac():
    a = calldata_2d('fixture/test63.vcf',
                    fields=['GT', 'genotype', 'genotype_ac', 'ploidy'],
                    ploidy=3, arities=dict(genotype_ac=3))
    debug(repr(a))

    # check GT
    expect = np.array([
        [b'0/0', b'0/0/0', b'0'],
        [b'1', b'0/1', b'0/1/2'],
        [b'././.', b'.', b'./3'],
        [b'././.', b'././.', b'././.'],
    ])
    actual = a['GT']
    assert_array_equal(expect, actual)

    # check genotype
    expect = np.array([
        [(0, 0, -1), (0, 0, 0), (0, -1, -1)],
        [(1, -1, -1), (0, 1, -1), (0, 1, 2)],
        [(-1, -1, -1), (-1, -1, -1), (-1, 3, -1)],
        [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1)],
    ])
    actual = a['genotype']
    assert_array_equal(expect, actual)

    # check ploidy
    expect = np.array([
        [2, 3, 1],
        [1, 2, 3],
        [3, 1, 2],
        [-1, -1, -1],
    ])
    actual = a['ploidy']
    assert_array_equal(expect, actual)

    # check genotype_ac
    expect = np.array([
        [(2, 0, 0), (3, 0, 0), (1, 0, 0)],
        [(0, 1, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1)],
    ])
    actual = a['genotype_ac']
    assert_array_equal(expect, actual)


def test_calldata_region():
    a = calldata('fixture/sample.vcf.gz', region='20')
    eq_(6, len(a))


def test_calldata_region_empty():
    a = calldata('fixture/sample.vcf.gz', region='18')
    eq_(0, len(a))
    a = calldata('fixture/sample.vcf.gz', region='19:113-200')
    eq_(0, len(a))


def test_condition():
    v = variants('fixture/sample.vcf')
    eq_(9, len(v))
    c = calldata('fixture/sample.vcf', condition=v['FILTER']['PASS'])
    eq_(5, len(c))
    vf = variants('fixture/sample.vcf', condition=v['FILTER']['PASS'])
    eq_(5, len(vf))


def test_variable_calldata():
    c = calldata('fixture/test1.vcf')
    eq_((1, 0), tuple(c['test2']['AD'][0]))
    eq_((1, 0), tuple(c['test2']['AD'][1]))
    eq_((1, 0), tuple(c['test2']['AD'][2]))
    eq_(b'.', c['test2']['GT'][0])
    eq_(b'0', c['test2']['GT'][1])
    eq_(b'1', c['test2']['GT'][2])


def test_missing_calldata():
    c = calldata('fixture/test1.vcf')

    # first variant, second sample
    eq_(b'.', c['test2']['GT'][0])
    eq_((-1, -1), tuple(c['test2']['genotype'][0]))
    eq_((1, 0), tuple(c['test2']['AD'][0]))  # data are present

    # third variant, third sample
    eq_(b'.', c['test3']['GT'][2])
    eq_((-1, -1), tuple(c['test3']['genotype'][2]))
    eq_((0, 0), tuple(c['test3']['AD'][2]))  # default fill

    # third variant, fourth sample
    eq_(b'./.', c['test4']['GT'][2])
    eq_((-1, -1), tuple(c['test4']['genotype'][2]))
    eq_((0, 0), tuple(c['test4']['AD'][2]))  # default fill


def test_missing_calldata_cleared():
    c = calldata('fixture/test32.vcf')['AC0093-C']

    # first variant, non-missing
    eq_(b'0/0', c['GT'][0])
    eq_((0, 0), tuple(c['genotype'][0]))
    eq_(8, c['DP'][0])
    eq_(3, c['GQ'][0])

    # second variant, missing
    eq_(b'./.', c['GT'][1])
    eq_((-1, -1), tuple(c['genotype'][1]))
    eq_(0, c['DP'][1])  # should be default fill value
    eq_(0, c['GQ'][1])  # should be default fill value


def test_override_vcf_types():
    v = variants('fixture/test4.vcf')
    eq_(0, v['MQ0FractionTest'][2])
    v = variants('fixture/test4.vcf', vcf_types={'MQ0FractionTest': 'Float'})
    assert_almost_equal(0.03, v['MQ0FractionTest'][2], )


def test_variants_transformers_eff():

    def _test(v):

        eq_(b'STOP_GAINED', v['EFF']['Effect'][0])
        eq_(b'HIGH', v['EFF']['Effect_Impact'][0])
        eq_(b'NONSENSE', v['EFF']['Functional_Class'][0])
        eq_(b'Cag/Tag', v['EFF']['Codon_Change'][0])
        eq_(b'Q236*', v['EFF']['Amino_Acid_Change'][0])
        eq_(749, v['EFF']['Amino_Acid_Length'][0])
        eq_(b'NOC2L', v['EFF']['Gene_Name'][0])
        eq_(b'.', v['EFF']['Transcript_BioType'][0])
        eq_(1, v['EFF']['Gene_Coding'][0])
        eq_(b'NM_015658', v['EFF']['Transcript_ID'][0])
        eq_(-1, v['EFF']['Exon'][0])

        eq_(b'NON_SYNONYMOUS_CODING', v['EFF']['Effect'][1])
        eq_(b'MODERATE', v['EFF']['Effect_Impact'][1])
        eq_(b'MISSENSE', v['EFF']['Functional_Class'][1])
        eq_(b'gTt/gGt', v['EFF']['Codon_Change'][1])
        eq_(b'V155G', v['EFF']['Amino_Acid_Change'][1])
        eq_(-1, v['EFF']['Amino_Acid_Length'][1])
        eq_(b'PF3D7_0108900', v['EFF']['Gene_Name'][1])
        eq_(b'.', v['EFF']['Transcript_BioType'][1])
        eq_(-1, v['EFF']['Gene_Coding'][1])
        eq_(b'rna_PF3D7_0108900-1', v['EFF']['Transcript_ID'][1])
        eq_(1, v['EFF']['Exon'][1])

        eq_(b'.', v['EFF']['Effect'][2])
        eq_(b'.', v['EFF']['Effect_Impact'][2])
        eq_(b'.', v['EFF']['Functional_Class'][2])
        eq_(b'.', v['EFF']['Codon_Change'][2])
        eq_(b'.', v['EFF']['Amino_Acid_Change'][2])
        eq_(-1, v['EFF']['Amino_Acid_Length'][2])
        eq_(b'.', v['EFF']['Gene_Name'][2])
        eq_(b'.', v['EFF']['Transcript_BioType'][2])
        eq_(-1, v['EFF']['Gene_Coding'][2])
        eq_(b'.', v['EFF']['Transcript_ID'][2])
        eq_(-1, v['EFF']['Exon'][2])

    varr = variants('fixture/test12.vcf',
                    dtypes={'EFF': vcfnp.eff.EFF_DEFAULT_DTYPE},
                    arities={'EFF': 1},
                    transformers={'EFF': vcfnp.eff.eff_default_transformer()})
    _test(varr)

    # test EFF is included in defaults
    varr = variants('fixture/test12.vcf')
    _test(varr)


def test_variants_transformers_ann():

    def _test(v):

        eq_(b'T', v['ANN']['Allele'][0])
        eq_(b'intergenic_region', v['ANN']['Annotation'][0])
        eq_(b'MODIFIER', v['ANN']['Annotation_Impact'][0])
        eq_(b'AGAP004677', v['ANN']['Gene_Name'][0])
        eq_(b'AGAP004677', v['ANN']['Gene_ID'][0])
        eq_(b'intergenic_region', v['ANN']['Feature_Type'][0])
        eq_(b'AGAP004677', v['ANN']['Feature_ID'][0])
        eq_(b'.', v['ANN']['Transcript_BioType'][0])
        eq_(-1, v['ANN']['Rank'][0])
        eq_(b'.', v['ANN']['HGVS_c'][0])
        eq_(b'.', v['ANN']['HGVS_p'][0])
        eq_(-1, v['ANN']['cDNA_pos'][0])
        eq_(-1, v['ANN']['cDNA_length'][0])
        eq_(-1, v['ANN']['CDS_pos'][0])
        eq_(-1, v['ANN']['CDS_length'][0])
        eq_(-1, v['ANN']['AA_pos'][0])
        eq_(-1, v['ANN']['AA_length'][0])
        eq_(-1, v['ANN']['Distance'][0])

        eq_(b'.', v['ANN']['Allele'][1])
        eq_(b'.', v['ANN']['Annotation'][1])
        eq_(b'.', v['ANN']['Annotation_Impact'][1])
        eq_(b'.', v['ANN']['Gene_Name'][1])
        eq_(b'.', v['ANN']['Gene_ID'][1])
        eq_(b'.', v['ANN']['Feature_Type'][1])
        eq_(b'.', v['ANN']['Feature_ID'][1])
        eq_(b'.', v['ANN']['Transcript_BioType'][1])
        eq_(-1, v['ANN']['Rank'][1])
        eq_(b'.', v['ANN']['HGVS_c'][1])
        eq_(b'.', v['ANN']['HGVS_p'][1])
        eq_(-1, v['ANN']['cDNA_pos'][1])
        eq_(-1, v['ANN']['cDNA_length'][1])
        eq_(-1, v['ANN']['CDS_pos'][1])
        eq_(-1, v['ANN']['CDS_length'][1])
        eq_(-1, v['ANN']['AA_pos'][1])
        eq_(-1, v['ANN']['AA_length'][1])
        eq_(-1, v['ANN']['Distance'][1])

        eq_(b'T', v['ANN']['Allele'][2])
        eq_(b'missense_variant', v['ANN']['Annotation'][2])
        eq_(b'MODERATE', v['ANN']['Annotation_Impact'][2])
        eq_(b'AGAP005273', v['ANN']['Gene_Name'][2])
        eq_(b'AGAP005273', v['ANN']['Gene_ID'][2])
        eq_(b'transcript', v['ANN']['Feature_Type'][2])
        eq_(b'AGAP005273-RA', v['ANN']['Feature_ID'][2])
        eq_(b'VectorBase', v['ANN']['Transcript_BioType'][2])
        eq_(1, v['ANN']['Rank'][2])
        eq_(b'n.17A>T', v['ANN']['HGVS_c'][2])
        eq_(b'p.Asp6Val', v['ANN']['HGVS_p'][2])
        eq_(17, v['ANN']['cDNA_pos'][2])
        eq_(4788, v['ANN']['cDNA_length'][2])
        eq_(17, v['ANN']['CDS_pos'][2])
        eq_(-1, v['ANN']['CDS_length'][2])
        eq_(6, v['ANN']['AA_pos'][2])
        eq_(-1, v['ANN']['AA_length'][2])
        eq_(-1, v['ANN']['Distance'][2])

    varr = variants('fixture/test_ann.vcf',
                    dtypes={'ANN': vcfnp.eff.ANN_DEFAULT_DTYPE},
                    arities={'ANN': 1},
                    transformers={'ANN': vcfnp.eff.ann_default_transformer()})
    _test(varr)

    # test ANN is included in defaults
    varr = variants('fixture/test_ann.vcf')
    _test(varr)


def test_svlen():
    v = variants('fixture/test13.vcf').view(np.recarray)
    assert hasattr(v, 'svlen')
    eq_(0, v.svlen[0])
    eq_(1, v.svlen[1])
    eq_(-1, v.svlen[2])
    eq_(3, v.svlen[3])
    eq_(3, v.svlen[4])
    v = variants('fixture/test13.vcf', arities={'svlen': 2}).view(np.recarray)
    assert hasattr(v, 'svlen')
    eq_((3, 0), tuple(v.svlen[3]))
    eq_((3, -2), tuple(v.svlen[4]))


def test_duplicate_field_definitions():
    variants('fixture/test10.vcf')
    # should not raise, but print useful message to stderr
    calldata('fixture/test10.vcf')
    # should not raise, but print useful message to stderr


def test_missing_info_definition():
    # INFO field DP not declared in VCF header
    v = variants('fixture/test14.vcf', fields=['DP'])
    eq_(b'14', v[2]['DP'])  # default is string
    v = variants('fixture/test14.vcf', fields=['DP'],
                 vcf_types={'DP': 'Integer'})
    eq_(14, v[2]['DP'])
    # what about a field which isn't present at all?
    v = variants('fixture/test14.vcf', fields=['FOO'])
    eq_(b'', v[2]['FOO'])  # default missing value for string field


def test_missing_format_definition():
    # FORMAT field DP not declared in VCF header
    c = calldata('fixture/test14.vcf', fields=['DP'],
                 vcf_types={'DP': 'Integer'})
    eq_(1, c[2]['NA00001']['DP'])


def test_explicit_pass_definition():
    # explicit PASS FILTER definition
    variants('fixture/test16.vcf')
    # should not raise


def test_caching():
    vcf_fn = 'fixture/sample.vcf.gz'

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='variants')
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = variants(vcf_fn, cache=True, verbose=True)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata')
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata(vcf_fn, cache=True, verbose=True)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata_2d')
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata_2d(vcf_fn, cache=True, verbose=True)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)


def test_caching_cachedir():
    vcf_fn = 'fixture/sample.vcf.gz'
    cachedir = 'fixture/custom.vcfnp_cache/foo'

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='variants',
                                        cachedir=cachedir)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = variants(vcf_fn, cache=True, verbose=True, cachedir=cachedir)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata',
                                        cachedir=cachedir)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata(vcf_fn, cache=True, verbose=True, cachedir=cachedir)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata_2d',
                                        cachedir=cachedir)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata_2d(vcf_fn, cache=True, verbose=True, cachedir=cachedir)
    a2 = np.load(cache_fn)
    assert_array_equal(a, a2)


def test_caching_compression():
    vcf_fn = 'fixture/sample.vcf.gz'

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='variants',
                                        compress=True)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = variants(vcf_fn, cache=True, compress_cache=True, verbose=True)
    a2 = np.load(cache_fn)['data']
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata',
                                        compress=True)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata(vcf_fn, cache=True, compress_cache=True, verbose=True)
    a2 = np.load(cache_fn)['data']
    assert_array_equal(a, a2)

    cache_fn = vcfnp.array._mk_cache_fn(vcf_fn, array_type='calldata_2d',
                                        compress=True)
    debug(cache_fn)
    if os.path.exists(cache_fn):
        os.remove(cache_fn)
    a = calldata_2d(vcf_fn, cache=True, compress_cache=True, verbose=True)
    a2 = np.load(cache_fn)['data']
    assert_array_equal(a, a2)


def test_error_handling():

    # try to open a directory
    vcf_fn = '.'
    with assert_raises(ValueError):
        vcfnp.variants(vcf_fn)

    # try to open a file that doesn't exist
    vcf_fn = 'doesnotexist'
    with assert_raises(ValueError):
        vcfnp.variants(vcf_fn)

    # file is nothing like a VCF (has no header etc.)
    vcf_fn = 'fixture/test48a.vcf'
    with assert_raises(RuntimeError):
        vcfnp.variants(vcf_fn)

    # file has mode sample columns than in header row
    vcf_fn = 'fixture/test48b.vcf'
    with assert_raises(RuntimeError):
        vcfnp.calldata(vcf_fn)


def test_truncate():
    # https://github.com/alimanfoo/vcfnp/issues/54

    vcf_fn = 'fixture/test54.vcf.gz'

    # truncate by default
    v = variants(vcf_fn, region='chr1:10-100')
    eq_(2, len(v))
    c = calldata(vcf_fn, region='chr1:10-100')
    eq_(2, len(c))
    c2d = calldata_2d(vcf_fn, region='chr1:10-100')
    eq_(2, len(c2d))

    # don't truncate
    v = variants(vcf_fn, region='chr1:10-100', truncate=False)
    eq_(3, len(v))
    c = calldata(vcf_fn, region='chr1:10-100', truncate=False)
    eq_(3, len(c))
    c2d = calldata_2d(vcf_fn, region='chr1:10-100', truncate=False)
    eq_(3, len(c2d))
