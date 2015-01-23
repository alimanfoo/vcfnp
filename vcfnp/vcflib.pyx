from __future__ import print_function, division, absolute_import


from cython.operator cimport dereference as deref
from vcfnp.compat cimport s as _s, b as _b
from collections import namedtuple


# expose these constants to Python
TYPE_INTEGER = FIELD_INTEGER
TYPE_FLOAT = FIELD_FLOAT
TYPE_STRING = FIELD_STRING
TYPE_BOOL = FIELD_BOOL
TYPE_UNKNOWN = FIELD_UNKNOWN
NUMBER_ALLELE = ALLELE_NUMBER
NUMBER_GENOTYPE = GENOTYPE_NUMBER


VariantTuple = namedtuple(
    'Variant',
    ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'samples']
)


cdef class PyVariantCallFile:

    def __cinit__(self, filename):
        self.thisptr = new VariantCallFile()
        self.thisptr.open(_b(filename))

    def __init__(self, filename):
        pass

    def __dealloc__(self):
        del self.thisptr        
        
    def __len__(self):
        cdef Variant var
        cdef long n = 0
        var.setVariantCallFile(self.thisptr)
        while self.thisptr.getNextVariant(var):
            n += 1
        return n

    def __iter__(self):
        cdef Variant *var
        cdef vector[string] filters
        cdef char semicolon = b';'
        var = new Variant(deref(self.thisptr))
        while self.thisptr.getNextVariant(deref(var)):
            # split the filter field here in C++ to avoid having to do it in
            # Python later
            filters = split(var.filter, semicolon)
            v = VariantTuple(
                _s(var.sequenceName),
                var.position,
                _s(var.id),
                _s(var.ref),
                _s(var.alt),
                var.quality,
                _s(filters),
                _s(var.info),
                _s(var.samples)
            )
            yield v
        del var
        
    def set_region(self, *args):
        if len(args) == 1:
            region = _b(args[0])
            self.thisptr.setRegion(region)
        elif len(args) == 3:
            chrom, start, stop = args
            chrom = _b(chrom)
            start = int(start)
            stop = int(stop)
            self.thisptr.setRegion(chrom, start, stop)
        else:
            raise Exception('Either provide a single region string '
                            'or provide chrom, start, stop.')

    property info_ids:
        def __get__(self):
            l = _s(<list>self.thisptr.infoIds())
            return l

    property format_ids:
        def __get__(self):
            l = _s(<list>self.thisptr.formatIds())
            return l

    property filter_ids:
        def __get__(self):
            l = _s(<list>self.thisptr.filterIds())
            return l

    property info_types:
        def __get__(self):
            d = _s(<dict>self.thisptr.infoTypes)
            return d

    property format_types:
        def __get__(self):
            d = _s(<dict>self.thisptr.formatTypes)
            return d

    property info_counts:
        def __get__(self):
            d = _s(<dict>self.thisptr.infoCounts)
            return d

    property format_counts:
        def __get__(self):
            d = _s(<dict>self.thisptr.formatCounts)
            return d

    property parse_samples:
        def __get__(self):
            return <bool>self.thisptr.parseSamples
        def __set__(self, v):
            self.thisptr.parseSamples = v

    property header:
        def __get__(self):
            s = _s(self.thisptr.header)
            return s

    property file_format: # [sic] no camel case
        def __get__(self):
            s = _s(self.thisptr.fileformat)
            return s
        
    property file_date:
        def __get__(self):
            s = _s(self.thisptr.fileDate)
            return s
        
    property source:
        def __get__(self):
            s = _s(self.thisptr.source)
            return s
        
    property reference:
        def __get__(self):
            s = _s(self.thisptr.reference)
            return s
        
    property phasing:
        def __get__(self):
            s = _s(self.thisptr.phasing)
            return s
        
    property sample_names:
        def __get__(self):
            l = _s(<list>self.thisptr.sampleNames)
            return l
