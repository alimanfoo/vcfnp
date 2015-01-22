from __future__ import print_function, division, absolute_import


import sys
from cython.operator cimport dereference as deref
from collections import namedtuple


# PY2/3 compatibility
PY2 = sys.version_info[0] == 2
if PY2:
    text_type = unicode
    binary_type = str
else:
    text_type = str
    binary_type = bytes


VariantTuple = namedtuple(
    'Variant',
    ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'samples']
)


# expose constants to Python
TYPE_FLOAT = FIELD_FLOAT
TYPE_INTEGER = FIELD_INTEGER
TYPE_BOOL = FIELD_BOOL
TYPE_STRING = FIELD_STRING
TYPE_UNKNOWN = FIELD_UNKNOWN


def _py(x):
    if isinstance(x, binary_type) and not PY2:
        # always pass str to Python
        y = str(x, 'ascii')
    elif isinstance(x, list):
        y = [_py(i) for i in x]
    elif isinstance(x, tuple):
        y = tuple(_py(i) for i in x)
    elif isinstance(x, dict):
        y = {_py(k): _py(v) for (k, v) in x.items()}
    else:
        y = x
    return y


def _c(x):
    if isinstance(x, text_type):
        # always pass bytes to C
        y = x.encode('ascii')
    elif isinstance(x, list):
        y = [_c(i) for i in x]
    elif isinstance(x, tuple):
        y = tuple(_c(i) for i in x)
    elif isinstance(x, dict):
        y = {_c(k): _c(v) for (k, v) in x.items()}
    else:
        y = x
    return y


cdef class PyVariantCallFile:

    def __cinit__(self, filename):
        self.thisptr = new VariantCallFile()
        filename = _c(filename)
        self.thisptr.open(filename)

    def __dealloc__(self):
        del self.thisptr        
        
    def __len__(self):
        cdef Variant var
        var.setVariantCallFile(self.thisptr)
        n = 0
        while self.thisptr.getNextVariant(var):
            n += 1
        return n

    def __iter__(self):
        cdef Variant *var
        cdef vector[string] filters
        cdef char semicolon = b';'
        var = new Variant(deref(self.thisptr))
        while self.thisptr.getNextVariant(deref(var)):
            # split the filter field here in C++ to avoid having to do it in Python later
            filters = split(var.filter, semicolon)
            t = _py([
                var.sequenceName,
                var.position,
                var.id,
                var.ref,
                var.alt,
                var.quality,
                filters,
                var.info,
                var.samples
            ])
            yield VariantTuple(*t)
        del var
        
    def set_region(self, *args):
        if len(args) == 1:
            s = _c(args[0])
            self.thisptr.setRegion(s)
        elif len(args) == 3:
            chrom, start, stop = _c(args)
            start = int(start)
            stop = int(stop)
            self.thisptr.setRegion(chrom, start, stop)
        else:
            raise Exception('either provide a single region string '
                            'or provide chrom, start, stop')

    property info_ids:
        def __get__(self):
            l = _py(<list>self.thisptr.infoIds())
            return l

    property format_ids:
        def __get__(self):
            l = _py(<list>self.thisptr.formatIds())
            return l

    property filter_ids:
        def __get__(self):
            l = _py(<list>self.thisptr.filterIds())
            return l

    property info_types:
        def __get__(self):
            d = _py(<dict>self.thisptr.infoTypes)
            return d

    property format_types:
        def __get__(self):
            d = _py(<dict>self.thisptr.formatTypes)
            return d

    property info_counts:
        def __get__(self):
            d = _py(<dict>self.thisptr.infoCounts)
            return d

    property format_counts:
        def __get__(self):
            d = _py(<dict>self.thisptr.formatCounts)
            return d

    property parse_samples:
        def __get__(self):
            return <bool>self.thisptr.parseSamples
        def __set__(self, v):
            self.thisptr.parseSamples = v

    property header:
        def __get__(self):
            s = _py(self.thisptr.header)
            return s

    property file_format: # [sic] no camel case
        def __get__(self):
            s = _py(self.thisptr.fileformat)
            return s
        
    property file_date:
        def __get__(self):
            s = _py(self.thisptr.fileDate)
            return s
        
    property source:
        def __get__(self):
            s = _py(self.thisptr.source)
            return s
        
    property reference:
        def __get__(self):
            s = _py(self.thisptr.reference)
            return s
        
    property phasing:
        def __get__(self):
            s = _py(self.thisptr.phasing)
            return s
        
    property sample_names:
        def __get__(self):
            l = _py(<list>self.thisptr.sampleNames)
            return l
