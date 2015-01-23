from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport PyBytes_Check, PyUnicode_Check, PyList_Check, \
    PyTuple_Check, PyDict_Check


if PY_MAJOR_VERSION < 3:
    string_types = basestring,
else:
    string_types = str,


cpdef object s(object x):
    if x is None:
        return None
    elif PY_MAJOR_VERSION < 3:
        return x
    elif PyBytes_Check(x):
        return x.decode('ascii')
    elif PyList_Check(x):
        return [s(i) for i in x]
    elif PyTuple_Check(x):
        return tuple([s(i) for i in x])
    elif PyDict_Check(x):
        return {s(k): s(v) for k, v in x.items()}
    else:
        return x


cpdef object b(object x):
    if x is None:
        return None
    elif PY_MAJOR_VERSION < 3:
        return x
    elif PyBytes_Check(x):
        return x
    elif PyUnicode_Check(x):
        return x.encode('ascii')
    elif PyList_Check(x):
        return [b(i) for i in x]
    elif PyTuple_Check(x):
        return tuple([b(i) for i in x])
    elif PyDict_Check(x):
        return {b(k): b(v) for k, v in x.items()}
    else:
        return x
