import typing

import traitlets

T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')
T3 = typing.TypeVar('T3')
T4 = typing.TypeVar('T4')

T = typing.TypeVar('T')
K = typing.TypeVar('K')
V = typing.TypeVar('V')


# Define wrappers for traitlets classes.  These simply provide Python type hints
# that correspond to actual instance type that will result after a class is
# instantiated (e.g. Unicode() becomes a string).
#
# This allows PyCharm style type hinting to resolve types properly.
def Float(*args, **kw) -> float:
    return traitlets.Float(*args, **kw)


def CFloat(*args, **kw) -> float:
    return traitlets.CFloat(*args, **kw)


def Int(*args, **kw) -> int:
    return traitlets.Int(*args, **kw)


def Bool(*args, **kw) -> bool:
    return traitlets.Bool(*args, **kw)


def Enum(options: typing.List[T], **kw) -> T:
    return traitlets.Enum(options, **kw)


def List(klass: T, **kw) -> typing.List[T]:
    return traitlets.List(klass, **kw)


def Set(klass: T, **kw) -> typing.Set[T]:
    return traitlets.Set(klass, **kw)


# N.B. traitlets.Dict does not check key types.
def Dict(val_class: V, **kw) -> typing.Dict[typing.Any, V]:
    return traitlets.Dict(val_class, **kw)


def Tuple1(a: T1) -> typing.Tuple[T1]:
    return traitlets.Tuple(a)


def Tuple2(a: T1, b: T2) -> typing.Tuple[T1, T2]:
    return traitlets.Tuple(a, b)


def Unicode(*args, **kw) -> str:
    return traitlets.Unicode(*args, **kw)


def Instance(klass: T, **kw) -> T:
    return traitlets.Instance(klass, **kw)


def Array(**kw):
    import numpy
    return Instance(numpy.ndarray, **kw)


def DataFrameType(**kw):
    import pandas
    return Instance(pandas.DataFrame, **kw)


def Any(**kw) -> typing.Any:
    return traitlets.Any(**kw)


# Just a direct copy for now to provide a consistent interface.
HasTraits = traitlets.HasTraits
