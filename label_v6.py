import collections, jenums

AX        = jenums.Axes

_attrl    = [AX.P, AX.CH, AX.SB, AX.FQ, AX.BL, AX.SRC, AX.TIME, AX.TYPE]
_attrf    = [AX.P, AX.CH, AX.SB, AX.FQ, AX.BL, AX.SRC, AX.TIME]
_nattrf   = range(len(_attrf))
BaseClass = collections.namedtuple('BaseClass', _attrl)

_nones    = [None]*len(_attrl)
_defaults = dict((zip(_attrl, _nones)))

fmtStr    = "{1}".format
fmtInt    = "{0}{1}".format
#            AX.P    AX.CH    AX.SB  AX.FQ   AX.BL   AX.SRC  AX.TIME
_fmt      = [fmtStr, fmtInt, fmtInt, fmtStr, fmtStr, fmtStr, fmtStr]
_fmtDict  = dict((zip(_attrf, _fmt)))

def getitem(o, a):
    return o[a]

class label(BaseClass):
    @classmethod
    def format(self, kv):
        return "/".join((_fmtDict[k](k, v) for k, v in kv))

    __slots__ = ()
    _attrs    = set(_attrl)

    def __new__(_cls, kwdict, which):
        accf = getitem if type(kwdict) is dict else getattr
        kwds = dict(((k, accf(kwdict,k)) for k in which))
        return tuple.__new__(_cls, map(kwds.pop, _attrl, _nones))

    def key(self):
        return tuple(((_attrl[n], self[n]) for n in _nattrf if self[n] is not None))

    def attrs(self, which):
        return tuple((w, getattr(self, w)) for w in which)

    def __repr__(self):
        return "/".join( [_fmt[n](_attrf[n], self[n]) for n in _nattrf if self[n] is not None] )
