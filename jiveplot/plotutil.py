# parsers.py included plots.py which included parsers.py which ...
# leading to undefined module attribute(s).
# Extract all the info from plots.py that parsers.py also needs
# and put in here such that both modules can just import this one
# system import(s)
import copy
import operator

# extensions
import numpy

# own stuff
from .           import (jenums, enumerations)
from .label_v6   import label
from .functional import (filter_, reduce)

CP       = copy.deepcopy
FLAG     = jenums.Flagstuff
YTypes   = enumerations.Enum("amplitude", "phase", "real", "imag", "weight")

class minidataset(object):
    __slots__ = ('xval', 'yval', 'xlims', 'ylims', 'xval_f', 'yval_f', 'extra')
    def __init__(self, xv, yv, xf, yf, xl, yl):
        self.xval   = xv
        self.yval   = yv
        self.xval_f = xf
        self.yval_f = yf
        self.xlims  = xl
        self.ylims  = yl

## glish had a very nice feature: arrays could have attributes
## it's a convenient way to pass data around. Let's emulate that
## in python
class Dict(dict):
    pass
## done :-) !!!!
## Now you can have:
## >>> l = Dict()
## >>> l['key'] = 42    /* use dict method */
## >>> l.x = 12         /* set attribute x to value 12 */


class plt_dataset(object):
    __slots__ = ('_xval', '_yval', '_xlims', '_ylims', 'isSorted', 'prevFlag', 'n', 'n_nan', '__xval', '__yval', '_m_flagged','_m_unflagged', '_m_nan', '_m_useless', 'extra')
    _xformMap = { list:                      lambda a, m: numpy.ma.MaskedArray(numpy.array(a), mask=m),
                  numpy.ndarray:             lambda a, m: numpy.ma.MaskedArray(a, mask=m),
                  numpy.ma.core.MaskedArray: lambda a, m: numpy.ma.MaskedArray(a, mask=numpy.logical_or(a.mask, False if m is None else m))}


    @property
    def useless(self):
        if self._m_useless is None:
            self._m_useless = "all NaN!" if numpy.all(self._m_nan) else ""
            #rv = []
            #if numpy.all(self._m_nan):
            #    rv.append("all NaN")
            #if numpy.all(self._m_flagged) or not numpy.any(self._m_unflagged):
            #    rv.append("no unflagged data")
            #self._m_useless = ",".join(rv)
        return self._m_useless

    def __init__(self, x, y, m=None):
        #print "Create plt_dataset with: type(x)=",type(x)," type(y)=",type(y)," m=",m
        #print "Create plt_dataset with: dtype(x)=",x.dtype," dtype(y)=",y.dtype," m=",m
        #print "  x.shape=", x.shape, " y.shape=", y.shape
        self._yval        = numpy.ma.array( plt_dataset._xformMap[type(y)](y, m) )
        self._xval        = numpy.ma.array( numpy.array(x), mask=CP(self._yval.mask) )
        # cache the masks so that switching between them is easy
        self._m_flagged   = CP(self._yval.mask)
        self._m_unflagged = ~self._m_flagged
        self._m_nan       = ~numpy.isfinite(self._yval.data)
        self._m_useless   = None
        self.isSorted     = False
        #print "    all(flagged)? ", numpy.all(self._m_flagged)," all(unflagged)? ",numpy.all(self._m_unflagged)," all(NaN)? ",numpy.all(self._m_nan)
        #print "    any(flagged)? ", numpy.any(self._m_flagged)," any(unflagged)? ",numpy.any(self._m_unflagged)," any(NaN)? ",numpy.any(self._m_nan)

    # sort by x-axis value. do that once
    def sort(self):
        if self.isSorted:
            return self
        # sort by x-value!
        idxes             = numpy.argsort(self._xval.data, kind='heapsort')
        self._xval        = self._xval[idxes]
        self._yval        = self._yval[idxes]
        # if any of the masks was an array those need to be remapped as well
        self._m_flagged   = self._m_flagged[idxes]
        self._m_unflagged = self._m_unflagged[idxes]
        self._m_nan       = self._m_nan[idxes]
        self.isSorted = True
        return self

    def getxy(self, m):
        return (numpy.ma.array(self._xval.data, mask=m).compressed(), numpy.ma.array(self._yval.data, mask=m).compressed())

    def prepare_for_display(self, flagSetting):
        # prepare variables for unflagged x+y, flagged x+y and x+y min+max
        (xu, yu, xf, yf) = [None]*4
        (xi, xa, yi, ya) = [list(), list(), list(), list()]
        # Depending on what to show, get those datapoints
        if flagSetting in [FLAG.Unflagged, FLAG.Both]:
            # reset to creation mask [and always block NaN & friends]
            newMask    = numpy.logical_or(self._m_flagged, self._m_nan)
            if not numpy.all(newMask):
                # there are non-flagged entries
                (xu, yu) = self.getxy( newMask )
                xi.append( min(xu) )
                xa.append( max(xu) )
                yi.append( min(yu) )
                ya.append( max(yu) )
        if flagSetting in [FLAG.Flagged, FLAG.Both]:
            newMask   = numpy.logical_or(self._m_unflagged, self._m_nan)
            if not numpy.all(newMask):
                # there are non-flagged entries
                (xf, yf) = self.getxy( newMask )
                xi.append( min(xf) )
                xa.append( max(xf) )
                yi.append( min(yf) )
                ya.append( max(yf) )

        # if the list(s) of min/max don't have any entries there were no points to display at all
        # so only need to test one of 'm
        return None if not xi else minidataset(xu, yu, xf, yf, (min(xi), max(xa)), (min(yi), max(ya)))


# Take two labels and join them - i.e. to go from separate plot/data set labels
# to full data set label
is_not_none = lambda x: x is not None
def join_label(l1, l2):
    def attrval(a):
        # it's ok if _either_ of l1 or l2 has the attribute but not both
        aval = filter_(is_not_none, [getattr(l1, a), getattr(l2, a)])
        if len(aval)>1:
            raise RuntimeError("Duplicate attribute value {0}: {1} {2}".format(a, aval[0], aval[1]))
        return None if len(aval)==0 else aval[0]
    return label(reduce(lambda acc, a: operator.setitem(acc, a, attrval(str(a))) or acc, label._attrs, dict()), label._attrs)

# take one label and split it in two; the plot and data set label, based
# on the contents of the 'inplot' list [the rest ends up in the data set label]
def split_label(l, inplot):
    # two brand new empty labels
    mk_lab = lambda : label({}, [])
    def reductor(pl_dsl, attr):
        v = getattr(l, attr)
        setattr(pl_dsl[0], attr, v) if attr in inplot else setattr(pl_dsl[1], attr, v)
        return pl_dsl
    return reduce(reductor, label._attrs, (mk_lab(), mk_lab()))

def label_splitter(inPlot):
    inDataset = set(label._attrs) - set(inPlot)
    def do_split(l):
        return (label(l, inPlot), label(l, inDataset))
    return do_split
