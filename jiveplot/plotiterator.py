## Generate plots of quantities from a measurment set
##
## $Id: plotiterator.py,v 1.25 2017-02-21 09:10:05 jive_cc Exp $
##
## $Log: plotiterator.py,v $
## Revision 1.25  2017-02-21 09:10:05  jive_cc
## HV: * DesS requests normalized vector averaging - complex numbers are first
##       normalized before being averaged. See "help avt" or "help avc".
##
## Revision 1.24  2017-01-27 13:50:28  jive_cc
## HV: * jplotter.py: small edits
##         - "not refresh(e)" => "refresh(e); if not e.plots ..."
##         - "e.rawplots.XXX" i.s.o. "e.plots.XXX"
##     * relatively big overhaul: in order to force (old) pyrap to
##       re-read tables from disk all table objects must call ".close()"
##       when they're done.
##       Implemented by patching the pyrap.tables.table object on the fly
##       with '__enter__' and '__exit__' methods (see "ms2util.opentable(...)")
##       such that all table access can be done in a "with ..." block:
##          with ms2util.opentable(...) as tbl:
##             tbl.getcol('DATA') # ...
##       and then when the with-block is left, tbl gets automagically closed
##
## Revision 1.23  2015-12-09 07:02:11  jive_cc
## HV: * big change! the plotiterators now return a one-dimensional dict
##       of label => dataset. The higher level code reorganizes them
##       into plots, based on the 'new plot' settings. Many wins in this one:
##         - the plotiterators only have to index one level in stead of two
##         - when the 'new plot' setting is changed, we don't have to read the
##           data from disk again [this is a *HUGE* improvement, especially for
##           larger data sets]
##         - the data set expression parser is simpler, it works on the
##           one-dimensional 'list' of data sets and it does not have to
##           flatten/unflatten any more
##     * The code to deal with refreshing the plots has been rewritten a bit
##       such that all necessary steps (re-organizing raw plots into plots,
##       re-running the label processing, re-running the post processing,
##       re-running the min/max processing) are executed only once; when
##       necessary. And this code is now shared between the "pl" command and
##       the "load/store" commands.
##
## Revision 1.22  2015-09-23 12:28:36  jive_cc
## HV: * Lorant S. requested sensible requests (ones that were already in the
##       back of my mind too):
##         - option to specify the data column
##         - option to not reorder the spectral windows
##       Both options are now supported by the code and are triggered by
##       passing options to the "ms" command
##
## Revision 1.21  2015-04-29 14:34:55  jive_cc
## HV: * add support for plotting quantity vs UV-distance
##
## Revision 1.20  2015-04-08 14:34:12  jive_cc
## HV: * Correct checking of wether dataset.[xy] are of the numpy.ndarray
##       persuasion
##
## Revision 1.19  2015-02-16 12:56:53  jive_cc
## HV: * Now that we do our own slicing, found that some of the index limits
##       were off-by-one
##
## Revision 1.18  2015-02-02 08:55:22  jive_cc
## HV: * support for storing/loading plots, potentially manipulating them
##       via arbitrary arithmetic expressions
##     * helpfile layout improved
##
## Revision 1.17  2015-01-09 14:27:57  jive_cc
## HV: * fixed copy-paste error in weight-thresholded quantity-versus-time fn
##     * sped up SOLINT processing by factor >= 2
##     * output of ".... took    XXXs" consistentified & beautified
##     * removed "Need to convert ..." output; the predicted expected runtime
##       was usually very wrong anyway.
##
## Revision 1.16  2015-01-09 00:02:27  jive_cc
## HV: * support for 'solint' - accumulate data in time bins of size 'solint'
##       now also in "xxx vs time" plots. i.e. can be used to bring down data
##       volume by e.g. averaging down to an arbitrary amount of seconds.
##     * solint can now be more flexibly be set using d(ays), h(ours),
##       m(inutes) and/or s(econds). Note that in the previous versions a
##       unitless specification was acceptable, in this one no more.
##
## Revision 1.15  2014-11-28 14:25:04  jive_cc
## HV: * spelling error in variable name ...
##
## Revision 1.14  2014-11-26 14:56:21  jive_cc
## HV: * pycasa autodetection and use
##
## Revision 1.13  2014-05-14 17:35:15  jive_cc
## HV: * if weight threshold is applied this is annotated in the plot
##     * the plotiterators have two implementations now, one with weight
##       thresholding and one without. Until I find a method that is
##       equally fast with/without weight masking
##
## Revision 1.12  2014-05-14 17:02:01  jive_cc
## HV: * Weight thresholding implemented - but maybe I'll double the code
##       to two different functions, one with weight thresholding and one
##       without because weight thresholding is sloooooow
##
## Revision 1.11  2014-05-12 21:27:28  jive_cc
## HV: * IF time was an essential part of a label, its resolution of 1second
##       was not enough - made it 1/100th of a second. So now you can safely
##       plot data sets with individual time stamps even if they're << 1 second
##       apart
##
## Revision 1.10  2014-05-06 14:20:39  jive_cc
## HV: * Added marking capability
##
## Revision 1.9  2014-04-15 07:53:17  jive_cc
## HV: * time averaging now supports 'solint' = None => average all data in
##       each time-range selection bin
##
## Revision 1.8  2014-04-14 21:04:44  jive_cc
## HV: * Information common to all plot- or data set labels is now stripped
##       and displayed in the plot heading i.s.o in the plot- or data set label
##
## Revision 1.7  2014-04-14 14:46:05  jive_cc
## HV: * Uses pycasa.so for table data access waiting for pyrap to be fixed
##     * added "indexr" + scan-based selection option
##
## Revision 1.6  2014-04-10 21:14:40  jive_cc
## HV: * I fell for the age-old Python trick where a default argument is
##       initialized statically - all data sets were integrating into the
##       the same arrays! Nice!
##     * Fixed other efficiency measures: with time averaging data already
##       IS in numarray so no conversion needs to be done
##     * more improvements
##
## Revision 1.5  2014-04-09 08:26:46  jive_cc
## HV: * Ok, moved common plotiterator stuff into baseclass
##
## Revision 1.4  2014-04-08 23:34:13  jive_cc
## HV: * Minor fixes - should be better now
##
## Revision 1.3  2014-04-08 22:41:11  jive_cc
## HV: Finally! This might be release 0.1!
##     * python based plot iteration now has tolerable speed
##       (need to test on 8M row MS though)
##     * added quite a few plot types, simplified plotters
##       (plotiterators need a round of moving common functionality
##        into base class)
##     * added generic X/Y plotter
##
## Revision 1.2  2014-04-02 17:55:30  jive_cc
## HV: * another savegame, this time with basic plotiteration done in Python
##
## Revision 1.1  2013-12-12 14:10:16  jive_cc
## HV: * another savegame. Now going with pythonic based plotiterator,
##       built around ms2util.reducems
##
##
# system import(s)
from   __future__ import print_function

import copy
import math
import time
import operator
import itertools
import collections

from six import iteritems

# extensions
import numpy
import pyrap.quanta

# own stuff
from . import (hvutil, ms2util, jenums, plotutil, functional)

# Auto-detect of pycasa
havePyCasa = True
try:
    import pycasa
    print("*** using PyCasa for measurementset data access ***")
except:
    havePyCasa = False


# imp vs importlib
try:
    import imp
    new_module = imp.new_module
except ImportError:
    import types
    new_module = types.ModuleType


## Introduce some shorthands
NOW        = time.time
CP         = copy.deepcopy
AX         = jenums.Axes
AVG        = jenums.Averaging
YTypes     = plotutil.YTypes
Quantity   = collections.namedtuple('Quantity', ['quantity_name', 'quantity_fn'])
# do not drag in all numpy.* names but by "resolving" them at this level
# we shave off a lot of python name lookups. this has an effect on code which
# is called a lot of times per second - like the code in here
ANY        = numpy.any
ALL        = numpy.all
ADD        = numpy.add
SQRT       = numpy.sqrt
SQUARE     = numpy.square
ARRAY      = numpy.array
MARRAY     = numpy.ma.array
ISFINITE   = numpy.isfinite
LOGICAL_OR = numpy.logical_or
_ArrayT    = numpy.ndarray
_MArrayT   = numpy.ma.core.MaskedArray
IsArray    = lambda x: isinstance(x, _ArrayT) or isinstance(x, _MArrayT) or isinstance(x, list)

# numpy 2.0 API removes a number of things _sigh_
INF        = numpy.inf if hasattr(numpy, 'inf') else numpy.Inf
INT        = numpy.int if hasattr(numpy, 'int') else int

# Useful simple functions

# take the current channel selection, and produce a list of the sorted unique channels
mk_chansel = functional.compose(list, sorted, set, CP)
print_if   = functional.choice(operator.truth, functional.printf, functional.const(None))

# We support different kinds of averaging
def avg_vectornorm(ax):
    def do_it(x):
        # first set all flagged + NaN/Inf values to 0 such that
        #  (1) any NaN/Inf's don't screw up the total sum
        #  (2) flagged data doesn't count towards the sum
        # We're going to need the input-mask twice
        imask    = LOGICAL_OR(~ISFINITE(x.data), x.mask)
        # Sum all values along the requested axis
        x.data[imask] = 0
        total    = numpy.sum(x.data, axis=ax, keepdims=True)
        # figure out where the counts along the requested axis are 0
        # and transform to a mask such that masked values in the *output*
        # may effectively be removed
        flags   = ARRAY(numpy.sum(~imask, axis=ax, keepdims=True)==0, dtype=numpy.bool)
        # Find the maximum unflagged value along the input axis.
        # Flagged data gets set to -inf such that max()
        # may yield a useful result
        mdata        = numpy.abs(x.data)
        mdata[imask] = -numpy.inf
        total       /= numpy.max(mdata, axis=ax, keepdims=True)
        # remove those data points
        total[flags] = numpy.nan
        return MARRAY(total, mask = flags)
    return do_it

def avg_arithmetic(ax):
    # normal arithmetic mean, should work both on complex or scalar data
    def do_it(x):
        # first set all flagged + NaN/Inf values to 0 such that
        #  (1) any NaN/Inf's don't screw up the total sum
        #  (2) flagged data doesn't count towards the sum
        imask  = LOGICAL_OR(~ISFINITE(x.data), x.mask)
        x[ imask ] = 0
        total  = numpy.sum(x.data,  axis=ax, keepdims=True)
        counts = numpy.sum(~imask,  axis=ax, keepdims=True)
        # figure out where the counts are 0 - effectively
        # remove those data points
        nmask  = ARRAY(counts==0, dtype=numpy.bool)
        # we have computed where the count==0 so we can now
        # overwrite with 1 to prevent divide-by-zero errors.
        # Later we'll replace those values with NaN
        counts[nmask]=1
        total       /= counts
        # And indicate where there was no average at all
        total[nmask] = numpy.NaN
        return MARRAY(total, mask = nmask)
    return do_it

def avg_sum(ax):
    # normal arithmetic mean, should work both on complex or scalar data
    def do_it(x):
        # first set all flagged + NaN/Inf values to 0 such that
        #  (1) any NaN/Inf's don't screw up the total sum
        #  (2) flagged data doesn't count towards the sum
        imask   = LOGICAL_OR(~ISFINITE(x.data), x.mask)
        x.data[ imask ] = 0
        # Sum all values along the requested axis
        total   = numpy.sum(x.data, axis=ax, keepdims=True)
        # count unflagged points along that axis and set flag if
        # there aren't any of those
        flags   = ARRAY(numpy.sum(~imask, axis=ax, keepdims=True)==0, dtype=numpy.bool)
        # remove points that didn't have any unflagged data
        total[flags] = numpy.nan
        return MARRAY(total, mask = flags)
    return do_it

def avg_none(_):
    return functional.identity


## The base class holds the actual table object -
## makes sure the selection etc gets done
class plotbase(object):
    def __enter__(self, *args, **kwargs):
        return self
    def __exit__(self, *args, **kwargs):
        if hasattr(self, 'table'):
            self.table.close()

    # depending on combination of query or not and read flags or not
    # we have optimum call sequence for processing a table
    # key = (qryYesNo, readFlagYesNo)
    # I think that executing an empty query
    #   tbl.query('')
    # takes longer than
    #   tbl.query()
    _qrycolmapf = {
            (False, False): lambda tbl, q, c: tbl,                    # no query, no flagcolum reading
            (True , False): lambda tbl, q, c: tbl.query(q),           # only query
            (False, True ): lambda tbl, q, c: tbl.query(columns=c),   # only different columns
            (True,  True ): lambda tbl, q, c: tbl.query(q, columns=c) # the works
        }

    ## selection is a selection object from 'selection.py'
    def __init__(self, msname, selection, mapping, **kwargs):

        self.verbose  = kwargs.setdefault('verbose', True)
        self.flags    = kwargs.get('readflags', True)
        self.datacol  = CP(mapping.domain.column)

        self.table    = pycasa.table(msname) if havePyCasa else ms2util.opentable(msname)
        # when reading flags is enabled let the C++ do the OR'ing of FLAG_ROW, FLAG columns for us
        colnames      = None #"*, (FLAG_ROW || FLAG) AS FLAGCOL" if self.flags else None

        ## apply selection if necessary
        qry = selection.selectionTaQL()
        s = NOW()
        self.table = plotbase._qrycolmapf[(bool(qry), bool(colnames))](self.table, qry, colnames)
        e = NOW()
        if not self.table:
            raise RuntimeError("No data available for your selection criteria")
        if qry and self.verbose:
            print("Query took\t\t{0:.3f}s".format(e-s))

        # we'll provide overrides for specific column readers
        # for efficiency for the WEIGHT and/or FLAG data
        #
        # Subclasses can request 'WEIGHTCOL' and/or 'FLAGCOL'
        # in their call to reducems2(...), provided they pass
        # in the self.slicers{..} object which we'll have prepared
        self.slicers = dict()

        # Set up weight thresholding.
        # We have to be able to deal with the following weight shapes:
        #    numpy.Inf              (weights not read)
        #    (n_int, n_pol)         [WEIGHT column read]
        #    (n_int, n_freq, n_pol) [WEIGHT_SPECTRUM read]
        #
        # In order to turn the weight criterion (if any) into a mask we must
        # potentially broadcast the WEIGHT shape (one weight per polarization)
        # to all channels; the data always has shape:
        #    (n_int, n_freq, n_pol)
        #
        # We can do that efficiently by transposing the data array in that case to be:
        #    (n_freq, n_int, n_pol)
        # Now the data mask also has this shape.
        # Then, numpy.logical_or(data.mask, weight_mask) does the right thing:
        #     weight_mask == (n_int, n_pol)   -> dimensions match on both data.mask and weight_mask
        #                                        but for the first -> broadcasted along dim 0, which
        #                                        is n_freq, i.e. each spectral point gets the same weight per pol
        self.threshold = CP(selection.weightThreshold) if selection.weightThreshold is not None else -INF
        transpose      = weight_rd = None
        if self.threshold == -INF:
            # No weight thresholding? Return an infinite weight to effectively disable thresholding
            weight_rd = lambda _a,_b,_c,_d: INF
            # Also no need to transpose/untranspose the data array for this 'shape'
            transpose = functional.identity
        else:
            # weight read from MS, choose which column to use
            weight_col = 'WEIGHT_SPECTRUM' if 'WEIGHT_SPECTRUM' in self.table.colnames() else 'WEIGHT'
            weight_rd  = lambda tab, _, s, n: tab.getcol(weight_col, startrow=s, nrow=n)
            # the need to transpose/untranspose the data array depends on which is the weight column
            transpose  = operator.methodcaller('transpose', (1,0,2)) if weight_col == 'WEIGHT' else functional.identity

        # install the appropriate weight reader for the 'WEIGHTCOL' column
        self.slicers['WEIGHTCOL'] = weight_rd
        self.transpose            = transpose

        # Set up FLAG reading
        # If self.flags is set, then we take FLAG, FLAG_ROW from the MS,
        # otherwise we override the slicer with something that don't do nothing
        # but return 'not flagged'
        # Because we need to do (FLAG || FLAG_ROW) we play the same transpose
        # trick as with the WEIGHT/WEIGHT_SPECTRUM above only different.
        #  FLAG     = (nrow, nchan, npol)
        #  FLAG_ROW = (nrow,)
        #  so by doing "flag.transpose((1,2,0))" it becomes (nchan, npol, nrow)
        #  and now numpy.logical_or(flag, flag_row) broadcasts Just Fine(tm)!
        # But we only have to do that if we actually read flags from the MS ...
        self.transpose_flag   = operator.methodcaller('transpose', (1,2,0))
        self.untranspose_flag = operator.methodcaller('transpose', (2,0,1))
        if not self.flags:
            # no flags to be read, replace with no-ops
            no_flag   = lambda _a,_b,_c,_d: False
            self.slicers['FLAG']     = no_flag
            self.slicers['FLAG_ROW'] = no_flag
            self.transpose_flag      = self.untranspose_flag = functional.identity

        ## Parse data-description-id selection into a map:
        ## self.ddSelection will be
        ##   map [ DATA_DESC_ID ] => (FQ, SB, POLS)
        ##
        ## With FQ, SB integer - the indices,
        ##      POLS = [ (idx, str), ... ]
        ##        i.e. list of row indices and the polarization string
        ##             to go with it, such that the polarization data
        ##             is put in the correct plot/data set immediately
        ##
        # The matter of the fact is that the polarization row index ('idx'
        # above) is not a unique mapping to physical polarization so we cannot
        # get away with using the numerical label, even though that would be
        # faster
        _pMap    = mapping.polarizationMap
        _spwMap  = mapping.spectralMap
        GETF     = _spwMap.frequenciesOfFREQ_SB
        # Frequencies get done in MHz
        scale    = 1e6 if mapping.domain.domain == jenums.Type.Spectral else 1

        ## if user did not pass DATA_DESC_ID selection, default to all
        if selection.ddSelection:
            ## An element in "ddSelection" is a 4-element tuple with
            ## fields (FQ, SB, POLID, [product indices])
            ## So all we need is to pair the product indices with the
            ## appropriate polarization strings
            GETDDID = _spwMap.datadescriptionIdOfFREQ_SB_POL
            ITEMGET = hvutil.itemgetter
            def ddIdAdder(acc, ddSel):
                (fq, sb, pid, l) = ddSel
                ddId             = GETDDID(fq, sb, pid)
                polStrings       = _pMap.getPolarizations(pid)
                acc[0][ ddId ]   = (fq, sb, functional.zip_(l, ITEMGET(*l)(polStrings)))
                acc[1][ ddId ]   = GETF(fq, sb)/scale
                return acc
            (self.ddSelection, self.ddFreqs)   = functional.reduce(ddIdAdder, selection.ddSelection, [{}, {}])
        else:
            ddids     = _spwMap.datadescriptionIDs()
            UNMAPDDID = _spwMap.unmapDDId
            def ddIdAdder(acc, dd):
                # Our data selection is rather simple: all the rows!
                r         = UNMAPDDID(dd)
                acc[0][ dd ] = (r.FREQID, r.SUBBAND, list(enumerate(_pMap.getPolarizations(r.POLID))))
                acc[1][ dd ] = GETF(r.FREQID, r.SUBBAND)/scale
                return acc
            (self.ddSelection, self.ddFreqs)   = functional.reduce(ddIdAdder, ddids, [{}, {}])

        ## Provide for a label unmapping function.
        ## After creating the plots we need to transform the labels - some
        ## of the numerical indices must be unmapped into physical quantities
        #unmapBL   = mapping.baselineMap.baselineName
        #unmapFQ   = mapping.spectralMap.freqGroupName
        #unmapSRC  = mapping.fieldMap.field

        unmap_f   = { AX.BL:   mapping.baselineMap.baselineName,
                      AX.FQ:   mapping.spectralMap.freqGroupName,
                      AX.SRC:  mapping.fieldMap.field,
                      AX.TIME: lambda t: pyrap.quanta.quantity(t, "s").formatted("time", precision=8) }
        identity  = lambda x: x
        def unmap( fld_val ):
            return (fld_val[0], unmap_f.get(fld_val[0], identity)(fld_val[1]))
        # flds is the list of field names that the values in the tuple mean
        self.MKLAB = lambda flds, tup: plotutil.label( dict(map(unmap, zip(flds, tup))), flds )

    ##
    ##   Should return the generated plots according to the following
    ##   structure:
    ##
    ##   Update: Dec 2015 - we start doing things a little different
    ##                      the raw data sets will be delivered as a dict of
    ##                      Dict: Key -> Value, where Key is the full data set
    ##                      label and Value the dataset() object.
    ##                      The division into plots will be done at a higher
    ##                      level. Reasons:
    ##                        - generation of raw data is faster as only one level
    ##                          of dict indexing is needed i.s.o. two
    ##                        - if user changes the new plot settings, we don't
    ##                          have to read from disk no more, it then is a mere
    ##                          rearrangement of the raw data sets
    ##                        - load/store with expressions on data sets now work
    ##                          on the one-dimensional 'list' of data sets, no need
    ##                          to flatten/unflatten anymore
    ##
    ##   plots = dict( Key -> Value ) with
    ##              Key   = <plot index>  # contains physical quantities/labels
    ##              Value = DataSet
    ##   DataSet = instance of 'dataset' (see below) with
    ##             attributes ".x" and ".y"
    def makePlots(self, *args):
        raise RuntimeError("Someone forgot to implement this function for this plottype")


## Unfortunately, our code relies on the fact that the numarrays returned
## from "ms.getcol()" are 3-dimensional: (nintg x npol x nchannel)
## Sadly, casa is smart/stoopid enough to return no more dimensions
## than are needed; no degenerate axes are present.
## So if your table consists of one row, you get at best a matrix:
##     npol x nchannel
## Further, if you happen to read single-pol data, guess what,
## you get a matrix at best and a vector at worst!:
##    matrix: nintg x nchannel
##    vector: nchannel   (worst case: a table with one row of single pol data!)
##
## m3d() can be used to reshape an array to always be at least 3d,
##   it inserts degenerate axis from the end, assuming that there
##   won't be data sets with only one row ...
##   (single pol does happen! a lot!)
#def m3d(ar):
#    shp = list(ar.shape)
#    while len(shp)<3:
#        shp.insert(-1, 1)
#    return ar.reshape( shp )
#
#def m2d(ar):
#    shp = list(ar.shape)
#    while len(shp)<2:
#        shp.insert(-1, 1)
#    return ar.reshape( shp )
#
#class dataset_org:
#    __slots__ = ['x', 'y', 'n', 'a', 'sf', 'm']
#
#    @classmethod
#    def add_sumy(self, obj, xs, ys, m):
#        obj.y = obj.y + ys
#        obj.n = obj.n + 1
#        obj.m = numpy.logical_and(obj.m, m)
#
#    @classmethod
#    def init_sumy(self, obj, xs, ys, m):
#        obj.x  = numpy.array(xs)
#        obj.y  = numpy.array(ys)
#        obj.sf = dataset_org.add_sumy
#        obj.m  = m
#
#    def __init__(self, x=None, y=None, m=None):
#        if x is not None and len(x)!=len(y):
#            raise RuntimeError("attempt to construct data set where len(x) != len(y)?!!!")
#        self.x  = list() if x is None else x
#        self.y  = list() if y is None else y
#        self.m  = list() if m is None else m
#        self.n  = 0 if x is None else 1
#        self.sf = dataset_org.init_sumy if x is None else dataset_org.add_sumy
#        self.a  = False
#
#    def append(self, xv, yv, m):
#        self.x.append(xv)
#        self.y.append(yv)
#        self.m.append(m)
#
#    # integrate into the current buffer
#    def sumy(self, xs, ys, m):
#        self.sf(self, xs, ys, m)
#
#    def average(self):
#        if not self.a and self.n>1:
#            self.y = self.y / self.n
#        self.a = True
#
#    def is_numarray(self):
#        return (type(self.x) is numpy.ndarray and type(self.y) is numpy.ndarray)
#
#    def as_numarray(self):
#        if self.is_numarray():
#            return self
#        # note to self: float32 has insufficient precision for e.g.
#        # <quantity> versus time
#        self.x  = numpy.array(self.x, dtype=numpy.float64)
#        self.y  = numpy.array(self.y, dtype=numpy.float64)
#        self.m  = numpy.array(self.m, dtype=numpy.bool)
#        return self
#
#    def __str__(self):
#        return "DATASET: {0} MASK: {1}".format(zip(self.x, self.y), self.m)
#
#    def __repr__(self):
#        return str(self)

class dataset_fixed:
    __slots__ = ['x', 'y', 'm']

    def __init__(self, x=None, y=None):
        if x is not None and len(x)!=len(y):
            raise RuntimeError("attempt to construct data set where len(x) != len(y)?!!!")
        self.x  = list() if x is None else x
        self.y  = list() if y is None else y
        self.m  = False

    def append(self, xv, yv, m):
        raise NotImplemented("append() does not apply to dataset_fixed!")

    # integrate into the current buffer
    def sumy(self, xs, ys, m):
        raise NotImplemented("sumy() does not apply to dataset_fixed!")

    def average(self, method):
        raise NotImplemented("average() does not apply to dataset_fixed!")

    def is_numarray(self):
        return type(self) is _ArrayT and type(self.y) is _MArrayT

    def as_numarray(self):
        if self.is_numarray():
            return self
        # note to self: float32 has insufficient precision for e.g.
        # <quantity> versus time
        self.x  = ARRAY(self.x, dtype=numpy.float64)
        if type(self.y) is not _MArrayT: #numpy.ma.MaskedArray:
            self.y  = MARRAY(self.y, mask=~ISFINITE(self.y), dtype=numpy.float64)
        return self

    def __str__(self):
        return "DATASET<fixed>: {0}".format(functional.zip_(self.x, self.y))

    def __repr__(self):
        return str(self)

#################################################################################
#
# .append() means append to list, fastest for collecting individual samples
# .average() verifies that no averaging is requested - this one can't handle that
#
#################################################################################
class dataset_list:
    __slots__ = ['x', 'y', 'n', 'a', 'm']

    @classmethod
    def add_sumy(self, obj, xs, ys, m):
        obj.y = obj.y + ys
        obj.n = obj.n + 1
        obj.m = numpy.logical_and(obj.m, m)

    @classmethod
    def init_sumy(self, obj, xs, ys, m):
        obj.x  = numpy.array(xs)
        obj.y  = numpy.array(ys)
        obj.sf = dataset_list.add_sumy
        obj.m  = m

    def __init__(self, x=None, y=None, m=None):
        if x is not None and len(x)!=len(y):
            raise RuntimeError("attempt to construct data set where len(x) != len(y)?!!!")
        self.x  = list() if x is None else x
        self.y  = list() if y is None else y
        self.m  = list() if m is None else m
        self.n  = 0 if x is None else 1
        self.a  = False

    def append(self, xv, yv, m):
        self.x.append(xv)
        self.y.append(yv)
        self.m.append(m)

    def extend(self, xseq, yseq, mseq):
        self.x.extend(xseq)
        self.y.extend(yseq)
        self.m.extend(mseq)

    def average(self, method):
        if method != AVG.NoAveraging:
            raise RuntimeError("dataset_list was not made for time averaging")

    def is_numarray(self):
        return type(self.x) is _ArrayT and type(self.y) is _MArrayT

    def as_numarray(self):
        if self.is_numarray():
            return self
        # note to self: float32 has insufficient precision for e.g.
        # <quantity> versus time
        self.x  = ARRAY(self.x, dtype=numpy.float64)
        self.y  = MARRAY(self.y, mask=self.m, dtype=numpy.float64)
        return self

    def __str__(self):
        return "DATASET<list>: len(x)={0}, len(y)={1} len(m)={2}".format(len(self.x), len(self.y), len(self.m))

    def __repr__(self):
        return str(self)

#################################################################################
#
# Specialization for holding one (1) spectrum. x-axis = channel numbers
# The .add_y method may be called only once
#
#################################################################################
#class dataset_chan_wtf:
#    __slots__ = ['x', 'y', 'sf']
#
#    @classmethod
#    def add_sumy(self, *_):
#        raise RuntimeError("dataset_chan was not meant to integrate > 1 spectrum")
#
#    @classmethod
#    def init_sumy(self, obj, xs, ys, m):
#        obj.x  = ARRAY(xs)
#        obj.y  = MARRAY(ys, mask=m)
#        obj.sf = dataset_chan.add_sumy
#
#    def __init__(self):
#        self.x  = self.y = None
#        self.sf = dataset_chan_wtf.init_sumy
#
#    def append(self, xv, yv, m):
#        raise NotImplemented("dataset_chan not meant for appending")
#
#    def add_y(self, x,v, yv, m):
#        self.sf(self, xv, yv, m)
#
#    def average(self, method):
#        if method != AVG.NoAveraging:
#            raise RuntimeError("dataset_chan was not meant for averaging!")
#
#    def is_numarray(self):
#        return (type(self.x) is numpy.array and type(self.y) is numpy.ma.MaskedArray)
#
#    def as_numarray(self):
#        if self.is_numarray():
#            return self
#        # note to self: float32 has insufficient precision for e.g.
#        # <quantity> versus time
#        self.x  = numpy.array(self.x, dtype=numpy.float64)
#        self.y  = numpy.ma.MaskedArray(self.y, mask=self.m, dtype=numpy.float64)
#        return self
#
#    def __str__(self):
#        return "DATASET<chan>: len(x)={0}, len(y)={1} len(m)={2}".format(len(self.x), len(self.y), len(self.m))
#
#    def __repr__(self):
#        return str(self)

# Specialization for (potentially) averaging > 1 spectrum when solint'ing
class dataset_chan:
    __slots__ = ['x', 'y', 'm', 'sf', 'af']

    # 2nd call to .add_y() means that we have to transform
    # the mask to integers and remove NaN/Inf (masked) values from
    # the previously stored y-values in order to make sure that those
    # don't screw up the totals
    @classmethod
    def add_sumy_first(self, obj, xs, ys, m):
        # set masked values to 0 and convert mask to counts in existing object
        obj.y[ obj.m ] = 0
        obj.m          = ARRAY(~obj.m, dtype=INT)
        # from now on, averaging has to do something
        obj.af         = dataset_chan.average_n
        # from now on extra .add_y() calls will do something slight different
        obj.sf         = dataset_chan.add_sumy
        # use the new .add_y() to do the "integration" for us
        obj.sf(obj, xs, ys, m)

    # before 'integrating' the y-values we must
    # make sure no NaN/Inf values are present
    # because a single NaN/Inf in a channel makes
    # the whole summation for that channel go NaN/Inf.
    # The reading process auto-flags points which have
    # Inf/NaN so we can just set flagged values to 0
    @classmethod
    def add_sumy(self, obj, xs, ys, m):
        ys[ m ] = 0
        obj.y = obj.y + ys
        obj.m = obj.m + ARRAY(~m, dtype=INT) # transform mask into counts

    # very first call to .add_y() just store the parameters
    # no fancy processing
    @classmethod
    def init_sumy(self, obj, xs, ys, m):
        obj.x  = ARRAY(xs)
        obj.y  = ARRAY(ys)
        obj.m  = ARRAY(m) #ARRAY(~m, dtype=INT) # tranform mask into counts
        obj.sf = dataset_chan.add_sumy_first
        obj.af = dataset_chan.average_noop

    @classmethod
    def average_empty(self, obj, method):
        raise RuntimeError("dataset_chan: attempt to average uninitialized dataset (.add_y() never called)")

    @classmethod
    def average_noop(self, obj, method):
        # nothing to average, really
        obj.af = None

    @classmethod
    def average_n(self, obj, method):
        # normal average = arithmetic mean i.e. summed value / count of valid values
        fn = numpy.divide
        if method==AVG.Vectornorm:
            # for vector norm we divide by the largest (complex) amplitude
            fn = lambda x, _: x/numpy.max(numpy.abs(x))
        elif method in [AVG.NoAveraging, AVG.Sum, AVG.Vectorsum]:
            fn = lambda x, _: x
        # from counts form mask [note: do not clobber obj.m just yet, we need the counts!]
        m          = ARRAY(obj.m==0, dtype=numpy.bool)
        # set counts == 1 where counts were 0 to prevent dividing by 0
        obj.m[ m ] = 1
        # our various add_y() functions have made sure that no NaN/Inf exist in the data
        # so we don't have to blank anything; 's already done
        # compute average y
        obj.y      = fn(obj.y, obj.m)
        # replace the counts by the mask
        obj.m      = m
        # and set masked values to NaN because averaging no values has no answer
        obj.y[m]   = numpy.nan
        # and indicate we did do the averaging
        obj.af     = None

    def __init__(self):
        self.x  = self.y = self.m = None
        self.sf = dataset_chan.init_sumy
        self.af = dataset_chan.average_empty

    def append(self, xv, yv, m):
        raise NotImplemented("dataset_chan not meant for appending")

    def add_y(self, xv, yv, m):
        if not IsArray(xv):
            raise RuntimeError("dataset_chan:add_y() adding xv of non-array type! xv = {0}".format(xv))
        self.sf(self, xv, yv, m)

    def average(self, method):
        if self.af is not None:
            self.af(self, method)
        else:
            raise RuntimeError("Calling .average() > once on dataset_chan object?!")

    def is_numarray(self):
        return type(self.x) is _ArrayT and type(self.y) is _MArrayT

    def as_numarray(self):
        if self.is_numarray():
            return self
        if self.af is not None:
            raise RuntimeError("Request to convert unaveraged dataset_chan_solint to nd-array?!!")
        # note to self: float32 has insufficient precision for e.g.
        # <quantity> versus time
        self.x  = ARRAY(self.x, dtype=numpy.float64)
        self.y  = MARRAY(self.y, mask=self.m)
        return self

    def __str__(self):
        return "DATASET<chan>: len(x)={0}, len(y)={1} len(m)={2}".format(len(self.x), len(self.y), len(self.m))

    def __repr__(self):
        return str(self)


#################################################################################
#
# specialization of dataset for grouping multiple channels w/ mask by x value
# (think solint/group by time interval)
#
# the .append() accumulates the channels grouped by the x value
# .average() computes the channel averages over all data collected for each x value
#
#################################################################################
class dataset_solint_array:
    __slots__ = ['x', 'y', 'a', 'd', 'm']

    def __init__(self):
        self.x = self.y = None
        self.a = None
        self.d = collections.defaultdict(int)
        self.m = collections.defaultdict(int)

    # this specialization assumes yv, m are instances of numpy.ndarray or numpy.ma.core.MaskedArray
    def append(self, xv, yv, m):
        # masked data shall not count towards the total for computing the average
        #yv[m]      = 0
        # Also: any floating point aggregation function (sum, mean, etc.) will barf on
        # any of the aggregated values being [+-]Inf or NaN - i.e. the net result
        # will be NaN/Inf whatever. Therefore we must replace these with 0.
        # We'll put back NaN if it turns out that no unmasked values were averaged because
        # in such a situation there IS no average/sum/etc. ("what is the sum of no values?"):
        #   >>> import numpy
        #   >>> numpy.sum([1,2,numpy.nan])
        #   nan
        yv[ LOGICAL_OR(~ISFINITE(yv), m) ] = 0
        # accumulate in the bin for this specific x value
        self.d[xv] = numpy.add(yv, self.d[xv])
        self.m[xv] = numpy.add(~m, self.m[xv])

    def average(self, method):
        if self.a is not None:
            return
        # normal average = arithmetic mean i.e. summed value / count of valid values by default
        fn = numpy.divide
        if method==AVG.Vectornorm:
            # for vector norm we divide by the largest complex amplitude
            fn = lambda x, _: x/numpy.max(numpy.abs(x))
        elif method in [AVG.NoAveraging, AVG.Sum, AVG.Vectorsum]:
            # because we already integrate (==sum) then no averaging equals summing and v.v. :-)
            fn = lambda x, _: x
        # construct a new dict with the averaged data values and set mask wether
        # any unmasked values were collected for that x, channel
        self.a = dict()
        while self.d:
            (x, ys)    = self.d.popitem()
            counts     = self.m.pop(x)
            # ---- latest ---------------
            counts     = ARRAY(counts)
            mask       = ARRAY(counts==0, dtype=numpy.bool)
            counts[mask] = 1
            data       = fn(ARRAY(ys), counts)
            # after averaging, points with zero counts should be set to NaN
            # to effectively remove them.
            data[mask] = numpy.nan
            self.a[x]  = MARRAY(data, mask=mask)
            # ---------------------------

    def is_numarray(self):
        return type(self.x) is _ArrayT and type(self.y) is _MArrayT

    def as_numarray(self):
        if self.is_numarray():
            return self
        # note to self: float32 has insufficient precision for e.g.
        # the <time> axis in <quantity> versus time datasets
        if self.a is None:
            raise RuntimeError("solint dataset has not been averaged yet")
        self.x  = numpy.fromiter(self.a.iterkeys(), dtype=numpy.float64, count=len(self.a))
        self.y  = MARRAY(self.a.values())
        return self

    def __str__(self):
        return "DATASET<solint-array>: len(d)={0} len(m)={1}".format(len(self.d), len(self.m))

    def __repr__(self):
        return str(self)


##################################################################################
##
## specialization of dataset for grouping a single channels w/ mask by x value
## (think solint/group by time interval)
##
## the .append() accumulates the values  grouped by the x value
## .average() computes the value averages over all data collected for each x value
##
##################################################################################
class dataset_solint_scalar:
    __slots__ = ['x', 'y', 'a', 'd', 'm']

    def __init__(self):
        self.x = self.y = None
        self.a = None
        self.d = collections.defaultdict(list)
        self.m = collections.defaultdict(int)

    # this specialization assumes yv, m are scalar value + boolean (or anything
    # indicating the truth of yv)
    def append(self, xv, yv, m):
        # don't let masked or Inf/NaN values count towards the total before averaging
        self.d[xv].append( 0 if m or not ISFINITE(yv) else yv )
        # do count truth values
        self.m[xv] += 0 if m else 1

    def average(self, method):
        if self.a is not None:
            return
        # normal average = arithmetic mean i.e. summed value / count of valid values
        fn = operator.truediv
        if method==AVG.Vectornorm:
            # for vector norm we divide by the largest complex amplitude
            fn = lambda x, _: x/max(map(abs,x))
        elif method in [AVG.NoAveraging, AVG.Sum, AVG.Vectorsum]:
            # because our data is already summed then no averaging == summing
            fn = lambda x, _: x
        # construct a new dict with the averaged data values and set mask based on
        # number of unmasked
        self.a = dict()
        while self.d:
            (x, ys) = self.d.popitem()
            counts  = self.m.pop(x)
            # ---- latest ---------------
            # if no valid data at all substitute a value of nan
            self.a[x] = fn(sum(ys), counts) if counts else numpy.nan
            # ---------------------------

    def is_numarray(self):
        return type(self.x) is _ArrayT and type(self.y) is _MArrayT

    def as_numarray(self):
        if self.is_numarray():
            return self
        # note to self: float32 has insufficient precision for e.g.
        # the <time> axis in <quantity> versus time datasets
        if self.a is None:
            raise RuntimeError("solint dataset has not been averaged yet")
        self.x  = numpy.fromiter(self.a.iterkeys(), dtype=numpy.float64, count=len(self.a))
        self.y  = MARRAY(self.a.values())
        return self

    def __str__(self):
        return "DATASET<solint-scalar>: len(d)={0} len(m)={1}".format(len(self.d), len(self.m))

    def __repr__(self):
        return str(self)

## Partition a data set into two separate data sets,
## one with those elements satisfying the predicate,
## the other those who dont.
## Returns (ds_true, ds_false)
##
## Implementation note:
##  Yes, there is hvutil.partition() which does much the same but
##  using a reduce(). The problem is that it expects a single list of values
##  to which to apply the predicate.
##  In order to turn a dataset() into a single list, we'd have to
##   zip() the ".x" and ".y" lists. After having partition'ed the list,
##  we'd have to unzip them again into separate ".x" and ".y" arrays,
##  for the benefit of PGPLOT.
##  Summarizing: in order to use hvutil.partition() we'd have to do two (2)
##  cornerturning operations, which seems to be wasteful.
class partitioner:
    def __init__(self, expr):
        # solution found in:
        # http://stackoverflow.com/questions/10303248/true-dynamic-and-anonymous-functions-possible-in-python
        self.code = compile(
                "from numpy import *\n"+
                "from math  import *\n"+
                "avg  = None\n"+
                "sd   = None\n"+
                "xmin = None\n"+
                "xmax = None\n"+
                "ymin = None\n"+
                "ymax = None\n"+
                "f   = lambda x, y: "+expr,
                'dyn-mark-string', 'exec')
        self.mod  = new_module("dyn_marker_mod")
        exec(self.code, self.mod.__dict__)

    def __call__(self, x, y):
        ds_true       = []
        self.mod.avg  = numpy.mean(y)
        self.mod.sd   = numpy.std(y)
        self.mod.xmin = numpy.min(x)
        self.mod.xmax = numpy.max(x)
        self.mod.ymin = numpy.min(y)
        self.mod.ymax = numpy.max(y)
        for i in functional.range_(len(x)):
            if self.mod.f(x[i], y[i]):
                ds_true.append(i)
        return ds_true


### Turn an array of channel indices (the channels that we're interested in)
### into a 3D mask function
### Assumes that the indices have been shifted to 0 by slicing the column
### This implies that IF chanidx is a list of length 1, it must automatically
### be channel 0
#def mk3dmask_fn_idx(nrow, chanidx, npol):
#    return lambda x: x[:,chanidx,:]
#
#def mk3dmask_fn_mask(nrow, chanidx, npol):
#    if len(chanidx)>1 and (len(chanidx)!=(chanidx[-1]+1)):
#        # Start off with all channels masked, up to the last index
#        m              = numpy.ones( (nrow, chanidx[-1]+1, npol), dtype=numpy.int8 )
#        # all indexed channels have no mask
#        m[:,chanidx,:] = 0
#        return lambda x: numpy.ma.MaskedArray(x, mask=m)
#    else:
#        # single channel - or all channels
#        if len(chanidx)==1 and chanidx[0]!=0:
#            raise RuntimeError("consistency problem, chanidx[0] isn't 0 for single channel selection")
#        return lambda x: numpy.ma.MaskedArray(x, mask=numpy.ma.nomask)
#

def genrows(bls, ddids, fldids):
    tm = 0
    while True:
        for (bl, dd, fld) in itertools.product(bls, ddids, fldids):
            yield (tm, bl, dd, fld)
        tm = tm + 1

class fakems:
    def __init__(self, ms, mapping):
        #self.ms      = ms
        self.length  = len(ms)

        #(self.a1, self.a2) = zip( *mapping.baselineMap.baselineIndices() )
        self.bls     = mapping.baselineMap.baselineIndices()
        self.ddids   = mapping.spectralMap.datadescriptionIDs()
        self.flds    = mapping.fieldMap.getFieldIDs()
        shp = ms[0]["LAG_DATA" if "LAG_DATA" in ms.colnames() else "DATA"].shape
        while len(shp)<2:
            shp.append(1)
        self.shp   = shp
        self.rowgen = genrows(self.bls, self.ddids, self.flds)
        self.chunk = {}
        print("fakems/",len(self.bls)," baselines, ",len(self.ddids)," SB, ",len(self.flds)," SRC, shape:",self.shp)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        theshape = self.shp
        class column:
            def __init__(self, shp):
                self.shape = shp

        class row:
            def __init__(self):
                self.rdict = { 'DATA': column(theshape), 'LAG_DATA': column(theshape) }

            def __getitem__(self, colnm):
                return self.rdict[colnm]

        return row()

    def getcol(self, col, **kwargs):
        nrow     = kwargs['nrow']
        startrow = kwargs['startrow']
        if not startrow in self.chunk:
            # new block of rows. delete existing
            del self.chunk
            i = [0]
            def predicate(x):
                i[0] = i[0] + 1
                return i[0]<=nrow
            self.chunk = {startrow: list(itertools.takewhile(predicate, self.rowgen))}
        # rows = [ (tm, (a1, a2), dd, fld), .... ]
        rows = self.chunk[startrow]

        coldict = {
                "ANTENNA1"    : (lambda x: functional.map_(lambda tm, a1_a2, dd, fl: a1_a2[0], x), numpy.int32),
                "ANTENNA2"    : (lambda x: functional.map_(lambda tm, a1_a2, dd, fl: a1_a2[1], x), numpy.int32),
                "TIME"        : (lambda x: functional.map_(lambda tm, a1_a2, dd, fl: tm, x), numpy.float64),
                "DATA_DESC_ID": (lambda x: functional.map_(lambda tm, a1_a2, dd, fl: dd, x), numpy.int32),
                "FIELD_ID"    : (lambda x: functional.map_(lambda tm, a1_a2, dd, fl: fl, x), numpy.int32)
                }
        (valfn, tp) = coldict.get(col, (None, None))
        #print("getcol[{0}]/var={1}".format(col, var))
        if valfn:
            return numpy.array(valfn(rows), dtype=tp)
        if col=="WEIGHT":
            # nrow x npol
            shp = (nrow, self.shp[1])
            rv = numpy.ones( functional.reduce(operator.mul, shp), dtype=numpy.float32 )
            rv.shape = shp
            return rv
        if col=="DATA" or col=="LAG_DATA":
            shp = (nrow, self.shp[0], self.shp[1])
            rv = numpy.zeros( functional.reduce(operator.mul, shp), dtype=numpy.complex64 )
            rv.shape = shp
            return rv
        raise RuntimeError("Unhandled column {0}".format(col))


##### Different solint functions
#
#def solint_none(dsref):
#    return 0.0
#
## Tried a few different approaches for solint processing.
## The functions below are kept as illustrative references.
##
## They are ordered from slowest to fastest operation, as benchmarked on running
## on the same data set with the same settings.
##
##  solint_numpy_indexing:       7.2s runtime
##  solint_numpy_countbin:       5.9s
##  solint_pure_python:          3.8s
##  solint_pure_python3:         3.2s
##  solint_pure_python2:         2.8s
#
#
#def solint_numpy_indexing(dsref):
#    start = time.time()
#
#    dsref.as_numarray()
#    tms = numpy.unique(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    newds = dataset()
#    for tm in tms:
#        idxs = numpy.where(dsref.x==tm)
#        newds.append(tm, numpy.average(dsref.y[idxs]), numpy.any(dsref.m[idxs]) )
#    dsref.x = newds.x
#    dsref.y = newds.y
#    return time.time() - start
#
#def solint_numpy_countbin(dsref):
#    start = time.time()
#    dsref.as_numarray()
#
#    # get the unique time stamps
#    tms   = numpy.unique(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    # "bins" will be the destination bin where the quantity
#    # will be summed into for each unique time stamp
#    # i.e. all data having time stamp tms[0] will be summed into
#    #      bin 0, all data having time stamp tms[x] will be summed
#    #      into bin x
#    #bins  = range( len(tms) )
#    # Now we must transform the array of times (dsref.x) into an
#    # array with bin indices
#    dests = reduce(lambda acc, (ix, tm): \
#                      numpy.put(acc, numpy.where(dsref.x==tm), ix) or acc, \
#                   enumerate(tms), \
#                   numpy.empty(dsref.x.shape, dtype=numpy.int32))
#    # Good, now that we have that ...
#    sums  = numpy.bincount(dests, weights=dsref.y)
#    count = numpy.bincount(dests)
#    dsref.y = sums/count
#    dsref.x = tms
#    return time.time() - start
#
#
#def solint_pure_python(dsref):
#    start = time.time()
#    tms   = set(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    # accumulate data into bins of the same time
#    r = reduce(lambda acc, (tm, y): acc[tm].append(y) or acc, \
#               itertools.izip(dsref.x, dsref.y), \
#               collections.defaultdict(list))
#    # do the averaging
#    (x, y) = reduce(lambda (xl, yl), (tm, ys): (xl+[tm], yl+[sum(ys)/len(ys)]), \
#                    iteritems(r), (list(), list()))
#    dsref.x = x
#    dsref.y = y
#    return time.time() - start
#
#class average(object):
#    __slots__ = ['total', 'n']
#
#    def __init__(self):
#        self.total = 0.0
#        self.n     = 0
#
#    def add(self, other):
#        self.total += other
#        self.n     += 1
#        return None
#
#    def avg(self):
#        return self.total/self.n
#
#def solint_pure_python3(dsref):
#    start = time.time()
#    tms   = set(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    # accumulate data into bins of the same time
#    r = reduce(lambda acc, (tm, y): acc[tm].add(y) or acc, \
#               itertools.izip(dsref.x, dsref.y), \
#               collections.defaultdict(average))
#    # do the averaging
#    (x, y) = reduce(lambda (xl, yl), (tm, ys): (xl.append(tm) or xl, yl.append(ys.avg()) or yl), \
#                    iteritems(r), (list(), list()))
#    dsref.x = x
#    dsref.y = y
#    return time.time() - start
#
#def solint_pure_python2(dsref):
#    start = time.time()
#    tms   = set(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    # accumulate data into bins of the same time
#    r = reduce(lambda acc, (tm, y): acc[tm].append(y) or acc, \
#               itertools.izip(dsref.x, dsref.y), \
#               collections.defaultdict(list))
#    # do the averaging
#    (x, y) = reduce(lambda (xl, yl), (tm, ys): (xl.append(tm) or xl, yl.append(sum(ys)/len(ys)) or yl), \
#                    iteritems(r), (list(), list()))
#    dsref.x = x
#    dsref.y = y
#    return time.time() - start
#
#def solint_pure_python2a(dsref):
#    start = time.time()
#    tms   = set(dsref.x)
#
#    # check if there is something to be averaged at all
#    if len(tms)==len(dsref.x):
#        return time.time() - start
#
#    # accumulate data into bins of the same time
#    acc = collections.defaultdict(list)
#    y   = dsref.y
#    m   = dsref.m
#    for (i, tm) in enumerate(dsref.x):
#        if m[i] == False:
#            acc[ tm ].append( y[i] )
#    # do the averaging
#    (xl, yl) = (list(), list())
#    for (tm, ys) in iteritems(acc):
#        xl.append(tm)
#        yl.append( sum(ys)/len(ys) )
#    dsref.x = xl
#    dsref.y = yl
#    dsref.m = numpy.zeros(len(xl), dtype=numpy.bool)
#    return time.time() - start
#
## In solint_pure_python4 we do not check IF we need to do something, just DO it
#def solint_pure_python4(dsref):
#    start = time.time()
#
#    # accumulate data into bins of the same time
#    r = reduce(lambda acc, (tm, y): acc[tm].append(y) or acc, \
#               itertools.izip(dsref.x, dsref.y), \
#               collections.defaultdict(list))
#    # do the averaging
#    (dsref.x, dsref.y) = reduce(lambda (xl, yl), (tm, ys): (xl.append(tm) or xl, yl.append(sum(ys)/len(ys)) or yl), \
#                                iteritems(r), (list(), list()))
#    return time.time() - start
#
## solint_pure_python5 is solint_pure_python4 with the lambda's removed; replaced by
## calls to external functions. This shaves off another 2 to 3 milliseconds (on large data sets)
#def grouper(acc, (tm, y)):
#    acc[tm].append(y)
#    return acc
#
#def averager((xl, yl), (tm, ys)):
#    xl.append(tm)
#    yl.append(sum(ys)/len(ys))
#    return (xl, yl)
#
#def solint_pure_python5(dsref):
#    start = time.time()
#
#    # accumulate data into bins of the same time
#    r = reduce(grouper, itertools.izip(dsref.x, dsref.y), collections.defaultdict(list))
#    # do the averaging
#    (dsref.x, dsref.y) = reduce(averager, iteritems(r), (list(), list()))
#    return time.time() - start


## This plotter will iterate over "DATA" or "LAG_DATA"
## and produce a number of quantities per data point, possibly averaging over time and/or channels
class data_quantity_time(plotbase):

    # should set up a choice table based on the combination of averaging methods
    # key into the lookup is '(avgChannelMethod, avgTimeMethod)'
    # Also return wether the quantities must be postponed
    _averaging = {
        # no averaging at all, no need to postpone computing the quantity(ies)
        (AVG.NoAveraging, AVG.NoAveraging):             (avg_none, avg_none, False),
        # only time averaging requested
        # scalar in time means we can collect the quantities themselves
        (AVG.NoAveraging, AVG.Scalar):           (avg_none, avg_arithmetic, False),
        (AVG.NoAveraging, AVG.Sum):              (avg_none, avg_sum,        False),
        (AVG.NoAveraging, AVG.Vectorsum):        (avg_none, avg_sum,        True),
        # when vector(norm) averaging we must first collect all time data
        # before we can compute the quantities, i.e. their computation  must be postponed
        (AVG.NoAveraging, AVG.Vector):           (avg_none, avg_arithmetic, True),
        (AVG.NoAveraging, AVG.Vectornorm):       (avg_none, avg_vectornorm, True),
        # When scalar averaging the channels no vector averaging in time possible
        # Also no need to postpone computing the quantities
        (AVG.Scalar, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Scalar, AVG.Sum):            (avg_arithmetic, avg_sum,  False),
        (AVG.Scalar, AVG.Scalar):         (avg_arithmetic, avg_arithmetic, False),
        # When vector averaging the channels, the time averaging governs
        # the choice of when to compute the quantity(ies)
        (AVG.Vector, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Vector, AVG.Sum):            (avg_arithmetic, avg_sum,  False),
        (AVG.Vector, AVG.Vectorsum):      (avg_arithmetic, avg_sum,  True),
        (AVG.Vector, AVG.Scalar):         (avg_arithmetic, avg_arithmetic, False),
        # when doing vector in both dims we must first add up all the complex numbers
        # for each channel(selection) and then in time and THEN compute the quantity(ies)
        (AVG.Vector, AVG.Vector):         (avg_arithmetic, avg_arithmetic, True),
        (AVG.Vector, AVG.Vectornorm):     (avg_arithmetic, avg_vectornorm, True),
        # vectornorm averaging over the channels, see what's requested in time
        (AVG.Vectornorm, AVG.NoAveraging):       (avg_vectornorm, avg_none, False),
        (AVG.Vectornorm, AVG.Scalar):     (avg_vectornorm, avg_arithmetic, False),
        (AVG.Vectornorm, AVG.Vector):     (avg_vectornorm, avg_arithmetic, True),
        (AVG.Vectornorm, AVG.Vectornorm): (avg_vectornorm, avg_vectornorm, True),
        (AVG.Vectornorm, AVG.Sum):        (avg_vectornorm, avg_sum,        False),
        (AVG.Vectornorm, AVG.Vectorsum):  (avg_vectornorm, avg_sum,        True),

        (AVG.Sum, AVG.NoAveraging):              (avg_sum, avg_none,        False),
        (AVG.Sum, AVG.Sum):               (avg_sum, avg_sum,         False),
        (AVG.Sum, AVG.Scalar):            (avg_sum, avg_arithmetic,  False),

        (AVG.Vectorsum, AVG.NoAveraging):        (avg_sum, avg_none,        False),
        (AVG.Vectorsum, AVG.Scalar):      (avg_sum, avg_arithmetic,  False),
        (AVG.Vectorsum, AVG.Sum):         (avg_sum, avg_sum,         False),
        (AVG.Vectorsum, AVG.Vectorsum):   (avg_sum, avg_sum,         True),
        (AVG.Vectorsum, AVG.Vectornorm):  (avg_sum, avg_vectornorm,  True),
    }


    ## our construct0r
    ##   qlist = [ (quantity_name, quantity_fn), ... ]
    ##
    def __init__(self, qlist):
        self.quantities = list(itertools.starmap(Quantity, qlist))

    def makePlots(self, msname, selection, mapping, **kwargs):
        # Deal with channel averaging
        #   Scalar => average the derived quantity
        #   Vector => compute average cplx number, then the quantity
        avgChannel = CP(selection.averageChannel)
        avgTime    = CP(selection.averageTime)
        solchan    = CP(selection.solchan)
        solint     = CP(selection.solint)
        timerng    = CP(selection.timeRange)

        # some sanity checks
        if solchan is not None and avgChannel==AVG.NoAveraging:
            raise RuntimeError("nchav value was set without specifiying a channel averaging method; please tell me how you want them averaged")
        if solint is not None and avgTime==AVG.NoAveraging:
            raise RuntimeError("solint value was set without specifiying a time averaging method; please tell me how you want your time range(s) averaged")

        ## initialize the base class
        super(data_quantity_time, self).__init__(msname, selection, mapping, **kwargs)

        # channel selection+averaging schemes; support averaging over channels (or chunks of channels)
        chansel  = Ellipsis
        n_chan   = self.table[0][self.datacol].shape[0]
        if selection.chanSel:
            channels = mk_chansel(selection.chanSel)
            max_chan = max(channels)
            # if any of the indexed channels > n_chan that's an error
            if max_chan>=n_chan:
                raise RuntimeError("At least one selected channel ({0}) > largest channel index ({1})".format(max_chan, n_chan-1))
            # also <0 is not quite acceptable
            if min(channels)<0:
                raise RuntimeError("Negative channel number {0} is not acceptable".format(min(channels)))
            # if the user selected all channels (by selection
            # 'ch 0:<last>' in stead of 'ch none' we don't
            # override the default channel selection (which is more efficient)
            if channels!=functional.range_(n_chan):
                chansel = channels
            # ignore channel averaging if only one channel specified
            if (n_chan if chansel is Ellipsis else len(chansel))==1 and avgChannel != AVG.NoAveraging:
                print("WARNING: channel averaging method {0} ignored because only one channel selected or available".format( avgChannel ))
                avgChannel = AVG.NoAveraging

        # Test if the selected combination of averaging settings makes sense
        setup = data_quantity_time._averaging.get((avgChannel, avgTime), None)
        if setup is None:
            raise RuntimeError("the combination of {0} channel + {1} time averaging is not supported".format(avgChannel, avgTime))
        (avgchan_fn, avgtime_fn, postpone) = setup

        # How integration/averaging actually is implemented is by modifying the
        # time stamp.  By massaging the time stamp into buckets of size
        # 'solint', we influence the label of the TIME field, which will make
        # all data points with the same TIME stamp be integrated into the same
        # data set
        self.timebin_fn = functional.identity
        if avgTime!=AVG.NoAveraging:
            if solint is None:
                # Ah. Hmm. Have to integrate different time ranges
                # Let's transform our timerng list of (start, end) intervals into
                # a list of (start, end, mid) such that we can easily map
                # all time stamps [start, end] to mid
                # If no time ranges defined at all average everything down to middle of experiment?

                # It is important to KNOW that "selection.timeRange" (and thus our
                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
                timerng = functional.map_(lambda s_e: (s_e[0], s_e[1], sum(s_e)/2.0), timerng if timerng is not None else [(mapping.timeRange.start, mapping.timeRange.end)])
                if len(timerng)==1:
                    print("WARNING: averaging all data into one point in time!")
                    print("         This is because no solint was set or no time")
                    print("         ranges were selected to average. Your plot")
                    print("         may contain less useful info than expected")

                # try to be a bit optimized in time stamp replacement - filter the
                # list of time ranges to those applying to the time stamps we're
                # replacing
                def do_it(x):
                    mi,ma  = numpy.min(x), numpy.max(x)
                    ranges = functional.filter_(lambda tr: not (tr[0]>ma or tr[1]<mi), timerng)
                    return functional.reduce(lambda acc, s_e_m: numpy.put(acc, numpy.where((acc>=s_e_m[0]) & (acc<=s_e_m[1])), s_e_m[2]) or acc, ranges, x)
                self.timebin_fn = do_it
            else:
                # Check if solint isn't too small
                ti = mapping.timeRange.inttm[0]
                if solint<=ti:
                    raise RuntimeError("solint value {0:.3f} is less than integration time {1:.3f}".format(solint, ti))
                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0


        # chansel now is Ellipsis (all channels) or a list of some selected channels
        self.chanidx   = list()
        self.vectorAvg = functional.identity
        self.scalarAvg = functional.identity
        self.tmVectAvg = functional.identity
        self.tmScalAvg = functional.identity

        if avgChannel==AVG.NoAveraging:
            # No channel averaging - each selected channel goes into self.chanidx
            self.chanidx = list(enumerate(range(n_chan) if chansel is Ellipsis else chansel))
            # The vector average step will be misused to just apply the channel selection such that all selected channels
            # are mapped to 0..n-1. This is only necessary in case not all channels were selected
            if chansel is not Ellipsis:
                self.vectorAvg = lambda x: x[:,chansel,:]
        else:
            # ok channel averaging requested
            chbin_fn = None
            if solchan is None:
                # average all selected channels down to one
                # data array 'x' has shape (n_int, n_chan, n_pol)
                #self.chbin_fn = lambda x: normalize_ch(1)(numpy.ma.mean(x[:,avg_over,:], axis=1, keepdims=True))
                # average the selected channels according the requested averaging method
                chbin_fn      = lambda x: avgchan_fn(1)(x[:,chansel,:])
                self.chanidx  = [(0, '*')]
            else:
                # average bins of solchan channels down to one
                if solchan > n_chan:
                    raise RuntimeError("request to average channels in bins of {0} channels but only {1} are available".format(solchan, n_chan))
                # Create a mask which is the complement of the selected channels
                # (remember: chansel == Ellipsis => all channels
                ch_mask       = (numpy.zeros if chansel is Ellipsis else numpy.ones)(n_chan, dtype=numpy.bool)
                # only in case chansel != everything we must modify the mask
                if chansel is not Ellipsis:
                    ch_mask[chansel] = False

                # Since we're going to zap masked values (replace by 0) we can usefully use
                # reduceat! So all we then need is an index array, informing reduceat what the
                # reduction boundaries are!
                # First up: the actual bin numbers we're interested in, we compute the actual
                #           start + end indices from that
                bins    = numpy.unique((numpy.array(chansel) if chansel is not Ellipsis else numpy.arange(0, n_chan, solchan))//solchan)
                bins.sort()

                # we're going to apply channel binning so we must replace 'chansel'
                # by 'bins' in order for downstream accounting of how many "channels" there will
                # be in the data
                chansel = bins

                # Did timings on comparing simplistic 'loop over list of slices' and numpy.add.reduceat based approaches.
                # Results: grab bag - depending on problem set size:
                #      - simplistic approach between 2.5-6x faster (!) when averaging small number of
                #        channels in small number of bins (say <= 5 bins of ~5 channels)
                #      - reduceat approach slightly more than 2x faster when binning
                #        large number of channels in large-ish amount of bins (say 32 bins
                #        of 4 channels)

                # Update: the reduceat also only wins if all bins are adjacent.
                #         the way <operator>.reduceat works is, given a list of indices [i,j,k]
                #         and applied to an array A, it will produce the following outputs:
                #             [ <operator>(A[i:j]), <operator>(A[j:k]), <operator>(A[k:-1]) ]
                #         (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html)
                #
                #         Basically we can use this to efficiently bin i:j, j:k, ..., z:-1 ranges
                #         If our bins (or in the future, arbitrary channels ranges)  are NOT adjacent, then we must
                #         feed these operators to <operator>.reduceat:
                #             [ start0, end0, start1, end1, ..., startN, endN ]
                #         with the start, end indices of channel ranges 0..N
                #         will produce the following outputs:
                #             [ <operator>( A[start0:end0] ), <operator>( A[end0:start1] ), <operator>( A[start1:end1] ), ... ]
                #         so we'd have to throw out every second entry in the output.
                #         In numpy that's simple enough but it also means that <operator>.reduceat() does twice as must work
                #         for no apparent reason.

                # Detect if the bins are adjacent
                adjacent_bins = (len(set(bins[1:] - bins[:-1])) == 1) if len(bins)>1 else False

                chbin_fn      = None
                if adjacent_bins:
                    # we're going to use reduceat() which means it's good enough
                    # to generate [bin0*solchan, bin1*solchan, ..., bin<nbin-1>*solchan]
                    indices = CP(bins)
                    # generate the channel index labels for correct labelling
                    self.chanidx = list()
                    for (ch_idx, start) in enumerate(indices):
                        self.chanidx.append( (ch_idx, start) )
                        #self.chanidx.append( (ch_idx, "{0}*".format(start)) )
                    # need to carefully check last entry in there; if 'last bin' < 'n_chan//solchan'
                    # we must add an extra final boundary or else reduceat() will add up to the end
                    # of the number of channels in stead of until the end of the bin ...
                    if bins[-1]<((n_chan-1)//solchan):
                        #print("Must add one more bin limit; bins=",bins)
                        # add one more bin limit, set slice to keep only n-1 bins
                        keepbins = slice(0, len(indices))
                        indices  = numpy.r_[indices, [indices[-1]+1]]
                        #print("  indices now = ",indices)
                    else:
                        keepbins = Ellipsis
                    # indices are in units of solchan bins so for reduceat must
                    # scale them back to actual channels
                    indices  *= solchan
                    #print("  indices now = ",indices)
                    # This is where the magic happens
                    transpose_ch = operator.methodcaller('transpose', (0, 2, 1))
                    def use_reduceat(x):
                        # (n_int, n_ch, n_pol) => (n_int, n_pol, n_ch)
                        tmpx = transpose_ch(x)
                        # also we must reshape it to a 2-D array of ((n_int * n_pol), n_chan) shape orelse
                        # the reduceat() don't work [https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html]
                        # remember the dimensions for later
                        n_int,n_pol = tmpx.shape[:-1]
                        tmpx        = tmpx.reshape( (n_int*n_pol, -1) )
                        # mask out channels that we don't want averaged, joining it
                        # with the mask that excludes flagged data (by the user) and/or
                        # whatever was weight-thresholded ...
                        #print("====> previous mask shape=", tmpx.mask.shape,"/",tmpx.mask[-1])
                        #print("      channel mask shape", ch_mask.shape,"/",ch_mask)
                        #print("      indices=",indices)
                        #print("      data=",tmpx.data[-1](
                        tmpx.mask = LOGICAL_OR(tmpx.mask, ch_mask)
                        #print("      final mask shape", tmpx.mask.shape,"/",tmpx.mask[-1])
                        #if tmpx.mask.all():
                        #    print("   NO UNMASKED DATA")
                        # set all masked values to 0 such that they don't ever count towards *anything*
                        # e.g. suppose all channels in a bin are masked then the average should be NaN or something
                        #      unrepresentable because there was no valid data at all
                        tmpx.data[tmpx.mask] = 0
                        #print("      data after masking=",tmpx.data[-1])
                        # do the summation.
                        result = numpy.add.reduceat(tmpx.data, indices, axis=1)[:,keepbins]
                        #print(" RESULT=",result)
                        #tmp = numpy.add.reduceat(tmpx.data, indices, axis=1)
                        #result = tmp[:,keepbins]
                        # also count the number of unmasked values that went into each point
                        # we may use it for averaging, definitely be using it to create the mask
                        counts = numpy.add.reduceat(~tmpx.mask, indices, axis=1)[:,keepbins]
                        # pre-create the masked based on places where the count of unflagged points == 0
                        # these values have to be removed in the output (and also we can prevent
                        # divide-by-zero errors)
                        mask   = ARRAY(counts == 0, dtype=numpy.bool)
                        #print(" COUNTS=",counts)
                        # Because we do things different here than in the ordinary averaging,
                        # we must look at what was requested in order to mimic that behaviour
                        if avgchan_fn is avg_vectornorm:
                            # ok need to find the maximum complex number in each bin to scale it by
                            # take proper care of flagged/inf data
                            tmpx.data[tmpx.mask] = -numpy.inf
                            result /= (numpy.maximum.reduceat(numpy.abs(tmpx.data), indices, axis=1)[:,keepbins])
                        elif avgchan_fn in [avg_sum, avg_none]:
                            # either vector or scalar sum or no averaging, don't do anything
                            pass
                        else:
                            # ordinary arithmetic mean
                            # sets counts = 1 where counts == 0 so we don't divided by 0
                            counts[ mask ] = 1
                            result /= counts
                        # set entries where counts==0 to NaN to make it explicit
                        # that, mathematically speaking, there is nothing there
                        result[mask] = numpy.nan
                        # unshape + untranspose from 2-d ((n_int * n_pol), n_output_channels)
                        #                       into 3-d (n_int, n_pol, n_ouput_channels)
                        return transpose_ch(numpy.ma.array(result.reshape((n_int, n_pol, -1)), mask=mask.reshape((n_int, n_pol, -1))))
                    # set chbin_fn to use reduceat()
                    chbin_fn = use_reduceat
                else:
                    # not going to use reduceat() just bruteforcing over a list of slices()
                    # do some extra pre-processing for the simplistic approach
                    # it uses slice() indexing so we pre-create the slice objects for it
                    # for each range of channels to average we compute src and dst slice
                    indices = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                    slices  = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices)]
                    # for display + loopindexing create list of (array_index, "CH label") tuples
                    self.chanidx = list()
                    for (ch_idx, start_end) in enumerate(indices):
                        self.chanidx.append( (ch_idx, start_end[0]) )
                        #self.chanidx.append( (ch_idx, "{0}*".format(start_end[0])) )
                    n_slices = len(slices)
                    #  this is the simplistic approach
                    def use_dumbass_method(x):
                        # get an output array
                        n_int,_,n_pol = x.shape
                        result        = numpy.ma.empty((n_int, n_slices, n_pol), dtype=x.dtype)
                        for (dst_idx, src_idx) in slices:
                            result[:,dst_idx,:]      = numpy.ma.mean(x[:,src_idx,:], axis=1, keepdims=True)
                            result.mask[:,dst_idx,:] = (numpy.sum(x.mask[:,src_idx,:], axis=1, keepdims=True) == 0)
                        return result
                    # and set the channel bin function to use to this'un
                    chbin_fn = use_dumbass_method

            # Some channel averaging is to be applied so chbin_fn must not be None
            if chbin_fn is None:
                raise RuntimeError("chbin_fn is None whilst some channel averaging requested. Please yell at H. Verkouter (verkouter@jive.eu)")

            # depending on which kind of channel averaging, we apply it to the complex data before
            # producing quantities or on the scalars after computing the quantities
            if avgChannel == AVG.Scalar:
                self.scalarAvg = chbin_fn
            else:
                self.vectorAvg = chbin_fn

        # Now we must come up with a strategy for organizing the data processing chain
        #
        # If avgTime is Vector-like then we can only compute the quantities after all data's been
        # read & averaged. We've already ruled out that avgChannel == Scalar (for that makes no sense)
        #
        # So we may have to postpone computing of the quantities until after having collected + integrated all data
        post_quantities = lambda tp, x: [(tp, x)]
        org_quantities  = None
        if postpone:
            # create data sets based on the averaged data in a dataset
            org_quantities  = CP(self.quantities)
            post_quantities = lambda _, x: functional.map_(lambda q: (q.quantity_name, q.quantity_fn(x)), org_quantities)
            self.quantities = [Quantity('raw', functional.identity)]

        if len(self.chanidx)==1:
            # post_channel doesn't do nothing, self.chanidx remains a list of length 1
            post_channel  = lambda ch, x: [(ch, x)]
        else:
            # here the post_channel yields a list of extracted channels from the data 'x' coupled
            # with the assigned label from self.chanidx
            org_chanidx   = CP(self.chanidx)
            post_channel  = lambda _, x: functional.map_(lambda chi: (chi[1], x[:,chi[0]]), org_chanidx)
            # replace self.chanidx with a single entry which captures all channels and sets the
            # associated label to None - which we could use as a sentinel, if needed
            self.chanidx  = [(Ellipsis, None)]

        # Let's keep the channels together as long as possible. If only one channel remains then we can do it
        # in our inner loop
        #dataset_proto = dataset_list if avgTime == AVG.NoAveraging else dataset_solint_array
        if avgTime == AVG.NoAveraging:
            dataset_proto = dataset_list
        else:
            # depending on wether we need to solint one or more channels in one go
            # loop over the current self.chanidx and count nr of channels
            nChannel = functional.reduce(lambda acc, chi: acc + ((n_chan if chansel is Ellipsis else len(chansel)) if chi[0] is Ellipsis else 1), self.chanidx, 0)
            dataset_proto = dataset_solint_array if nChannel>1 else dataset_solint_scalar

        ## Now we can start the reduction of the table
        # Note: there will /always/ be WEIGHT+FLAGCOL - either read from the table or invented
        #          0        1      2      3      4       5     6
        fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.CH]
        columns = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHTCOL", "FLAG_ROW", "FLAG", self.datacol]
        pts     =  ms2util.reducems2(self, self.table, collections.defaultdict(dataset_proto), columns, verbose=True, slicers=self.slicers, chunksize=5000)

        # after the reduction's done we can put back our quantities if we did remove them before
        if org_quantities is not None:
            self.quantities = org_quantities

        rv  = {}
        for (label, dataset) in iteritems(pts):
            dl = list(label)
            dataset.average( avgTime )
            # convert x,y to numarrays
            dataset.as_numarray()
            for qn,qd in post_quantities(label[0], dataset.y):
                for chn,chd in post_channel(label[6], qd):
                    dl[0] = qn
                    dl[6] = chn
                    rv[ self.MKLAB(fields, dl) ] = dataset_fixed(dataset.x, chd)
        return rv

    ## Here we make the plots
    def __call__(self, acc, a1, a2, tm, dd, fld, weight, flag_row, flag, data):
        #print("************************************************")
        #print("*   __call__ data.shape=",data.shape)
        #print("************************************************")

        # Create masked array from the data with invalid data already masked off
        data = numpy.ma.masked_invalid(data)
        # now we can easily add in flag information;
        # flags either has shape of data or it's a single bool False
        data.mask = numpy.logical_or(data.mask, self.untranspose_flag(numpy.logical_or(self.transpose_flag(flag), flag_row)))
        # weight handling. It's been set up such that whatever the weight was
        # (WEIGHT, WEIGHT_SPECTRUM, no weight thresholding) the following sequence
        # always works
        data      = self.transpose(data)
        data.mask = numpy.logical_or(data.mask, weight<self.threshold)
        data      = self.transpose(data)

        # possibly vector-average the data
        data      = self.vectorAvg(data)
        # Now create the quantity data - map the quantity functions over the (vector averaged)
        # data and, if needed, scalar average them
        qd        = functional.map_(lambda q: (q.quantity_name, self.scalarAvg(q.quantity_fn(data))), self.quantities)

        # Transform the time stamps, if necessary
        tm        = self.timebin_fn(tm)

        # Now we can loop over all the rows in the data
        dds  = self.ddSelection
        ci   = self.chanidx
        # We don't have to test *IF* the current data description id is
        # selected; the fact that we see it here means that it WAS selected!
        # The only interesting bit is selecting the correct products
        for row in range(data.shape[0]):
            (fq, sb, plist) = dds[ dd[row] ]
            for (chi, chn) in ci:
                for (pidx, pname) in plist:
                    l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, chn]
                    for (qnm, qval) in qd:
                        l[0] = qnm
                        acc[tuple(l)].append(tm[row], qval.data[row, chi, pidx], qval.mask[row, chi, pidx])
        return acc


## This plotter will iterate over "DATA" or "LAG_DATA"
## and produce a number of quantities per frequency, possibly averaging over time and/or channels
class data_quantity_chan(plotbase):

    # should set up a choice table based on the combination of averaging methods
    # key into the lookup is '(avgChannelMethod, avgTimeMethod)'
    # Also return wether the quantities must be postponed
    _averaging = {
        # no averaging at all, no need to postpone computing the quantity(ies)
        (AVG.NoAveraging, AVG.NoAveraging):             (avg_none, avg_none, False),
        (AVG.NoAveraging, AVG.Sum):              (avg_none, avg_sum,  False),
        (AVG.NoAveraging, AVG.Vectorsum):        (avg_none, avg_sum,  True),
        # only time averaging requested
        # scalar in time means we can collect the quantities themselves
        (AVG.NoAveraging, AVG.Scalar):           (avg_none, avg_arithmetic, False),
        # when vector(norm) averaging we must first collect all time data
        # before we can compute the quantities, i.e. their computation  must be postponed
        (AVG.NoAveraging, AVG.Vector):           (avg_none, avg_arithmetic, True),
        (AVG.NoAveraging, AVG.Vectornorm):       (avg_none, avg_vectornorm, True),
        # When scalar averaging the channels no vector averaging in time possible
        # Also no need to postpone computing the quantities
        (AVG.Scalar, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Scalar, AVG.Sum):            (avg_arithmetic, avg_sum,  False),
        (AVG.Scalar, AVG.Scalar):         (avg_arithmetic, avg_arithmetic, False),
        # When vector averaging the channels, the time averaging governs
        # the choice of when to compute the quantity(ies)
        (AVG.Vector, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Vector, AVG.Sum):            (avg_arithmetic, avg_sum,  False),
        (AVG.Vector, AVG.Vectorsum):      (avg_arithmetic, avg_sum,  True),
        (AVG.Vector, AVG.Scalar):         (avg_arithmetic, avg_arithmetic, False),
        # when doing vector in both dims we must first add up all the complex numbers
        # for each channel(selection) and then in time and THEN compute the quantity(ies)
        (AVG.Vector, AVG.Vector):         (avg_arithmetic, avg_arithmetic, True),
        (AVG.Vector, AVG.Vectornorm):     (avg_arithmetic, avg_vectornorm, True),
        # vectornorm averaging over the channels, see what's requested in time
        (AVG.Vectornorm, AVG.NoAveraging):       (avg_vectornorm, avg_none, False),
        (AVG.Vectornorm, AVG.Scalar):     (avg_vectornorm, avg_arithmetic, False),
        (AVG.Vectornorm, AVG.Vector):     (avg_vectornorm, avg_arithmetic, True),
        (AVG.Vectornorm, AVG.Vectornorm): (avg_vectornorm, avg_vectornorm, True),
        (AVG.Vectornorm, AVG.Sum):        (avg_vectornorm, avg_sum,        False),
        (AVG.Vectornorm, AVG.Vectorsum):  (avg_vectornorm, avg_sum,        True),

        (AVG.Sum, AVG.NoAveraging):              (avg_sum, avg_none,        False),
        (AVG.Sum, AVG.Sum):               (avg_sum, avg_sum,         False),
        (AVG.Sum, AVG.Scalar):            (avg_sum, avg_arithmetic,  False),

        (AVG.Vectorsum, AVG.NoAveraging):        (avg_sum, avg_none,        False),
        (AVG.Vectorsum, AVG.Scalar):      (avg_sum, avg_arithmetic,  False),
        (AVG.Vectorsum, AVG.Sum):         (avg_sum, avg_sum,         False),
        (AVG.Vectorsum, AVG.Vectorsum):   (avg_sum, avg_sum,         True),
        (AVG.Vectorsum, AVG.Vectornorm):  (avg_sum, avg_vectornorm,  True),
    }


    ## our construct0r
    ##   qlist = [ (quantity_name, quantity_fn), ... ]
    ##
    def __init__(self, qlist, **kwargs):
        self.quantities  = list(itertools.starmap(Quantity, qlist))
        self.byFrequency = kwargs.get('byFrequency', False)

    def makePlots(self, msname, selection, mapping, **kwargs):
        # Deal with channel averaging
        #   Scalar => average the derived quantity
        #   Vector => compute average cplx number, then the quantity
        avgChannel = CP(selection.averageChannel)
        avgTime    = CP(selection.averageTime)
        solchan    = CP(selection.solchan)
        solint     = CP(selection.solint)
        timerng    = CP(selection.timeRange)

        # some sanity checks
        if solchan is not None and avgChannel==AVG.NoAveraging:
            raise RuntimeError("nchav value was set without specifiying a channel averaging method; please tell me how you want them averaged")
        if solint is not None and avgTime==AVG.NoAveraging:
            raise RuntimeError("solint value was set without specifiying a time averaging method; please tell me how you want your time range(s) averaged")

        ## initialize the base class
        super(data_quantity_chan, self).__init__(msname, selection, mapping, **kwargs)

        # channel selection+averaging schemes; support averaging over channels (or chunks of channels)
        chansel  = Ellipsis #None
        n_chan   = self.table[0][self.datacol].shape[0]
        if selection.chanSel:
            channels = list(sorted(set(CP(selection.chanSel))))
            max_chan = max(channels)
            # if any of the indexed channels > n_chan that's an error
            if max_chan>=n_chan:
                raise RuntimeError("At least one selected channel ({0}) > largest channel index ({1})".format(max_chan, n_chan-1))
            # also <0 is not quite acceptable
            if min(channels)<0:
                raise RuntimeError("Negative channel number {0} is not acceptable".format(min(channels)))
            # if the user selected all channels (by selection
            # 'ch 0:<last>' in stead of 'ch none' we don't
            # override the default channel selection (which is more efficient)
            if channels!=functional.range_(n_chan):
                chansel = channels
            # ignore channel averaging if only one channel specified
            if (n_chan if chansel is Ellipsis else len(chansel))==1 and avgChannel != AVG.NoAveraging:
                print("WARNING: channel averaging method {0} ignored because only one channel selected".format( avgChannel ))
                avgChannel = AVG.NoAveraging

        # Test if the selected combination of averaging settings makes sense
        setup = data_quantity_time._averaging.get((avgChannel, avgTime), None)
        if setup is None:
            raise RuntimeError("the combination of {0} channel + {1} time averaging is not supported".format(avgChannel, avgTime))
        (avgchan_fn, avgtime_fn, postpone) = setup

        # How integration/averaging actually is implemented is by modifying the
        # time stamp.  By massaging the time stamp into buckets of size
        # 'solint', we influence the label of the TIME field, which will make
        # all data points with the same TIME stamp be integrated into the same
        # data set
        self.timebin_fn = functional.identity
        if avgTime!=AVG.NoAveraging:
            if solint is None:
                # Ah. Hmm. Have to integrate different time ranges
                # Let's transform our timerng list of (start, end) intervals into
                # a list of (start, end, mid) such that we can easily map
                # all time stamps [start, end] to mid
                # If no time ranges defined at all average everything down to middle of experiment?

                # It is important to KNOW that "selection.timeRange" (and thus our
                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
                timerng = functional.map_(lambda s_e: (s_e[0], s_e[1], sum(s_e)/2.0), timerng if timerng is not None else [(mapping.timeRange.start, mapping.timeRange.end)])

                # try to be a bit optimized in time stamp replacement - filter the
                # list of time ranges to those applying to the time stamps we're
                # replacing
                def do_it(x):
                    mi,ma  = numpy.min(x), numpy.max(x)
                    ranges = functional.filter_(lambda tr: not (tr[0]>ma or tr[1]<mi), timerng)
                    return functional.reduce(lambda acc, s_e_m: numpy.put(acc, numpy.where((acc>=s_e_m[0]) & (acc<=s_e_m[1])), s_e_m[2]) or acc, ranges, x)
                self.timebin_fn = do_it
            else:
                # Check if solint isn't too small
                ti = mapping.timeRange.inttm[0]
                if solint<=ti:
                    raise RuntimeError("solint value {0:.3f} is less than integration time {1:.3f}".format(solint, ti))
                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0

        # chansel now is Ellipsis (all channels) or a list of some selected channels
        self.chanidx   = list()
        self.chanidx_fn= None
        self.vectorAvg = functional.identity
        self.scalarAvg = functional.identity
        self.tmVectAvg = functional.identity
        self.tmScalAvg = functional.identity

        # if the x-axis is frequency ... *gulp*
        self.freq_of_dd=CP(self.ddFreqs)

        if avgChannel==AVG.NoAveraging:
            # No channel averaging - the new x-axis will be the indices of the selected channels
            self.chanidx = list(functional.range_(n_chan) if chansel is Ellipsis else chansel)
            # The vector average step will be misused to just apply the channel selection such that all selected channels
            # are mapped to 0..n-1. This is only necessary in case not all channels were selected
            if chansel is not Ellipsis:
                self.vectorAvg = lambda x: x[:,chansel,:]
            if self.byFrequency:
                self.chanidx_fn = lambda dd: self.freq_of_dd[dd][self.chanidx]
            else:
                self.chanidx_fn = functional.const(self.chanidx)
        else:
            # ok channel averaging requested
            chbin_fn = None
            if solchan is None:
                # average all selected channels down to one
                # data array 'x' has shape (n_int, n_chan, n_pol)
                #self.chbin_fn = lambda x: normalize_ch(1)(numpy.ma.mean(x[:,avg_over,:], axis=1, keepdims=True))
                # average the selected channels according the requested averaging method
                chbin_fn      = lambda x: avgchan_fn(1)(x[:,chansel,:])
                # compute average channel number - honouring Ellipsis if necessary #[(0, '*')]
                self.chanidx  = [numpy.mean(list(functional.range_(n_chan) if chansel is Ellipsis else chansel))]
                # transform all frequencies to an average frequency
                for dd in self.freq_of_dd.keys():
                    self.freq_of_dd[dd] = [numpy.mean(self.freq_of_dd[dd][chansel])]
                if self.byFrequency:
                    self.chanidx_fn = lambda dd: self.freq_of_dd[dd]
                else:
                    self.chanidx_fn = functional.const(self.chanidx)
            else:
                # average bins of solchan channels down to one
                if solchan > n_chan:
                    raise RuntimeError("request to average channels in bins of {0} channels but only {1} are available".format(solchan, n_chan))
                # Create a mask which is the complement of the selected channels
                # (remember: chansel == Ellipsis => all channels
                ch_mask       = (numpy.zeros if chansel is Ellipsis else numpy.ones)(n_chan, dtype=numpy.bool)
                # only in case chansel != everything we must modify the mask
                if chansel is not Ellipsis:
                    ch_mask[chansel] = False

                # Since we're going to zap masked values (replace by 0) we can usefully use
                # reduceat! So all we then need is an index array, informing reduceat what the
                # reduction boundaries are!
                # First up: the actual bin numbers we're interested in, we compute the actual
                #           start + end indices from that
                bins    = numpy.unique((numpy.array(chansel) if chansel is not Ellipsis else numpy.arange(0, n_chan, solchan))//solchan)
                bins.sort()

                # we're going to apply channel binning so we must replace 'chansel'
                # by 'bins' in order for downstream accounting of how many "channels" there will
                # be in the data
                chansel = bins

                # Did timings on comparing simplistic 'loop over list of slices' and numpy.add.reduceat based approaches.
                # Results: grab bag - depending on problem set size:
                #      - simplistic approach between 2.5-6x faster (!) when averaging small number of
                #        channels in small number of bins (say <= 5 bins of ~5 channels)
                #      - reduceat approach slightly more than 2x faster when binning
                #        large number of channels in large-ish amount of bins (say 32 bins
                #        of 4 channels)

                # Update: the reduceat also only wins if all bins are adjacent.
                #         the way <operator>.reduceat works is, given a list of indices [i,j,k]
                #         and applied to an array A, it will produce the following outputs:
                #             [ <operator>(A[i:j]), <operator>(A[j:k]), <operator>(A[k:-1]) ]
                #         (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html)
                #
                #         Basically we can use this to efficiently bin i:j, j:k, ..., z:-1 ranges
                #         If our bins (or in the future, arbitrary channels ranges)  are NOT adjacent, then we must
                #         feed these operators to <operator>.reduceat:
                #             [ start0, end0, start1, end1, ..., startN, endN ]
                #         with the start, end indices of channel ranges 0..N
                #         will produce the following outputs:
                #             [ <operator>( A[start0:end0] ), <operator>( A[end0:start1] ), <operator>( A[start1:end1] ), ... ]
                #         so we'd have to throw out every second entry in the output.
                #         In numpy that's simple enough but it also means that <operator>.reduceat() does twice as must work
                #         for no apparent reason.

                # Detect if the bins are adjacent
                adjacent_bins = (len(set(bins[1:] - bins[:-1])) == 1) if len(bins)>1 else False

                chbin_fn      = None
                if adjacent_bins:
                    # we're going to use reduceat() which means it's good enough
                    # to generate [bin0*solchan, bin1*solchan, ..., bin<nbin-1>*solchan]
                    indices = CP(bins)
                    # generate the channel index labels for correct labelling
                    self.chanidx = CP(indices) #list()
                    # need to carefully check last entry in there; if 'last bin' < 'n_chan//solchan'
                    # we must add an extra final boundary or else reduceat() will add up to the end
                    # of the number of channels in stead of until the end of the bin ...
                    if bins[-1]<((n_chan-1)//solchan):
                        # add one more bin limit, set slice to keep only n-1 bins
                        keepbins = slice(0, len(indices))
                        indices  = numpy.r_[indices, [indices[-1]+1]]
                    else:
                        keepbins = Ellipsis
                    # indices are in units of solchan bins so for reduceat must
                    # scale them back to actual channels
                    indices  *= solchan
                    # This is where the magic happens
                    transpose_ch = operator.methodcaller('transpose', (0, 2, 1))
                    def use_reduceat(x):
                        # (n_int, n_ch, n_pol) => (n_int, n_pol, n_ch)
                        tmpx = transpose_ch(x)
                        # also we must reshape it to a 2-D array of ((n_int * n_pol), n_chan) shape orelse
                        # the reduceat() don't work [https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html]
                        # remember the dimensions for later
                        n_int,n_pol = tmpx.shape[:-1]
                        tmpx        = tmpx.reshape( (n_int*n_pol, -1) )
                        # mask out channels that we don't want averaged, joining it
                        # with the mask that excludes flagged data (by the user) and/or
                        # whatever was weight-thresholded ...
                        tmpx.mask = numpy.logical_or(tmpx.mask, ch_mask)
                        # set all masked values to 0 such that they don't ever count towards *anything*
                        # e.g. suppose all channels in a bin are masked then the average should be NaN or something
                        #      unrepresentable because there was no valid data at all
                        tmpx.data[tmpx.mask] = 0
                        # do the summation.
                        #print("###############################################################")
                        #print("    use_reduceat: x.shape=",x.shape)
                        #print("          tmpx.data.shape=",tmpx.data.shape)
                        #print("                  indices=",indices)
                        #print("                 keepbins=",keepbins)
                        #print("###############################################################")
                        result = numpy.add.reduceat(tmpx.data, indices, axis=1)[:,keepbins]
                        # also count the number of unmasked values that went into each point
                        # we may use it for averaging, definitely be using it to create the mask
                        counts = numpy.add.reduceat(~tmpx.mask, indices, axis=1)[:,keepbins]
                        # Because we do things different here than in the ordinary averaging,
                        # we must look at what was requested in order to mimic that behaviour
                        if avgchan_fn is avg_vectornorm:
                            # ok need to find the maximum complex number in each bin to scale it by
                            result /= (numpy.maximum.reduceat(numpy.abs(tmpx.data), indices, axis=1)[:,keepbins])
                        elif avgchan_fn in [avg_sum, avg_none]:
                            # no averaging/summing boils down to not doing anything
                            pass
                        else:
                            # ordinary arithmetic mean
                            result /= counts
                        # return masked array - can reuse the counts array by converting them
                        # to bool and inverting: no counts => False
                        mask = numpy.array(counts == 0, dtype=numpy.bool)
                        # set entries where counts==0 to NaN to make it explicit
                        # that, mathematically speaking, there is nothing there
                        result[mask] = numpy.nan
                        # unshape + untranspose from 2-d ((n_int * n_pol), n_output_channels)
                        #                       into 3-d (n_int, n_pol, n_ouput_channels)
                        return transpose_ch(numpy.ma.array(result.reshape((n_int, n_pol, -1)), mask=mask.reshape((n_int, n_pol, -1))))
                    # set chbin_fn to use reduceat()
                    chbin_fn = use_reduceat
                else:
                    # not going to use reduceat() just bruteforcing over a list of slices()
                    # do some extra pre-processing for the simplistic approach
                    # it uses slice() indexing so we pre-create the slice objects for it
                    # for each range of channels to average we compute src and dst slice
                    indices = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                    slices  = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices)]
                    # for display + loopindexing create list of (array_index, "CH label") tuples
                    self.chanidx = CP(bins)
                    n_slices = len(slices)
                    #  this is the simplistic approach
                    def use_dumbass_method(x):
                        # get an output array
                        n_int,_,n_pol = x.shape
                        result        = numpy.ma.empty((n_int, n_slices, n_pol), dtype=x.dtype)
                        for (dst_idx, src_idx) in slices:
                            result[:,dst_idx,:]      = numpy.ma.mean(x[:,src_idx,:], axis=1, keepdims=True)
                            result.mask[:,dst_idx,:] = (numpy.sum(x.mask[:,src_idx,:], axis=1, keepdims=True) == 0)
                        return result
                    # and set the channel bin function to use to this'un
                    chbin_fn = use_dumbass_method

                # transform all frequencies to an average frequency per bin
                indices_l  = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                slices_l   = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices_l)]
                def mk_fbins(x):
                    result = numpy.empty((len(slices_l)))
                    for (dst_idx, src_idx) in slices_l:
                        result[dst_idx]      = numpy.mean(x[src_idx])
                    return result
                for dd in self.freq_of_dd.keys():
                    freqs               = ARRAY(self.freq_of_dd[dd])
                    self.freq_of_dd[dd] = mk_fbins( ARRAY(self.freq_of_dd[dd]) )
                if self.byFrequency:
                    self.chanidx_fn = lambda dd: self.freq_of_dd[dd]
                else:
                    self.chanidx_fn = functional.const(self.chanidx)

            # Some channel averaging is to be applied so chbin_fn must not be None
            if chbin_fn is None:
                raise RuntimeError("chbin_fn is None whilst some channel averaging requested. Please yell at H. Verkouter (verkouter@jive.eu)")

            # depending on which kind of channel averaging, we apply it to the complex data before
            # producing quantities or on the scalars after computing the quantities
            if avgChannel == AVG.Scalar:
                self.scalarAvg = chbin_fn
            else:
                self.vectorAvg = chbin_fn

        if self.chanidx_fn is None:
            raise RuntimeError("The self.chanidx_fn is still None! Someone (verkouter@jive.eu) forgot something. Go yell at 'im!")

        # Now we must come up with a strategy for organizing the data processing chain
        #
        # If avgTime is Vector-like then we can only compute the quantities after all data's been
        # read & averaged. We've already ruled out that avgChannel == Scalar (for that makes no sense)
        #
        # So we may have to postpone computing of the quantities until after having collected + integrated all data
        post_quantities = lambda tp, x: [(tp, x)]
        org_quantities  = None
        if postpone:
            # create data sets based on the averaged data in a dataset
            org_quantities  = CP(self.quantities)
            post_quantities = lambda _, x: functional.map_(lambda q: (q.quantity_name, q.quantity_fn(x)), org_quantities)
            self.quantities = [Quantity('raw', functional.identity)]

        # Now that we've got an idea what our x-axis is going to be ('self.chanidx')
        # we can warn if we think it's a tad short)
        if len(self.chanidx)==1:
            print("WARNING: the output will contain only one channel, which might be an odd choice for your x-axis; it will be shorter than you think.")

        #if len(self.chanidx)==1:
        #    # post_channel doesn't do nothing, self.chanidx remains a list of length 1
        #    post_channel  = lambda ch, x: [(ch, x)]
        #else:
        #    # here the post_channel yields a list of extracted channels from the data 'x' coupled
        #    # with the assigned label from self.chanidx
        #    org_chanidx   = CP(self.chanidx)
        #    post_channel  = lambda _, x: map(lambda chi: (chi[1], x[:,chi[0]]), org_chanidx)
        #    # replace self.chanidx with a single entry which captures all channels and sets the
        #    # associated label to None - which we could use as a sentinel, if needed
        #    self.chanidx  = [(Ellipsis, None)]

        # Let's keep the channels together as long as possible. If only one channel remains then we can do it
        # in our inner loop
        #dataset_proto = dataset_list if avgTime == AVG.NoAveraging else dataset_solint_array
        #if avgTime == AVG.NoAveraging:
        #    # if no time averaging is to be done the (possibly channel averaged) data set IS the data set
        #    dataset_proto = dataset_list
        #else:
        #    # depending on wether we need to solint one or more channels in one go
        #    # loop over the current self.chanidx and count nr of channels
        #    nChannel = reduce(lambda acc, chi: acc + ((n_chan if chansel is Ellipsis else len(chansel)) if chi[0] is Ellipsis else 1), self.chanidx, 0)
        #    dataset_proto = dataset_solint_array if nChannel>1 else dataset_solint_scalar

        ## Now we can start the reduction of the table
        # Note: there will /always/ be WEIGHT+FLAGCOL - either read from the table or invented
        #          0        1      2      3      4       5     6
        fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.TIME]
        columns = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHTCOL", "FLAG_ROW", "FLAG", self.datacol]
        pts     =  ms2util.reducems2(self, self.table, collections.defaultdict(dataset_chan), columns, verbose=True, slicers=self.slicers, chunksize=5000)

        # after the reduction's done we can put back our quantities if we did remove them before
        if org_quantities is not None:
            self.quantities = org_quantities

        rv  = {}
        for (label, dataset) in iteritems(pts):
            dl = list(label)
            dataset.average( avgTime )
            # convert x,y to numarrays
            dataset.as_numarray()
            for qn,qd in post_quantities(label[0], dataset.y):
                dl[0] = qn
                rv[ self.MKLAB(fields, dl) ] = dataset_fixed(dataset.x, qd)

        return rv

    ## Here we make the plots
    def __call__(self, acc, a1, a2, tm, dd, fld, weight, flag_row, flag, data):
        # Create masked array from the data with invalid data already masked off
        data = numpy.ma.masked_invalid(data)
        # now we can easily add in flag information;
        # flags either has shape of data or it's a single bool False
        data.mask = numpy.logical_or(data.mask, self.untranspose_flag(numpy.logical_or(self.transpose_flag(flag), flag_row)))
        # weight handling. It's been set up such that whatever the weight was
        # (WEIGHT, WEIGHT_SPECTRUM, no weight thresholding) the following sequence
        # always works
        data      = self.transpose(data)
        data.mask = numpy.logical_or(data.mask, weight<self.threshold)
        data      = self.transpose(data)

        # possibly vector-average the data
        data      = self.vectorAvg(data)
        # Now create the quantity data - map the quantity functions over the (vector averaged)
        # data and, if needed, scalar average them
        qd        = functional.map_(lambda q: (q.quantity_name, self.scalarAvg(q.quantity_fn(data))), self.quantities)

        # Transform the time stamps, if necessary
        tm        = self.timebin_fn(tm)

        # Now we can loop over all the rows in the data
        dds  = self.ddSelection
        #ci   = self.chanidx
        cif  = self.chanidx_fn
        # We don't have to test *IF* the current data description id is
        # selected; the fact that we see it here means that it WAS selected!
        # The only interesting bit is selecting the correct products
        for row in range(data.shape[0]):
            ddid            = dd[row]
            (fq, sb, plist) = dds[ ddid ]
            #(fq, sb, plist) = dds[ dd[row] ]
            for (pidx, pname) in plist:
                l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, tm[row]]
                for (qnm, qval) in qd:
                    l[0] = qnm
                    acc[tuple(l)].add_y(cif(ddid), qval.data[row, :, pidx], qval.mask[row, :, pidx])
        return acc





### This plotter will iterate over "DATA" or "LAG_DATA"
### and produce a number of quantities per frequency
#class data_quantity_chan_old(plotbase):
#
#    ## our construct0r
#    ##   qlist = [ (quantity_name, quantity_fn), ... ]
#    ##
#    ##  Note that 'time averaging' will be implemented on a per-plot
#    ##  basis, not at the basic type of plot instance
#    def __init__(self, qlist, **kwargs):
#        self.quantities  = qlist
#        self.byFrequency = kwargs.get('byFrequency', False)
#
#    def makePlots(self, msname, selection, mapping, **kwargs):
#        datacol = CP(mapping.domain.column)
#
#        # Deal with time averaging
#        #   Scalar => average the derived quantity
#        #   Vector => compute average cplx number, then the quantity
#        avgTime = CP(selection.averageTime)
#        solint  = CP(selection.solint)
#        timerng = CP(selection.timeRange)
#
#        # need a function that (optionally) transforms the FQ/SB/CH idx to real frequencies
#        self.changeXaxis = lambda dd, chanidx: chanidx
#        if self.byFrequency:
#            if mapping.spectralMap is None:
#                raise RuntimeError("Request to plot by frequency but no spectral mapping available")
#            self.changeXaxis = lambda dd, chanidx: self.ddFreqs[ dd ][ chanidx ]
#
#        # solint must be >0.1 OR must be equal to None
#        # solint==None implies "aggregate all data into the selected time ranges in
#        #   their separate bins"
#        if avgTime!=AVG.NoAveraging and not (selection.solint is None or selection.solint>0.1):
#            raise RuntimeError("time averaging requested but solint is not none or >0.1: {0}".format(selection.solint))
#        # If solint is a number and averaging is not set, default to Scalar averaging
#        if selection.solint and avgTime==AVG.NoAveraging:
#            avgTime = AVG.Scalar
#            print "WARN: solint is set but no averaging method was specified. Defaulting to ",avgTime
#
#        if selection.averageChannel!=AVG.NoAveraging:
#            print "WARN: {0} channel averaging ignored for this plot".format(selection.averageChannel)
#
#        # If time averaging requested but solint==None and timerange==None, this means we
#        # have to set up a time range to integrate. timerange==None => whole data set
#        if avgTime!=AVG.NoAveraging and solint is None and timerng is None:
#            timerng = [(mapping.timeRange.start, mapping.timeRange.end)]
#
#        ## initialize the base class
#        super(data_quantity_chan, self).__init__(msname, selection, mapping, **kwargs)
#
#        ## Some variables must be stored in ourselves such
#        ## that they can be picked up by the callback function
#        slicers    = {}
#
#        # For data sets with a large number of channels
#        # (e.g. UniBoard data, 1024 channels spetral resolution)
#        # it makes a big (speed) difference if there is a channel
#        # selection to let the casa stuff [the ms column] do
#        # the (pre)selection so we do not get *all* the channels
#        # into casa
#
#        # 1) the channel selection. it is global; ie applies to
#        #    every data description id.
#        #    also allows us to create a slicer
#        #    default: iterate over all channels
#        shape           = self.table[0][datacol].shape
#        self.chunksize  = 5000
#        self.maskfn     = lambda x: numpy.ma.MaskedArray(x, mask=numpy.ma.nomask)
#        self.chanidx    = numpy.arange(shape[0])
#        self.chansel    = numpy.arange(shape[0])
#
#        # After having read the data, first we apply the masking function
#        # which disables the unselected channels
#        if selection.chanSel:
#            channels         = sorted(CP(selection.chanSel))
#            indices          = map_(lambda x: x-channels[0], channels)
#            self.chanidx     = numpy.array(channels, dtype=numpy.int32)
#            self.chansel     = numpy.array(indices, dtype=numpy.int32)
#            self.maskfn      = mk3dmask_fn_mask(self.chunksize, indices, shape[-1])
#            slicers[datacol] = ms2util.mk_slicer((channels[0],  0), (channels[-1]+1, shape[-1]))
#
#        # This is how we're going to go about dealing with time averaging
#        # The model is that, after having read the data, there is a function
#        # being called which produces (a list of) data products
#        #   * with scalar averaging, we produce a list of scalar quantities, the result
#        #     of calling self.quantities on the data. the .TYPE field in the data set
#        #     label is the actual quantity type
#        #   * with vector averaging, we produce nothing but the raw data itself; it is
#        #     the complex numbers that we must integrate/average. we give these data sets
#        #     the .TYPE of 'raw'.
#        #   * with no averaging at all, we also return the 'raw' data
#        #
#        # Then all the data is accumulated
#        # After the whole data set has been processed, we do the averaging and
#        # apply another transformation function:
#        #   * with scalar averaging, we don't have to do anything; the quantities have already
#        #     been produced
#        #   * with vector averaging, we take all data sets with type 'raw' and map our
#        #     quantity producing functions over the averaged data, producing new data sets
#        #     The raw data can now be deleted
#
#        # How integration/averaging actually is implemented is by modifying the
#        # time stamp.  By massaging the time stamp into buckets of size
#        # 'solint', we influence the label of the TIME field, which will make
#        # all data points with the same TIME stamp be integrated into the same
#        # data set
#        self.timebin_fn = lambda x: x
#        if avgTime!=AVG.NoAveraging:
#            if solint is None:
#                # Ah. Hmm. Have to integrate different time ranges
#                # Let's transform our timerng list of (start, end) intervals into
#                # a list of (start, end, mid) such that we can easily map
#                # all time stamps [start, end] to mid
#
#                # It is important to KNOW that "selection.timeRange" (and thus our
#                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
#                timerng = map_(lambda (s, e): (s, e, (s+e)/2.0), timerng)
#                self.timebin_fn = lambda x: \
#                        reduce(lambda acc, (s, e, m): numpy.put(acc, numpy.where((acc>=s) & (acc<=e)), m) or acc, timerng, x)
#            else:
#                # we have already checked the validity of solint
#                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0
#
#        # With no time averaging or with Scalar averaging, we can immediately produce
#        # the quantities. Only when doing Vector averaging, we must produce the quantities
#        # after having read all the data
#        self.preProcess = lambda x: map(lambda (qnm, qfn): (qnm, qfn(x)), self.quantities)
#        if avgTime in [AVG.Vector, AVG.Vectornorm]:
#            doNormalize     = (lambda x: x) if avgTime==AVG.Vector else (lambda x: x/numpy.abs(x))
#            self.preProcess = lambda x: [('raw', doNormalize(x))]
#
#        fields = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.TIME]
#
#        # weight filtering
#        self.nreject   = 0
#        self.reject_f  = lambda weight: False
#        self.threshold = -10000000
#        if selection.weightThreshold is not None:
#            self.threshold = CP(selection.weightThreshold)
#            self.reject_f  = lambda weight: weight<self.threshold
#
#        ## Now we can start the reduction of the table
#        if selection.weightThreshold is None:
#            columns        = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", datacol]
#            self.actual_fn = self.withoutWeightThresholding
#        else:
#            columns        = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHT", datacol]
#            self.actual_fn = self.withWeightThresholding
#        if self.flags:
#            columns.append( "FLAGCOL" )
#        pts     =  ms2util.reducems2(self, self.table, {}, columns, verbose=True, slicers=slicers, chunksize=self.chunksize)
#
#        if self.nreject:
#            print "Rejected ",self.nreject," points because of weight criterion"
#
#        ## Excellent. Now start post-processing
#        rv  = {}
#        for (label, ds) in iteritems(pts):
#            ds.average()
#            if label[0]=='raw':
#                dl = list(label)
#                for (qnm, qd) in map(lambda (qnm, qfn): (qnm, qfn(ds.y)), self.quantities):
#                    dl[0] = qnm
#                    rv[ self.MKLAB(fields, dl) ] = dataset(ds.x, qd, ds.m)
#            else:
#                rv[ self.MKLAB(fields, label) ] = ds
#        #for k in rv.keys():
#        #    print "Plot:",str(k),"/",map(str, rv[k].keys())
#        return rv
#
#
#    ## Here we make the plots
#    def __call__(self, *args):
#        return self.actual_fn(*args)
#
#    # This is the one WITHOUT WEIGHT THRESHOLDING
#    def withoutWeightThresholding(self, acc, a1, a2, tm, dd, fld, data, *flag):
#        # Make really sure we have a 3-D array of data ...
#        d3d  = m3d(data)
#        shp  = data.shape
#
#        # Good. We have a block of data, shape (nrow, nchan, npol)
#        # Step 1: apply the masking, such that any averaging later on
#        #         will skip the masked data.
#        #         'md' is "masked data"
#        #         Try to use the pre-computed channel mask, if it fits,
#        #         otherwise create one for this odd-sized block
#        #         (typically the last block)
#        mfn  = self.maskfn if shp[0]==self.chunksize else mk3dmask_fn_mask(shp[0], self.chansel, shp[2])
#
#        # Now create the quantity data
#        # qd will be a list of (quantity_name, quantity_data) tuples
#        #   original: qd = map_(lambda (qnm, qfn): (qnm, qfn(mfn(d3d))), self.quantities)
#        qd   = self.preProcess( mfn(d3d) )
#
#        # Transform the time stamps [rounds time to integer multiples of solint
#        # if that is set or the midpoint of the time range if solint was None]
#        tm   = self.timebin_fn( tm )
#        flag = flag[0] if flag else numpy.zeros(data.shape, dtype=numpy.bool)
#
#        # Now we can loop over all the rows in the data
#
#        # We don't have to test *IF* the current data description id is
#        # selected; the fact that we see it here means that it WAS selected!
#        # The only interesting bit is selecting the correct products
#        dds = self.ddSelection
#        cx  = self.changeXaxis
#        ci  = self.chanidx
#        cs  = self.chansel
#        for row in range(shp[0]):
#            ddr             = dd[row]
#            (fq, sb, plist) = dds[ ddr ]
#            # we can already precompute most of the label
#            # potentially, modify the TIME value to be a time bucket such
#            # that we can intgrate into it
#            l = ["", (a1[row], a2[row]), fq, sb, fld[row], "", tm[row]]
#            # we don't iterate over channels, only over polarizations
#            for (pidx, pname) in plist:
#                l[5] = pname
#                for (qnm, qval) in qd:
#                    l[0] = qnm
#                    acc.setdefault(tuple(l), dataset()).sumy(cx(ddr, ci), qval[row, cs, pidx], flag[row, cs, pidx])
#        return acc
#
#    # This is the one WITH WEIGHT THRESHOLDING
#    def withWeightThresholding(self, acc, a1, a2, tm, dd, fld, weight, data):
#        # Make really sure we have a 3-D array of data ...
#        d3d  = m3d(data)
#        shp  = data.shape
#
#        # compute weight mask
#        w3d  = numpy.zeros(shp, dtype=numpy.float)
#        for i in range_(shp[0]):
#            # we have weights per polzarization but we must
#            # expand them to per channel ...
#            cw = numpy.vstack( shp[1]*[weight[i]] )
#            w3d[i] = cw
#        w3m =  w3d<self.threshold
#        wfn = lambda a: numpy.ma.MaskedArray(a.data, numpy.logical_and(a.mask, w3m))
#
#        # Good. We have a block of data, shape (nrow, nchan, npol)
#        # Step 1: apply the masking, such that any averaging later on
#        #         will skip the masked data.
#        #         'md' is "masked data"
#        #         Try to use the pre-computed channel mask, if it fits,
#        #         otherwise create one for this odd-sized block
#        #         (typically the last block)
#        mfn  = self.maskfn if shp[0]==self.chunksize else mk3dmask_fn_mask(shp[0], self.chansel, shp[2])
#
#        # Now create the quantity data
#        # qd will be a list of (quantity_name, quantity_data) tuples
#        #   original: qd = map_(lambda (qnm, qfn): (qnm, qfn(mfn(d3d))), self.quantities)
#        qd   = self.preProcess( wfn(mfn(d3d)) )
#
#        # Transform the time stamps [rounds time to integer multiples of solint
#        # if that is set or the midpoint of the time range if solint was None]
#        tm   = self.timebin_fn( tm )
#        flag = flag[0] if flag else numpy.zeros(data.shape, dtype=numpy.bool)
#
#        # Now we can loop over all the rows in the data
#        dds = self.ddSelection
#        ci  = self.chanidx
#        cs  = self.chansel
#        cx  = self.changeXaxis
#        rf  = self.reject_f
#        # We don't have to test *IF* the current data description id is
#        # selected; the fact that we see it here means that it WAS selected!
#        # The only interesting bit is selecting the correct products
#        for row in range(shp[0]):
#            ddr             = dd[row]
#            (fq, sb, plist) = dds[ ddr ]
#            # we can already precompute most of the label
#            # potentially, modify the TIME value to be a time bucket such
#            # that we can intgrate into it
#            l = ["", (a1[row], a2[row]), fq, sb, fld[row], "", tm[row]]
#            # we don't iterate over channels, only over polarizations
#            for (pidx, pname) in plist:
#                if rf(w3d[row, 0, pidx]):
#                    self.nreject = self.nreject + 1
#                    continue
#                l[5] = pname
#                for (qnm, qval) in qd:
#                    l[0] = qnm
#                    acc.setdefault(tuple(l), dataset()).sumy(cx(ddr, ci), qval[row, cs, pidx], flag[row, cs, pidx])
#        return acc
#

class unflagged(object):
    def __getitem__(self, idx):
        return False

#class weight_time_old(plotbase):
#    def __init__(self):
#        # nothing yet ...
#        pass
#
#    def makePlots(self, msname, selection, mapping, **kwargs):
#        ## initialize the base class (opens table, does selection)
#        super(weight_time, self).__init__(msname, selection, mapping, **kwargs)
#
#        # Support "time averaging" by aggregating data points in time bins of 'solint' length
#        solint          = CP(selection.solint)
#        avgTime         = CP(selection.averageTime)
#        #solint_fn       = solint_none
#        self.timebin_fn = functional.identity
#        if avgTime!=AVG.NoAveraging:
#            if solint is None:
#                # Ah. Hmm. Have to integrate different time ranges
#                # Let's transform our timerng list of (start, end) intervals into
#                # a list of (start, end, mid) such that we can easily map
#                # all time stamps [start, end] to mid
#
#                # It is important to KNOW that "selection.timeRange" (and thus our
#                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
#                timerng = map_(lambda (s, e): (s, e, (s+e)/2.0), timerng)
#                if len(timerng)==1:
#                    print "WARNING: averaging all data into one point in time!"
#                    print "         This is because no solint was set. Your plot"
#                    print "         may contain less useful info than expected"
#
#                # try to be a bit optimized in time stamp replacement - filter the
#                # list of time ranges to those applying to the time stamps we're
#                # replacing
#                def do_it(x):
#                    mi,ma  = numpy.min(x), numpy.max(x)
#                    ranges = functional.filter_(lambda tr: not (tr[0]>ma or tr[1]<mi), timerng)
#                    return reduce(lambda acc, (s, e, m): numpy.put(acc, numpy.where((acc>=s) & (acc<=e)), m) or acc, ranges, x)
#                self.timebin_fn = do_it
#            else:
#                # Check if solint isn't too small
#                ti = mapping.timeRange.inttm[0]
#                if solint<ti:
#                    raise RuntimeError("solint value {0:.3f} is less than integration time {1:.3f}".format(solint, ti))
#                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0
#
#        self.dataset_proto = dataset_list if avgTime == AVG.NoAveraging else dataset_solint
#
#        ## we plot using the WEIGHT column
#
#        fields = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P]
#
#        #self.cnt = 0
#        #self.ts  = set()
#        ## Now we can start the reduction of the table
#        columns = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHT"] + ["FLAG_ROW"] if self.flags else []
#        pts     =  ms2util.reducems2(self, self.table, {}, columns, verbose=True, chunksize=5000)
#
#        #print "WE SHOULD HAVE ",self.cnt," DATA POINTS"
#        #print "ANDALSO ",len(self.ts)," TIME STAMPS"
#
#        rv  = {}
#        #dt  = 0.0
#        for (label, dataset) in iteritems(pts):
#            #dt += solint_fn( dataset )
#            dataset.average( avgTime )
#            rv[ self.MKLAB(fields, label) ] = dataset
#        #if solint:
#        #    print "SOLINT processing took\t{0:.3f}s".format( dt )
#        return rv
#
#    def __call__(self, acc, a1, a2, tm, dd, fld, weight, *flag_row):
#        #print "__call__: ",a1,a2,tm,dd,fld,weight.shape
#        # ok, process all the rows!
#        shp   = weight.shape
#        flags = unflagged() if not flag_row else flag_row[0]
#        # single-pol data will have shape (nrow,)
#        # but our code really would like it to be (nrow, npol), even if 'npol' == 1. (FFS casacore!)
#        tm  = self.timebin_fn( tm )
#        d2d = m2d(weight)
#        for row in range(shp[0]):
#            (fq, sb, plist) = self.ddSelection[ dd[row] ]
#            # we don't iterate over channels
#            for (pidx, pname) in plist:
#                acc.setdefault((YTypes.weight, (a1[row], a2[row]), fq, sb, fld[row], pname), self.dataset_proto()).append(tm[row], weight[row, pidx], flags[row])
#        return acc





class weight_time(plotbase):
    # should set up a choice table based on the combination of averaging methods
    # key into the lookup is '(avgChannelMethod, avgTimeMethod)'
    # For the weights we can remove a lot of entries:
    # no vector averaging applies here; weight IS a scalar.
    # Also no point in postponing because the weight IS the quantity;
    # it is not a derived quantity.
    # Note: we KEEP the postponed and have self.quantities be a list-of-quantities
    #       (currently 1 entry with the identity transform ...) because that way
    #       we could in the future have different flavours of this fn
    _averaging = {
        (AVG.NoAveraging,   AVG.NoAveraging):           (avg_none,       avg_none,       False),
        (AVG.NoAveraging,   AVG.Scalar):         (avg_none,       avg_arithmetic, False),
        (AVG.Scalar, AVG.NoAveraging):           (avg_arithmetic, avg_none,       False),
        (AVG.Scalar, AVG.Scalar):         (avg_arithmetic, avg_arithmetic, False),
        (AVG.NoAveraging,   AVG.Sum):            (avg_none,       avg_sum,        False),
        (AVG.Scalar, AVG.Sum):            (avg_arithmetic, avg_sum,        False),
        (AVG.Sum,    AVG.Sum):            (avg_sum,        avg_sum,        False),
    }


    ## our construct0r
    def __init__(self):
        # weight IS the quantity :-)
        self.quantities = [Quantity('weight', functional.identity)]

    def makePlots(self, msname, selection, mapping, **kwargs):
        # Deal with channel averaging
        #   Scalar => average the derived quantity
        avgChannel = CP(selection.averageChannel)
        avgTime    = CP(selection.averageTime)
        solchan    = CP(selection.solchan)
        solint     = CP(selection.solint)
        timerng    = CP(selection.timeRange)

        # some sanity checks
        if solchan is not None and avgChannel==AVG.NoAveraging:
            raise RuntimeError("nchav value was set without specifiying a channel averaging method; please tell me how you want them averaged")
        if solint is not None and avgTime==AVG.NoAveraging:
            raise RuntimeError("solint value was set without specifiying a time averaging method; please tell me how you want your time range(s) averaged")

        ## initialize the base class
        super(weight_time, self).__init__(msname, selection, mapping, **kwargs)

        n_chan   = self.table[0][self.datacol].shape[0]
        colname  = 'WEIGHT_SPECTRUM' if 'WEIGHT_SPECTRUM' in self.table.colnames() else 'WEIGHT'
        spectrum = (colname=='WEIGHT_SPECTRUM')
        # start of with default channel selection, if there is weight-per-channel anyhoos
        chansel  = Ellipsis if spectrum else None

        if selection.chanSel:
            if spectrum:
                channels = mk_chansel(selection.chanSel)
                max_chan = max(channels)
                # if any of the indexed channels > n_chan that's an error
                if max_chan>=n_chan:
                    raise RuntimeError("At least one selected channel ({0}) > largest channel index ({1})".format(max_chan, n_chan-1))
                # also <0 is not quite acceptable
                if min(channels)<0:
                    raise RuntimeError("Negative channel number {0} is not acceptable".format(min(channels)))
                # if the user selected all channels (by selection
                # 'ch 0:<last>' in stead of 'ch none' we don't
                # override the default channel selection (which is more efficient)
                if channels!=range(n_chan):
                    chansel = channels
                # ignore channel averaging if only one channel specified
                if (n_chan if chansel is Ellipsis else len(chansel))==1 and avgChannel != AVG.NoAveraging:
                    print("WARNING: channel averaging method {0} ignored because only one channel selected".format( avgChannel ))
                    avgChannel = AVG.NoAveraging
            else:
                # channel selection active but only WEIGHT column
                print("WARNING: you have selected channels but there is no WEIGHT_SPECTRUM column")
                print("         your channel selection will be IGNORED")
                chansel = None
                if avgChannel != AVG.NoAveraging:
                    print("WARNING: you specified {0} channel averaging".format(avgChannel))
                    print("         but there is no WEIGHT_SPECTRUM column so IGNORING channel averaging")
                    avgChannel = AVG.NoAveraging
        else:
            # no channels selected by user, check wether we actually *have* channels if averaging requested
            if avgChannel != AVG.NoAveraging and not spectrum:
                print("WARNING: you specified {0} channel averaging".format(avgChannel))
                print("         but there is no WEIGHT_SPECTRUM column so IGNORING channel averaging")
                avgChannel = AVG.NoAveraging

        # Test if the selected combination of averaging settings makes sense
        setup = weight_time._averaging.get((avgChannel, avgTime), None)
        if setup is None:
            raise RuntimeError("the combination of {0} channel + {1} time averaging is not supported".format(avgChannel, avgTime))
        (avgchan_fn, avgtime_fn, postpone) = setup

        # How integration/averaging actually is implemented is by modifying the
        # time stamp.  By massaging the time stamp into buckets of size
        # 'solint', we influence the label of the TIME field, which will make
        # all data points with the same TIME stamp be integrated into the same
        # data set
        self.timebin_fn = functional.identity
        if avgTime!=AVG.NoAveraging:
            if solint is None:
                # Ah. Hmm. Have to integrate different time ranges
                # Let's transform our timerng list of (start, end) intervals into
                # a list of (start, end, mid) such that we can easily map
                # all time stamps [start, end] to mid
                # If no time ranges defined at all average everything down to middle of experiment?

                # It is important to KNOW that "selection.timeRange" (and thus our
                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
                timerng = functional.map_(lambda s_e: (s_e[0], s_e[1], sum(s_e)/2.0),
                        timerng if timerng is not None else [(mapping.timeRange.start, mapping.timeRange.end)])
                if len(timerng)==1:
                    print("WARNING: averaging all data into one point in time!")
                    print("         This is because no solint was set or no time")
                    print("         ranges were selected to average. Your plot")
                    print("         may contain less useful info than expected")

                # try to be a bit optimized in time stamp replacement - filter the
                # list of time ranges to those applying to the time stamps we're
                # replacing
                def do_it(x):
                    mi,ma  = numpy.min(x), numpy.max(x)
                    ranges = functional.filter_(lambda tr: not (tr[0]>ma or tr[1]<mi), timerng)
                    return functional.reduce(lambda acc, s_e_m: numpy.put(acc, numpy.where((acc>=s_e_m[0]) & (acc<=s_e_m[1])), s_e_m[2]) or acc, ranges, x)
                self.timebin_fn = do_it
            else:
                # Check if solint isn't too small
                ti = mapping.timeRange.inttm[0]
                if solint<=ti:
                    raise RuntimeError("solint value {0:.3f} is less than integration time {1:.3f}".format(solint, ti))
                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0


        # chansel now is Ellipsis (all channels) or a list of some selected channels
        self.chanidx   = list()
        self.vectorAvg = functional.identity
        self.scalarAvg = functional.identity
        self.tmVectAvg = functional.identity
        self.tmScalAvg = functional.identity

        if avgChannel==AVG.NoAveraging:
            if spectrum:
                # No channel averaging - each selected channel goes into self.chanidx
                self.chanidx = list(enumerate(range(n_chan) if chansel is Ellipsis else chansel))
                # The vector average step will be misused to just apply the channel selection such that all selected channels
                # are mapped to 0..n-1. This is only necessary in case not all channels were selected
                if chansel is not Ellipsis:
                    self.vectorAvg = lambda x: x[:,chansel,:]
            else:
                # no channels at all
                self.chanidx = None
        else:
            # ok channel averaging requested
            # note that the setup code already verifies that it makes sense to average channels.
            # which is to say: if there is only the WEIGHT column then averaging will be disabled
            chbin_fn = None
            if solchan is None:
                # average all selected channels down to one
                # weight_spectrum array 'x' has shape (n_int, n_chan, n_pol)
                # average the selected channels according the requested averaging method
                chbin_fn      = lambda x: avgchan_fn(1)(x[:,chansel,:])
                self.chanidx  = [(0, '*')]
            else:
                # average bins of solchan channels down to one
                if solchan > n_chan:
                    raise RuntimeError("request to average channels in bins of {0} channels but only {1} are available".format(solchan, n_chan))
                # Create a mask which is the complement of the selected channels
                # (remember: chansel == Ellipsis => all channels
                ch_mask       = (numpy.zeros if chansel is Ellipsis else numpy.ones)(n_chan, dtype=numpy.bool)
                # only in case chansel != everything we must modify the mask
                if chansel is not Ellipsis:
                    ch_mask[chansel] = False

                # Since we're going to zap masked values (replace by 0) we can usefully use
                # reduceat! So all we then need is an index array, informing reduceat what the
                # reduction boundaries are!
                # First up: the actual bin numbers we're interested in, we compute the actual
                #           start + end indices from that
                bins    = numpy.unique((numpy.array(chansel) if chansel is not Ellipsis else numpy.arange(0, n_chan, solchan))//solchan)
                bins.sort()

                # we're going to apply channel binning so we must replace 'chansel'
                # by 'bins' in order for downstream accounting of how many "channels" there will
                # be in the data
                chansel = bins

                # Did timings on comparing simplistic 'loop over list of slices' and numpy.add.reduceat based approaches.
                # Results: grab bag - depending on problem set size:
                #      - simplistic approach between 2.5-6x faster (!) when averaging small number of
                #        channels in small number of bins (say <= 5 bins of ~5 channels)
                #      - reduceat approach slightly more than 2x faster when binning
                #        large number of channels in large-ish amount of bins (say 32 bins
                #        of 4 channels)

                # Update: the reduceat also only wins if all bins are adjacent.
                #         the way <operator>.reduceat works is, given a list of indices [i,j,k]
                #         and applied to an array A, it will produce the following outputs:
                #             [ <operator>(A[i:j]), <operator>(A[j:k]), <operator>(A[k:-1]) ]
                #         (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html)
                #
                #         Basically we can use this to efficiently bin i:j, j:k, ..., z:-1 ranges
                #         If our bins (or in the future, arbitrary channels ranges)  are NOT adjacent, then we must
                #         feed these operators to <operator>.reduceat:
                #             [ start0, end0, start1, end1, ..., startN, endN ]
                #         with the start, end indices of channel ranges 0..N
                #         will produce the following outputs:
                #             [ <operator>( A[start0:end0] ), <operator>( A[end0:start1] ), <operator>( A[start1:end1] ), ... ]
                #         so we'd have to throw out every second entry in the output.
                #         In numpy that's simple enough but it also means that <operator>.reduceat() does twice as must work
                #         for no apparent reason.

                # Detect if the bins are adjacent
                adjacent_bins = (len(set(bins[1:] - bins[:-1])) == 1) if len(bins)>1 else False
                chbin_fn      = None

                if adjacent_bins:
                    # we're going to use reduceat() which means it's good enough
                    # to generate [bin0*solchan, bin1*solchan, ..., bin<nbin-1>*solchan]
                    indices = CP(bins)
                    # generate the channel index labels for correct labelling
                    self.chanidx = list()
                    for (ch_idx, start) in enumerate(indices):
                        self.chanidx.append( (ch_idx, start) )
                    # need to carefully check last entry in there; if 'last bin' < 'n_chan//solchan'
                    # we must add an extra final boundary or else reduceat() will add up to the end
                    # of the number of channels in stead of until the end of the bin ...
                    if bins[-1]<((n_chan-1)//solchan):
                        # add one more bin limit, set slice to keep only n-1 bins
                        keepbins = slice(0, len(indices))
                        indices  = numpy.r_[indices, [indices[-1]+1]]
                    else:
                        keepbins = Ellipsis
                    # indices are in units of solchan bins so for reduceat must
                    # scale them back to actual channels
                    indices  *= solchan
                    # This is where the magic happens
                    transpose_ch = operator.methodcaller('transpose', (0, 2, 1))
                    def use_reduceat(x):
                        # (n_int, n_ch, n_pol) => (n_int, n_pol, n_ch)
                        tmpx = transpose_ch(x)
                        # also we must reshape it to a 2-D array of ((n_int * n_pol), n_chan) shape orelse
                        # the reduceat() don't work [https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html]
                        # remember the dimensions for later
                        n_int,n_pol = tmpx.shape[:-1]
                        tmpx        = tmpx.reshape( (n_int*n_pol, -1) )
                        # mask out channels that we don't want averaged, joining it
                        # with the mask that excludes flagged data (by the user) and/or
                        # whatever was weight-thresholded ...
                        tmpx.mask = LOGICAL_OR(tmpx.mask, ch_mask)
                        # set all masked values to 0 such that they don't ever count towards *anything*
                        # e.g. suppose all channels in a bin are masked then the average should be NaN or something
                        #      unrepresentable because there was no valid data at all
                        tmpx.data[tmpx.mask] = 0
                        # do the summation.
                        result = numpy.add.reduceat(tmpx.data, indices, axis=1)[:,keepbins]
                        # also count the number of unmasked values that went into each point
                        # we may use it for averaging, definitely be using it to create the mask
                        counts = numpy.add.reduceat(~tmpx.mask, indices, axis=1)[:,keepbins]
                        # pre-create the masked based on places where the count of unflagged points == 0
                        # these values have to be removed in the output (and also we can prevent
                        # divide-by-zero errors)
                        mask   = ARRAY(counts == 0, dtype=numpy.bool)
                        # ******************************************************************
                        # No vector* averaging for THIS data
                        # ******************************************************************
                        # ordinary arithmetic mean
                        # sets counts = 1 where counts == 0 so we don't divided by 0
                        counts[ mask ] = 1
                        result /= counts
#                        if avgchan_fn is avg_vectornorm:
#                            # ok need to find the maximum complex number in each bin to scale it by
#                            # take proper care of flagged/inf data
#                            tmpx.data[tmpx.mask] = -numpy.inf
#                            result /= (numpy.maximum.reduceat(numpy.abs(tmpx.data), indices, axis=1)[:,keepbins])
#                        else:
#                            # ordinary arithmetic mean
#                            # sets counts = 1 where counts == 0 so we don't divided by 0
#                            counts[ mask ] = 1
#                            result /= counts
                        # set entries where counts==0 to NaN to make it explicit
                        # that, mathematically speaking, there is nothing there
                        result[mask] = numpy.nan
                        # unshape + untranspose from 2-d ((n_int * n_pol), n_output_channels)
                        #                       into 3-d (n_int, n_pol, n_ouput_channels)
                        return transpose_ch(numpy.ma.array(result.reshape((n_int, n_pol, -1)), mask=mask.reshape((n_int, n_pol, -1))))
                    # set chbin_fn to use reduceat()
                    chbin_fn = use_reduceat
                else:
                    # not going to use reduceat() just bruteforcing over a list of slices()
                    # do some extra pre-processing for the simplistic approach
                    # it uses slice() indexing so we pre-create the slice objects for it
                    # for each range of channels to average we compute src and dst slice
                    indices = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                    slices  = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices)]
                    # for display + loopindexing create list of (array_index, "CH label") tuples
                    self.chanidx = list()
                    for (ch_idx, start_end) in enumerate(indices):
                        self.chanidx.append( (ch_idx, start_end[0]) )
                    n_slices = len(slices)
                    #  this is the simplistic approach
                    def use_dumbass_method(x):
                        # get an output array
                        n_int,_,n_pol = x.shape
                        result        = numpy.ma.empty((n_int, n_slices, n_pol), dtype=x.dtype)
                        for (dst_idx, src_idx) in slices:
                            result[:,dst_idx,:]      = numpy.ma.mean(x[:,src_idx,:], axis=1, keepdims=True)
                            result.mask[:,dst_idx,:] = (numpy.sum(x.mask[:,src_idx,:], axis=1, keepdims=True) == 0)
                        return result
                    # and set the channel bin function to use to this'un
                    chbin_fn = use_dumbass_method

            # Some channel averaging is to be applied so chbin_fn must not be None
            if chbin_fn is None:
                raise RuntimeError("chbin_fn is None whilst some channel averaging requested. Please yell at H. Verkouter (verkouter@jive.eu)")

            # depending on which kind of channel averaging, we apply it to the complex data before
            # producing quantities or on the scalars after computing the quantities
            if avgChannel == AVG.Scalar:
                self.scalarAvg = chbin_fn
            else:
                self.vectorAvg = chbin_fn

        # Now we must come up with a strategy for organizing the data processing chain
        #
        # If avgTime is Vector-like then we can only compute the quantities after all data's been
        # read & averaged. We've already ruled out that avgChannel == Scalar (for that makes no sense)
        #
        # So we may have to postpone computing of the quantities until after having collected + integrated all data
        post_quantities = lambda tp, x: [(tp, x)]
        org_quantities  = None
        if postpone:
            # create data sets based on the averaged data in a dataset
            org_quantities  = CP(self.quantities)
            post_quantities = lambda _, x: functional.map_(lambda q: (q.quantity_name, q.quantity_fn(x)), org_quantities)
            self.quantities = [Quantity('raw', functional.identity)]

        if self.chanidx is None or len(self.chanidx)==1:
            # post_channel doesn't do nothing, self.chanidx remains a list of length 1
            post_channel  = lambda ch, x: [(ch, x)]
        else:
            # here the post_channel yields a list of extracted channels from the data 'x' coupled
            # with the assigned label from self.chanidx
            org_chanidx   = CP(self.chanidx)
            post_channel  = lambda _, x: functional.map_(lambda chi: (chi[1], x[:,chi[0]]), org_chanidx)
            # replace self.chanidx with a single entry which captures all channels and sets the
            # associated label to None - which we could use as a sentinel, if needed
            self.chanidx  = [(Ellipsis, None)]

        # Let's keep the channels together as long as possible. If only one channel remains then we can do it
        # in our inner loop
        if avgTime == AVG.NoAveraging:
            dataset_proto = dataset_list
        else:
            # depending on wether we need to solint one or more channels in one go
            # loop over the current self.chanidx and count nr of channels
            if self.chanidx:
                nChannel = functional.reduce(lambda acc, chi: acc + ((n_chan if chansel is Ellipsis else len(chansel)) if chi[0] is Ellipsis else 1), self.chanidx, 0)
            else:
                nChannel = 1
            dataset_proto = dataset_solint_array if nChannel>1 else dataset_solint_scalar

        # Depending on wether we have channels or not, we choose different processing routines; one
        # having per-channel processing and one without
        if self.chanidx is None:
            # only WEIGHT column
            print("WARNING: weight-per-polarization (WEIGHT column) implies only FLAG_ROW column used for flags")
            fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P]
            columns = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHT", "FLAG_ROW"]
            self.actual_fn = self.process_weight
        else:
            # Note: there will /always/ be WEIGHT+FLAGCOL - either read from the table or invented
            #          0        1      2      3      4       5     6
            fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.CH]
            columns = ["ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID", "FIELD_ID", "WEIGHT_SPECTRUM", "FLAG_ROW", "FLAG"]
            self.actual_fn = self.process_weight_spectrum

        ## Now we can start the reduction of the table
        pts     =  ms2util.reducems2(self, self.table, collections.defaultdict(dataset_proto), columns, verbose=True, slicers=self.slicers, chunksize=5000)

        # after the reduction's done we can put back our quantities if we did remove them before
        if org_quantities is not None:
            self.quantities = org_quantities

        rv  = {}
        if self.chanidx:
            for (label, dataset) in iteritems(pts):
                dl = list(label)
                dataset.average( avgTime )
                # convert x,y to numarrays
                dataset.as_numarray()
                for qn,qd in post_quantities(label[0], dataset.y):
                    for chn,chd in post_channel(label[6], qd):
                        dl[0] = qn
                        dl[6] = chn
                        rv[ self.MKLAB(fields, dl) ] = dataset_fixed(dataset.x, chd)
        else:
            for (label, dataset) in iteritems(pts):
                dl = list(label)
                dataset.average( avgTime )
                # convert x,y to numarrays
                dataset.as_numarray()
                for qn,qd in post_quantities(label[0], dataset.y):
                    dl[0] = qn
                    rv[ self.MKLAB(fields, dl) ] = dataset_fixed(dataset.x, qd)
        return rv

    ## Here we make the plots
    def __call__(self, *args):
        return self.actual_fn(*args)

    # handle WEIGHT = (n_int, n_pol), no FLAG data
    def process_weight(self, acc, a1, a2, tm, dd, fld, weight, flag_row):
        # Create masked array from the data with invalid data already masked off
        data      = numpy.ma.masked_invalid(weight)
        # now we can easily add in flag information:
        # FLAG_ROW = (n_int) do logical_or with data.mask.T [data.mask.T = (n_int,n_pol).T == (n_pol, n_int)]
        #            so the FLAG_ROW(n_int) will broadcast nicely across polarizations
        data.mask = numpy.logical_or(data.mask.T, flag_row).T
        # weight handling. A lot easier for this one :-)
        data.mask = numpy.logical_or(data.mask, weight<self.threshold)

        # possibly vector-average the data
        data      = self.vectorAvg(data)
        # Now create the quantity data - map the quantity functions over the (vector averaged)
        # data and, if needed, scalar average them
        qd        = functional.map_(lambda q: (q.quantity_name, self.scalarAvg(q.quantity_fn(data))), self.quantities)

        # Transform the time stamps, if necessary
        tm        = self.timebin_fn(tm)

        # Now we can loop over all the rows in the data
        dds  = self.ddSelection
        # We don't have to test *IF* the current data description id is
        # selected; the fact that we see it here means that it WAS selected!
        # The only interesting bit is selecting the correct products
        for row in range(data.shape[0]):
            (fq, sb, plist) = dds[ dd[row] ]
            for (pidx, pname) in plist:
                l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname]
                for (qnm, qval) in qd:
                    l[0] = qnm
                    acc[tuple(l)].append(tm[row], qval.data[row, pidx], qval.mask[row, pidx])
        return acc

    # handle WEIGHT_SPECTRUM = (n_int, n_chan, n_pol)
    def process_weight_spectrum(self, acc, a1, a2, tm, dd, fld, weight_spectrum, flag_row, flag):
        # Create masked array from the data with invalid data already masked off
        data      = numpy.ma.masked_invalid(weight_spectrum)
        # now we can easily add in flag information;
        # flags either has shape of data or it's a single bool False
        data.mask = numpy.logical_or(data.mask, self.untranspose_flag(numpy.logical_or(self.transpose_flag(flag), flag_row)))
        # weight thresholding - is easier for this one!
        data.mask = numpy.logical_or(data.mask, weight_spectrum<self.threshold)

        # possibly vector-average the data
        data      = self.vectorAvg(data)
        # Now create the quantity data - map the quantity functions over the (vector averaged)
        # data and, if needed, scalar average them
        qd        = functional.map_(lambda q: (q.quantity_name, self.scalarAvg(q.quantity_fn(data))), self.quantities)

        # Transform the time stamps, if necessary
        tm        = self.timebin_fn(tm)

        # Now we can loop over all the rows in the data
        dds  = self.ddSelection
        ci   = self.chanidx
        # We don't have to test *IF* the current data description id is
        # selected; the fact that we see it here means that it WAS selected!
        # The only interesting bit is selecting the correct products
        for row in range(data.shape[0]):
            (fq, sb, plist) = dds[ dd[row] ]
            for (chi, chn) in ci:
                for (pidx, pname) in plist:
                    l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, chn]
                    for (qnm, qval) in qd:
                        l[0] = qnm
                        acc[tuple(l)].append(tm[row], qval.data[row, chi, pidx], qval.mask[row, chi, pidx])
        return acc



class uv(plotbase):
    def __init__(self):
        # nothing yet ...
        pass

    _sep = "\n\t - "
    def makePlots(self, msname, selection, mapping, **kwargs):
        ## initialize the base class (opens table, does selection)
        super(uv, self).__init__(msname, selection, mapping, **kwargs)

        # warn about any averaging or channel selection

        print_if("WARNING: Ignoring the following settings:"+uv._sep + uv._sep.join(
                 map(lambda tup: tup[0](tup[1]),
                     filter(lambda tup: tup[1] not in [AVG.NoAveraging, None],
                            zip(["your channel selection".format, "solint {0}".format, "{0} avg in time".format, "nchav of {0}".format, "{0} avg in frequency".format, "weight threshold of {0}".format],
                                [selection.chanSel, selection.solint, selection.averageTime, selection.solchan, selection.averageChannel, selection.weightThreshold])))))

        ## we plot using the UVW column
        ## UVW is not a function of POL (it should be a function
        ##     of CH but that would mean we'd have to actually
        ##     do computations - yikes)
        fields = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC]

        ## Now we can start the reduction of the table
        # The base class will have set up FLAG/FLAG_ROW accessors based on wether the user
        # specified reading flags or not. We can just use the transpose/untranspose functions
        # and expect them to Do The Right Thing (tm)
        columns = ["ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "FIELD_ID", "UVW", "FLAG_ROW"]
        pts     =  ms2util.reducems2(self, self.table, {}, columns, verbose=True, chunksize=5000)

        rv  = {}
        for (label, dataset) in iteritems(pts):
            rv[ self.MKLAB(fields, label) ] = dataset
        return rv

    def __call__(self, acc, a1, a2, dd, fld, uvw, row_flag):
        # transform the data into a masked array with inf/nan masked off
        uvw      = numpy.ma.masked_invalid(uvw)
        # uvw = (nrow, 3), flag_row = (nrow)
        # so to broadcast row_flag across the mask we use uvw.T [shape = (3, nrow)]
        uvw.mask = numpy.logical_or(uvw.mask.T, row_flag).T
        # now condense the (nrow, [u,v,w], dtype=bool) array to (nrow, [uflag || vflag]) [shape: (nrow,1)]:
        # one flag for the (u,v) data point: if either u or v was nan/inf or the row was
        # flagged, flag that datapoint
        row_flag = numpy.logical_or(uvw.mask[:,0], uvw.mask[:,1])
        u        = uvw.data[:,0]
        v        = uvw.data[:,1]
        # ok, process all the rows!
        for row in range(uvw.shape[0]):
            (fq, sb, _plist) = self.ddSelection[ dd[row] ]
            # we don't iterate over channels nor over polarizations
            ds = acc.setdefault(('V', (a1[row], a2[row]), fq, sb, fld[row]), dataset_list())
            f  = row_flag[row]
            ds.append( u[row],  v[row], f)
            ds.append(-u[row], -v[row], f)
        return acc




## This plotter will iterate over "DATA" or "LAG_DATA"
## and produce a number of quantities per frequency, possibly averaging channels (no time avg yet)
class data_quantity_uvdist(plotbase):

    # should set up a choice table based on the combination of averaging methods
    # key into the lookup is '(avgChannelMethod, avgTimeMethod)'
    # Also return wether the quantities must be postponed
    _averaging = {
        # no averaging at all, no need to postpone computing the quantity(ies)
        (AVG.NoAveraging, AVG.NoAveraging):             (avg_none, avg_none, False),
        (AVG.Scalar, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Vector, AVG.NoAveraging):           (avg_arithmetic, avg_none, False),
        (AVG.Vectornorm, AVG.NoAveraging):       (avg_vectornorm, avg_none, False),
        (AVG.Sum, AVG.NoAveraging):              (avg_sum, avg_none,        False),
        (AVG.Vectorsum, AVG.NoAveraging):        (avg_sum, avg_none,        False),
    }


    ## our construct0r
    ##   qlist = [ (quantity_name, quantity_fn), ... ]
    ##
    def __init__(self, qlist, **kwargs):
        self.quantities  = list(itertools.starmap(Quantity, qlist))
        # UV distance is frequency dependent
        self.byFrequency = True

    def makePlots(self, msname, selection, mapping, **kwargs):
        # Deal with channel averaging
        #   Scalar => average the derived quantity
        #   Vector => compute average cplx number, then the quantity
        avgChannel = CP(selection.averageChannel)
        avgTime    = CP(selection.averageTime)
        solchan    = CP(selection.solchan)
        solint     = CP(selection.solint)
        timerng    = CP(selection.timeRange)

        ## initialize the base class
        super(data_quantity_uvdist, self).__init__(msname, selection, mapping, **kwargs)

        # channel selection+averaging schemes; support averaging over channels (or chunks of channels)
        chansel  = Ellipsis #None
        n_chan   = self.table[0][self.datacol].shape[0]
        if selection.chanSel:
            channels = list(sorted(set(CP(selection.chanSel))))
            max_chan = max(channels)
            # if any of the indexed channels > n_chan that's an error
            if max_chan>=n_chan:
                raise RuntimeError("At least one selected channel ({0}) > largest channel index ({1})".format(max_chan, n_chan-1))
            # also <0 is not quite acceptable
            if min(channels)<0:
                raise RuntimeError("Negative channel number {0} is not acceptable".format(min(channels)))
            # if the user selected all channels (by selection
            # 'ch 0:<last>' in stead of 'ch none' we don't
            # override the default channel selection (which is more efficient)
            if channels!=range(n_chan):
                chansel = channels
            # ignore channel averaging if only one channel specified
            if (n_chan if chansel is Ellipsis else len(chansel))==1 and avgChannel != AVG.NoAveraging:
                print("WARNING: channel averaging method {0} ignored because only one channel selected".format( avgChannel ))
                avgChannel = AVG.NoAveraging

        # Test if the selected combination of averaging settings makes sense
        setup = data_quantity_time._averaging.get((avgChannel, avgTime), None)
        if setup is None:
            raise RuntimeError("the combination of {0} channel + {1} time averaging is not supported".format(avgChannel, avgTime))
        (avgchan_fn, avgtime_fn, postpone) = setup

        # some sanity checks
        if solchan is not None and avgChannel==AVG.NoAveraging:
            raise RuntimeError("nchav value was set without specifiying a channel averaging method; please tell me how you want them averaged")

        # How integration/averaging actually is implemented is by modifying the
        # time stamp.  By massaging the time stamp into buckets of size
        # 'solint', we influence the label of the TIME field, which will make
        # all data points with the same TIME stamp be integrated into the same
        # data set
        self.timebin_fn = functional.identity
        # currently we don't support time averaging!
#        if avgTime!=AVG.NoAveraging:
#            if solint is None:
#                # Ah. Hmm. Have to integrate different time ranges
#                # Let's transform our timerng list of (start, end) intervals into
#                # a list of (start, end, mid) such that we can easily map
#                # all time stamps [start, end] to mid
#                # If no time ranges defined at all average everything down to middle of experiment?
#
#                # It is important to KNOW that "selection.timeRange" (and thus our
#                # local copy 'timerng') is a list or sorted, non-overlapping time ranges
#                timerng = map_(lambda (s, e): (s, e, (s+e)/2.0), timerng if timerng is not None else [(mapping.timeRange.start, mapping.timeRange.end)])
#
#                # try to be a bit optimized in time stamp replacement - filter the
#                # list of time ranges to those applying to the time stamps we're
#                # replacing
#                def do_it(x):
#                    mi,ma  = numpy.min(x), numpy.max(x)
#                    ranges = functional.filter_(lambda tr: not (tr[0]>ma or tr[1]<mi), timerng)
#                    return reduce(lambda acc, (s, e, m): numpy.put(acc, numpy.where((acc>=s) & (acc<=e)), m) or acc, ranges, x)
#                self.timebin_fn = do_it
#            else:
#                # Check if solint isn't too small
#                ti = mapping.timeRange.inttm[0]
#                if solint<=ti:
#                    raise RuntimeError("solint value {0:.3f} is less than integration time {1:.3f}".format(solint, ti))
#                self.timebin_fn = lambda x: (numpy.trunc(x/solint)*solint) + solint/2.0

        # chansel now is Ellipsis (all channels) or a list of some selected channels
        self.chanidx   = list()
        self.chanidx_fn= None
        self.vectorAvg = functional.identity
        self.scalarAvg = functional.identity
        self.tmVectAvg = functional.identity
        self.tmScalAvg = functional.identity

        # the x-axis is frequency ... but we pre-convert to a multiplication factor
        # 1/lambda ( nu / c = 1 / lambda) such that going from baseline length in m
        # to baseline length in wavelengths is as easy as multiplying by 1 / lambda
        self.freq_of_dd=CP(self.ddFreqs)
        # transform all channel frequencies to 1 / lambda
        # NOTE: frequencies in this mapping are in units of MHz!!!!
        for dd in self.freq_of_dd.keys():
            self.freq_of_dd[dd] = (ARRAY(self.freq_of_dd[dd]) * 1e6) / 299792458.0

        if avgChannel==AVG.NoAveraging:
            # No channel averaging - the new x-axis will be the indices of the selected channels
            # [0,1,...,n-1] if ellipsis else [3, 6, 12, 128] (selected channels)
            self.chanidx = range(n_chan) if chansel is Ellipsis else chansel
            # The vector average step will be misused to just apply the channel selection such that all selected channels
            # are mapped to 0..n-1. This is only necessary in case not all channels were selected
            if chansel is not Ellipsis:
                self.vectorAvg  = lambda x: x[:,chansel,:] # collapses selection to indices [0, 1, ..., n-1]
                self.chanidx_fn = lambda dd: self.freq_of_dd[dd][chansel]
            else:
                self.chanidx_fn = self.freq_of_dd.__getitem__
        else:
            # ok channel averaging requested
            chbin_fn = None
            if solchan is None:
                # average all selected channels down to one
                # data array 'x' has shape (n_int, n_chan, n_pol)
                #self.chbin_fn = lambda x: normalize_ch(1)(numpy.ma.mean(x[:,avg_over,:], axis=1, keepdims=True))
                # average the selected channels according the requested averaging method
                chbin_fn      = lambda x: avgchan_fn(1)(x[:,chansel,:])
                self.chanidx  = [numpy.mean(range(n_chan) if chansel is Ellipsis else chansel)]
                # transform all frequencies to an average frequency
                for dd in self.freq_of_dd.keys():
                    self.freq_of_dd[dd] = [numpy.mean(self.freq_of_dd[dd][chansel])]
                self.chanidx_fn = self.freq_of_dd.__getitem__
            else:
                # average bins of solchan channels down to one
                if solchan > n_chan:
                    raise RuntimeError("request to average channels in bins of {0} channels but only {1} are available".format(solchan, n_chan))
                # Create a mask which is the complement of the selected channels
                # (remember: chansel == Ellipsis => all channels
                ch_mask       = (numpy.zeros if chansel is Ellipsis else numpy.ones)(n_chan, dtype=numpy.bool)
                # only in case chansel != everything we must modify the mask
                if chansel is not Ellipsis:
                    ch_mask[chansel] = False

                # Since we're going to zap masked values (replace by 0) we can usefully use
                # reduceat! So all we then need is an index array, informing reduceat what the
                # reduction boundaries are!
                # First up: the actual bin numbers we're interested in, we compute the actual
                #           start + end indices from that
                bins    = numpy.unique((numpy.array(chansel) if chansel is not Ellipsis else numpy.arange(0, n_chan, solchan))//solchan)
                bins.sort()

                # we're going to apply channel binning so we must replace 'chansel'
                # by 'bins' in order for downstream accounting of how many "channels" there will
                # be in the data
                chansel = bins

                # Detect if the bins are adjacent
                adjacent_bins = (len(set(bins[1:] - bins[:-1])) == 1) if len(bins)>1 else False

                chbin_fn      = None
                if adjacent_bins:
                    # we're going to use reduceat() which means it's good enough
                    # to generate [bin0*solchan, bin1*solchan, ..., bin<nbin-1>*solchan]
                    indices = CP(bins)
                    # generate the channel index labels for correct labelling
                    self.chanidx = CP(indices) #list()
                    # need to carefully check last entry in there; if 'last bin' < 'n_chan//solchan'
                    # we must add an extra final boundary or else reduceat() will add up to the end
                    # of the number of channels in stead of until the end of the bin ...
                    if bins[-1]<((n_chan-1)//solchan):
                        # add one more bin limit, set slice to keep only n-1 bins
                        keepbins = slice(0, len(indices))
                        indices  = numpy.r_[indices, [indices[-1]+1]]
                    else:
                        keepbins = Ellipsis
                    # indices are in units of solchan bins so for reduceat must
                    # scale them back to actual channels
                    indices  *= solchan
                    # This is where the magic happens
                    transpose_ch = operator.methodcaller('transpose', (0, 2, 1))
                    def use_reduceat(x):
                        # (n_int, n_ch, n_pol) => (n_int, n_pol, n_ch)
                        tmpx = transpose_ch(x)
                        # also we must reshape it to a 2-D array of ((n_int * n_pol), n_chan) shape orelse
                        # the reduceat() don't work [https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html]
                        # remember the dimensions for later
                        n_int,n_pol = tmpx.shape[:-1]
                        tmpx        = tmpx.reshape( (n_int*n_pol, -1) )
                        # mask out channels that we don't want averaged, joining it
                        # with the mask that excludes flagged data (by the user) and/or
                        # whatever was weight-thresholded ...
                        tmpx.mask = numpy.logical_or(tmpx.mask, ch_mask)
                        # set all masked values to 0 such that they don't ever count towards *anything*
                        # e.g. suppose all channels in a bin are masked then the average should be NaN or something
                        #      unrepresentable because there was no valid data at all
                        tmpx.data[tmpx.mask] = 0
                        # do the summation.
                        #print("###############################################################")
                        #print("    use_reduceat: x.shape=",x.shape)
                        #print("          tmpx.data.shape=",tmpx.data.shape)
                        #print("                  indices=",indices)
                        #print("                 keepbins=",keepbins)
                        #print("###############################################################")
                        result = numpy.add.reduceat(tmpx.data, indices, axis=1)[:,keepbins]
                        # also count the number of unmasked values that went into each point
                        # we may use it for averaging, definitely be using it to create the mask
                        counts = numpy.add.reduceat(~tmpx.mask, indices, axis=1)[:,keepbins]
                        # Because we do things different here than in the ordinary averaging,
                        # we must look at what was requested in order to mimic that behaviour
                        if avgchan_fn is avg_vectornorm:
                            # ok need to find the maximum complex number in each bin to scale it by
                            result /= (numpy.maximum.reduceat(numpy.abs(tmpx.data), indices, axis=1)[:,keepbins])
                        elif avgchan_fn in [avg_sum, avg_none]:
                            # no averaging/summing boils down to not doing anything
                            pass
                        else:
                            # ordinary arithmetic mean
                            result /= counts
                        # return masked array - can reuse the counts array by converting them
                        # to bool and inverting: no counts => False
                        mask = numpy.array(counts == 0, dtype=numpy.bool)
                        # set entries where counts==0 to NaN to make it explicit
                        # that, mathematically speaking, there is nothing there
                        result[mask] = numpy.nan
                        # unshape + untranspose from 2-d ((n_int * n_pol), n_output_channels)
                        #                       into 3-d (n_int, n_pol, n_ouput_channels)
                        return transpose_ch(numpy.ma.array(result.reshape((n_int, n_pol, -1)), mask=mask.reshape((n_int, n_pol, -1))))
                    # set chbin_fn to use reduceat()
                    chbin_fn = use_reduceat
                else:
                    # not going to use reduceat() just bruteforcing over a list of slices()
                    # do some extra pre-processing for the simplistic approach
                    # it uses slice() indexing so we pre-create the slice objects for it
                    # for each range of channels to average we compute src and dst slice
                    indices = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                    slices  = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices)]
                    # for display + loopindexing create list of (array_index, "CH label") tuples
                    self.chanidx = CP(bins)
                    n_slices = len(slices)
                    #  this is the simplistic approach
                    def use_dumbass_method(x):
                        # get an output array
                        n_int,_,n_pol = x.shape
                        result        = numpy.ma.empty((n_int, n_slices, n_pol), dtype=x.dtype)
                        for (dst_idx, src_idx) in slices:
                            result[:,dst_idx,:]      = numpy.ma.mean(x[:,src_idx,:], axis=1, keepdims=True)
                            result.mask[:,dst_idx,:] = (numpy.sum(x.mask[:,src_idx,:], axis=1, keepdims=True) == 0)
                        return result
                    # and set the channel bin function to use to this'un
                    chbin_fn = use_dumbass_method

                # transform all frequencies to an average frequency per bin
                indices_l  = functional.map_(lambda s: (s*solchan, min((s+1)*solchan, n_chan)), bins)
                slices_l   = [(slice(i, i+1), slice(rng[0], rng[1]+1)) for i, rng in enumerate(indices_l)]
                def mk_fbins(x):
                    result = numpy.empty((len(slices_l)))
                    for (dst_idx, src_idx) in slices_l:
                        result[dst_idx]      = numpy.mean(x[src_idx])
                    return result
                for dd in self.freq_of_dd.keys():
                    self.freq_of_dd[dd] = mk_fbins( ARRAY(self.freq_of_dd[dd]) )
                self.chanidx_fn = self.freq_of_dd.__getitem__

            # Some channel averaging is to be applied so chbin_fn must not be None
            if chbin_fn is None:
                raise RuntimeError("chbin_fn is None whilst some channel averaging requested. Please yell at H. Verkouter (verkouter@jive.eu)")

            # depending on which kind of channel averaging, we apply it to the complex data before
            # producing quantities or on the scalars after computing the quantities
            if avgChannel == AVG.Scalar:
                self.scalarAvg = chbin_fn
            else:
                self.vectorAvg = chbin_fn

        if self.chanidx_fn is None:
            raise RuntimeError("The self.chanidx_fn is still None! Someone (verkouter@jive.eu) forgot something. Go yell at 'im!")

        # Now we must come up with a strategy for organizing the data processing chain
        #
        # If avgTime is Vector-like then we can only compute the quantities after all data's been
        # read & averaged. We've already ruled out that avgChannel == Scalar (for that makes no sense)
        #
        # So we may have to postpone computing of the quantities until after having collected + integrated all data
        post_quantities = lambda tp, x: [(tp, x)]
        org_quantities  = None
        if postpone:
            # create data sets based on the averaged data in a dataset
            org_quantities  = CP(self.quantities)
            post_quantities = lambda _, x: functional.map_(lambda q: (q.quantity_name, q.quantity_fn(x)), org_quantities)
            self.quantities = [Quantity('raw', functional.identity)]

        ## Now we can start the reduction of the table
        # Note: there will /always/ be WEIGHT+FLAGCOL - either read from the table or invented
        #          0        1      2      3      4       5     6
        #fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.CH]
        fields  = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P]
        columns = ["ANTENNA1", "ANTENNA2", "UVW", "DATA_DESC_ID", "FIELD_ID", "WEIGHTCOL", "FLAG_ROW", "FLAG", self.datacol]
        pts     =  ms2util.reducems2(self, self.table, collections.defaultdict(dataset_list), columns, verbose=True, slicers=self.slicers, chunksize=5000)

        # after the reduction's done we can put back our quantities if we did remove them before
        if org_quantities is not None:
            self.quantities = org_quantities

        rv  = {}
        for (label, dataset) in iteritems(pts):
            dl = list(label)
            #dataset.average( avgTime )
            # convert x,y to numarrays
            dataset.as_numarray()
            for qn,qd in post_quantities(label[0], dataset.y):
                dl[0] = qn
                rv[ self.MKLAB(fields, dl) ] = dataset_fixed(dataset.x, qd)

        return rv

    ## Here we make the plots
    def __call__(self, acc, a1, a2, uvw, dd, fld, weight, flag_row, flag, data):
        # Create masked array from the data with invalid data already masked off
        data = numpy.ma.masked_invalid(data)
        # now we can easily add in flag information;
        # flags either has shape of data or it's a single bool False
        data.mask = numpy.logical_or(data.mask, self.untranspose_flag(numpy.logical_or(self.transpose_flag(flag), flag_row)))
        # weight handling. It's been set up such that whatever the weight was
        # (WEIGHT, WEIGHT_SPECTRUM, no weight thresholding) the following sequence
        # always works
        data      = self.transpose(data)
        data.mask = numpy.logical_or(data.mask, weight<self.threshold)
        data      = self.transpose(data)

        # possibly vector-average the data
        data      = self.vectorAvg(data)
        # Now create the quantity data - map the quantity functions over the (vector averaged)
        # data and, if needed, scalar average them
        qd        = functional.map_(lambda q: (q.quantity_name, self.scalarAvg(q.quantity_fn(data))), self.quantities)

        # Transform uvw column into uvw distance. Apparently ...
        # older numpy's have a numpy.linalg.norm() that does NOT take an 'axis' argument
        # so we have to write the distance computation out ourselves. #GVD
        uvw       = SQRT( ADD( SQUARE(uvw[:,0]), SQUARE(uvw[:,1]) ) )

        # Now we can loop over all the rows in the data
        dds  = self.ddSelection
        #ci   = range(len(self.chanidx))
        cif  = self.chanidx_fn
        # We don't have to test *IF* the current data description id is
        # selected; the fact that we see it here means that it WAS selected!
        # The only interesting bit is selecting the correct products
        for row in range(data.shape[0]):
            ddid            = dd[row]
            (fq, sb, plist) = dds[ ddid ]
            for (pidx, pname) in plist:
                l   = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, ""]
                # transform projected baseline length to wavelength for the current spectral window
                uvd = uvw[row] * cif(ddid)
                for (qnm, qval) in qd:
                    l[0] = qnm
                    acc[tuple(l)].extend(uvd, qval.data[row,:,pidx], qval.mask[row,:,pidx])
                    # loop over the channels; self.chanidx = [ChX, ChY, ChZ]
                    #for i in ci:
                    #    acc[tuple(l)].append(uvd[i], qval.data[row, i, pidx], qval.mask[row, i, pidx])
        return acc


### This plotter will iterate over "DATA" or "LAG_DATA"
### and produce a number of quantities per data point
#class data_quantity_uvdist_old(plotbase):
#
#    ## our construct0r
#    ##   qlist = [ (quantity_name, quantity_fn), ... ]
#    ##
#    ##  Note that 'channel averaging' will be implemented on a per-plot
#    ##  basis, not at the basic type of plot instance
#    def __init__(self, qlist):
#        self.quantities = qlist
#
#    def makePlots(self, msname, selection, mapping, **kwargs):
#        datacol = CP(mapping.domain.column)
#
#        # Deal with channel averaging
#        #   Scalar => average the derived quantity
#        #   Vector => compute average cplx number, then the quantity
#        avgChannel = CP(selection.averageChannel)
#
#        if selection.averageTime!=AVG.NoAveraging:
#            print "Warning: {0} time averaging ignored for this plot".format(selection.averageTime)
#
#        ## initialize the base class
#        super(data_quantity_uvdist, self).__init__(msname, selection, mapping, **kwargs)
#
#        ## Some variables must be stored in ourselves such
#        ## that they can be picked up by the callback function
#        slicers    = {}
#
#        # For data sets with a large number of channels
#        # (e.g. UniBoard data, 1024 channels spetral resolution)
#        # it makes a big (speed) difference if there is a channel
#        # selection to let the casa stuff [the ms column] do
#        # the (pre)selection so we do not get *all* the channels
#        # into casa
#
#        # 1) the channel selection. it is global; ie applies to
#        #    every data description id.
#        #    also allows us to create a slicer
#        #    default: iterate over all channels
#        shape           = self.table[0][datacol].shape
#        self.chunksize  = 5000
#        self.chanidx    = zip(range(shape[0]), range(shape[0]))
#        self.maskfn     = lambda x: numpy.ma.MaskedArray(x, mask=numpy.ma.nomask)
#        self.chansel    = range(shape[0])
#
#        # We must translate the selected channels to a frequency (or wavelength) - such that we can
#        # compute the uvdist in wavelengths
#        _spwMap  = mapping.spectralMap
#        ddids    = _spwMap.datadescriptionIDs()
#
#        # preallocate an array of dimension (nDDID, nCHAN) such that we can put
#        # the frequencies of DDID #i at row i - makes for easy selectin'
#        self.factors = numpy.zeros((max(ddids)+1, shape[0]))
#        for ddid in ddids:
#            fqobj                = _spwMap.unmap( ddid )
#            self.factors[ ddid ] = _spwMap.frequenciesOfFREQ_SB(fqobj.FREQID, fqobj.SUBBAND)
#
#        # After having read the data, first we apply the masking function
#        # which disables the unselected channels
#        if selection.chanSel:
#            channels         = sorted(CP(selection.chanSel))
#            indices          = map_(lambda x: x-channels[0], channels)
#            self.chanidx     = zip(indices, channels)
#            self.chansel     = indices
#            # select only the selected channels
#            self.factors     = self.factors[:, channels]
#            #print "channels=",channels," indices=",indices," self.chanidx=",self.chanidx
#            self.maskfn      = mk3dmask_fn_mask(self.chunksize, indices, shape[-1])
#            slicers[datacol] = ms2util.mk_slicer((channels[0],  0), (channels[-1]+1, shape[-1]))
#
#        # right - factors now contain *frequency*
#        # divide by speed of lite to get the multiplication factor
#        # to go from UV distance in meters to UV dist in lambda
#        self.factors /= 299792458.0
#        # older numpy's have a numpy.linalg.norm() that does NOT take an 'axis' argument
#        # so we have to write the distance computation out ourselves. #GVD
#        self.uvdist_f = lambda uvw: numpy.sqrt( numpy.square(uvw[...,0]) + numpy.square(uvw[...,1]) )
#
#        # If there is vector averaging to be done, this is done in the step after reading the data
#        # (default: none)
#        self.vectorAvg  = lambda x: x
#
#        if avgChannel==AVG.Vector:
#            self.vectorAvg = lambda x: numpy.average(x, axis=1).reshape( (x.shape[0], 1, x.shape[2]) )
#            self.chanidx   = [(0, '*')]
#
#        # Scalar averaging is done after the quantities have been computed
#        self.scalarAvg  = lambda x: x
#
#        if avgChannel==AVG.Scalar:
#            self.scalarAvg = lambda x: numpy.average(x, axis=1).reshape( (x.shape[0], 1, x.shape[2]) )
#            self.chanidx   = [(0, '*')]
#
#        if avgChannel!=AVG.NoAveraging:
#            self.factors   = numpy.average(self.factors[:,self.chansel], axis=1)
#
#        fields = [AX.TYPE, AX.BL, AX.FQ, AX.SB, AX.SRC, AX.P, AX.CH]
#
#        # weight filtering
#        self.nreject   = 0
#        self.reject_f  = lambda weight: False
#        self.threshold = -10000000
#        if selection.weightThreshold is not None:
#            self.threshold = CP(selection.weightThreshold)
#            self.reject_f  = lambda weight: weight<self.threshold
#
#        ## Now we can start the reduction of the table
#        ## INCORPORATE THE WEIGHT COLUMN
#        if selection.weightThreshold is None:
#            columns        = ["ANTENNA1", "ANTENNA2", "UVW", "DATA_DESC_ID", "FIELD_ID", datacol]
#            self.actual_fn = self.withoutWeightThresholding
#        else:
#            columns        = ["ANTENNA1", "ANTENNA2", "UVW", "DATA_DESC_ID", "FIELD_ID", "WEIGHT", datacol]
#            self.actual_fn = self.withWeightThresholding
#        if self.flags:
#            columns.append( "FLAGCOL" )
#        pts =  ms2util.reducems2(self, self.table, {}, columns, verbose=True, slicers=slicers, chunksize=self.chunksize)
#
#        if self.nreject:
#            print "Rejected ",self.nreject," points because of weight criterion"
#
#        rv  = {}
#        for (label, dataset) in iteritems(pts):
#            rv[ self.MKLAB(fields, label) ] = dataset
#        #for k in rv.keys():
#        #    print "Plot:",str(k),"/",map(str, rv[k].keys())
#        return rv
#
#    ## Here we make the plots
#    def __call__(self, *args):
#        return self.actual_fn(*args)
#
#    #### This is the version WITHOUT WEIGHT THRESHOLDING
#    def withoutWeightThresholding(self, acc, a1, a2, uvw, dd, fld, data, *flag):
#        #print "__call__: ",a1,a2,tm,dd,fld,data.shape
#        # Make really sure we have a 3-D array of data ...
#        d3d  = m3d(data)
#        shp  = data.shape
#        flg  = unflagged() if not flag else flag[0]
#        # Good. We have a block of data, shape (nrow, nchan, npol)
#        # Step 1: apply the masking + vector averaging
#        #         'vamd' = vector averaged masked data
#        #         Try to use the pre-computed channel mask, if it fits,
#        #         otherwise create one for this odd-sized block
#        #         (typically the last block)
#        mfn  = self.maskfn if shp[0]==self.chunksize else mk3dmask_fn_mask(shp[0], self.chansel, shp[2])
#        vamd = self.vectorAvg( mfn(d3d) )
#
#        # Now create the quantity data - map the quantity functions over the
#        # (potentially) vector averaged data and (potentially) scalar
#        # average them
#        qd   = map_(lambda (qnm, qfn): (qnm, self.scalarAvg(qfn(vamd))), self.quantities)
#
#        # we can compute the uv distances of all spectral points in units of lambda
#        # because we have the UVW's now and the nu/speed-of-lite for all spectral points
#        uvd  = numpy.atleast_2d( self.factors[dd].T * self.uvdist_f(uvw) )
#
#        # Now we can loop over all the rows in the data
#
#        # We don't have to test *IF* the current data description id is
#        # selected; the fact that we see it here means that it WAS selected!
#        # The only interesting bit is selecting the correct products
#        for row in range(shp[0]):
#            (fq, sb, plist) = self.ddSelection[ dd[row] ]
#            for (chi, chn) in self.chanidx:
#                for (pidx, pname) in plist:
#                    l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, chn]
#                    for (qnm, qval) in qd:
#                        l[0] = qnm
#                        acc.setdefault(tuple(l), dataset()).append(uvd[chi, row], qval[row, chi, pidx], flg[row, chi, pidx])
#        return acc
#
#    #### This is the version WITH WEIGHT THRESHOLDING
#    def withWeightThresholding(self, acc, a1, a2, uvw, dd, fld, weight, data, *flag):
#        #print "__call__: ",a1,a2,tm,dd,fld,data.shape
#        # Make really sure we have a 3-D array of data ...
#        d3d  = m3d(data)
#        shp  = data.shape
#        flg  = unflagged() if not flag else flag[0]
#        # compute weight mask
#        w3d  = numpy.zeros(shp, dtype=numpy.float)
#        for i in range_(shp[0]):
#            # we have weights per polzarization but we must
#            # expand them to per channel ...
#            cw = numpy.vstack( shp[1]*[weight[i]] )
#            w3d[i] = cw
#        w3m =  w3d<self.threshold
#        wfn = lambda a: numpy.ma.MaskedArray(a.data, numpy.logical_and(a.mask, w3m))
#        # Good. We have a block of data, shape (nrow, nchan, npol)
#        # Step 1: apply the masking + vector averaging
#        #         'vamd' = vector averaged masked data
#        #         Try to use the pre-computed channel mask, if it fits,
#        #         otherwise create one for this odd-sized block
#        #         (typically the last block)
#        mfn  = self.maskfn if shp[0]==self.chunksize else mk3dmask_fn_mask(shp[0], self.chansel, shp[2])
#        vamd = self.vectorAvg( wfn(mfn(d3d)) )
#
#        # Now create the quantity data - map the quantity functions over the
#        # (potentially) vector averaged data and (potentially) scalar
#        # average them
#        qd   = map_(lambda (qnm, qfn): (qnm, self.scalarAvg(qfn(vamd))), self.quantities)
#
#        # compute uv distances
#        uvd  = self.uvdist_f(uvw)
#        #for (qn, qv) in qd:
#        #    print qn,": shape=",qv.shape
#
#        # Now we can loop over all the rows in the data
#
#        # We don't have to test *IF* the current data description id is
#        # selected; the fact that we see it here means that it WAS selected!
#        # The only interesting bit is selecting the correct products
#        for row in range(shp[0]):
#            (fq, sb, plist) = self.ddSelection[ dd[row] ]
#            for (chi, chn) in self.chanidx:
#                for (pidx, pname) in plist:
#                    if self.reject_f(w3d[row, chi, pidx]):
#                        self.nreject = self.nreject + 1
#                        continue
#                    l = ["", (a1[row], a2[row]), fq, sb, fld[row], pname, chn]
#                    for (qnm, qval) in qd:
#                        l[0] = qnm
#                        #pi       = self.plot_idx(l)
#                        #di       = self.ds_idx(l)
#                        #print "row #",row,"/l=",l," => pi=",pi," di=",di," qval.shape=",qval.shape
#                        acc.setdefault(tuple(l), dataset()).append(tm[row], qval[row, chi, pidx], flag[row, chi, pidx])
#        return acc



Iterators = {
    'amptime' : data_quantity_time([(YTypes.amplitude, numpy.ma.abs)]),
    'phatime' : data_quantity_time([(YTypes.phase, lambda x: numpy.ma.angle(x, True))]),
    'anptime' : data_quantity_time([(YTypes.amplitude, numpy.ma.abs), (YTypes.phase, lambda x: numpy.ma.angle(x, True))]),
    'retime'  : data_quantity_time([(YTypes.real, numpy.real)]),
    'imtime'  : data_quantity_time([(YTypes.imag, numpy.imag)]),
    'rnitime' : data_quantity_time([(YTypes.real, numpy.real), (YTypes.imag, numpy.imag)]),
    'ampchan' : data_quantity_chan([(YTypes.amplitude, numpy.ma.abs)]),
    'ampfreq' : data_quantity_chan([(YTypes.amplitude, numpy.ma.abs)], byFrequency=True),
    'phachan' : data_quantity_chan([(YTypes.phase, lambda x: numpy.ma.angle(x, True))]),
    'phafreq' : data_quantity_chan([(YTypes.phase, lambda x: numpy.ma.angle(x, True))], byFrequency=True),
    'anpchan' : data_quantity_chan([(YTypes.amplitude, numpy.ma.abs), (YTypes.phase, lambda x: numpy.ma.angle(x, True))]),
    'anpfreq' : data_quantity_chan([(YTypes.amplitude, numpy.ma.abs), (YTypes.phase, lambda x: numpy.ma.angle(x, True))], byFrequency=True),
    'rechan'  : data_quantity_chan([(YTypes.real, numpy.real)]),
    'imchan'  : data_quantity_chan([(YTypes.imag, numpy.imag)]),
    'rnichan' : data_quantity_chan([(YTypes.real, numpy.real), (YTypes.imag, numpy.imag)]),
    'wt'      : weight_time(),
    'uv'      : uv(),
    'ampuv'   : data_quantity_uvdist([(YTypes.amplitude, numpy.ma.abs)]),
    'phauv'   : data_quantity_uvdist([(YTypes.phase, lambda x: numpy.ma.angle(x, True))])
        }
