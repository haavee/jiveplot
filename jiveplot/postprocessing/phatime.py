from __future__ import print_function
# system imports
import math
import operator

# extensions
import numpy

# own stuff
from jiveplot import plots

try:
    # xrange could be A Thing Of The Past
    from past.builtins import xrange
except ImportError:
    # ... but otherwise it's just a builtin
    pass

def do_wrap(seq, low=-numpy.pi, high=numpy.pi):
    out   = []
    off   = 0.0
    delta = high-low
    for v in seq:
        tmp = v - off
        if tmp<low:
            off -= delta
        if tmp>high:
            off += delta
        out.append(v-off)
    return numpy.array(out)

def unwrap(plotar, ms2mappings):
    for k in plotar.keys():
        for d in plotar[k].keys():
            # get a reference to the data set
            dsref = plotar[k][d]
            # get sorted data and unwrap the phases
            (xvals, yvals) = zip(*sorted(zip(dsref.xval, dsref.yval), key=operator.itemgetter(0)))
            yvals      = numpy.unwrap(numpy.deg2rad(yvals))
            dsref.xval = numpy.array(xvals)
            dsref.yval = numpy.rad2deg(yvals)

def drawline_fn(n, xv, yv):
    def f(device, **kwargs):
        xoffset = kwargs.get('xoffset', 0.0)
        #print("    f(): drawing extra Y for {0}".format(n))
        #print("    ",xv, yv)
        device.pgline(xv - xoffset, yv)
    return f

def phaserate(plotar, ms2mappings):
    if plotar.plotType!='phatime':
        raise RuntimeError( "phaserate() cannot run on plot type {0}".format(plotar.plotType) )
    spm = ms2mappings.spectralMap
    # iterate over all plots and all data sets within the plots
    for k in plotar.keys():
        for d in plotar[k].keys():
            # get a reference to the data set
            dsref = plotar[k][d]
            # get the full data set label - we have access to all the data set's properties (FQ, SB, POL etc)
            n     = plots.join_label(k, d)
            # fit a line through the unwrapped phase
            unw    = numpy.unwrap(numpy.deg2rad(dsref.yval))
            coeffs = numpy.polyfit(dsref.xval, unw, 1)
            # evaluate the fitted polynomial at the x-loci
            extray = numpy.polyval(coeffs, dsref.xval)
            # here we could compute the reliability of the fit
            diff   = unw - extray
            ss_tot = numpy.sum(numpy.square(unw - unw.mean()))
            ss_res = numpy.sum(numpy.square(diff))
            r_sq   = 1.0 - ss_res/ss_tot
            # compare std deviation and variance in the residuals after fit
            std_r  = numpy.std(diff)
            var_r  = numpy.var(diff)
            f      = spm.frequencyOfFREQ_SB(n.FQ, n.SB)
            rate   = coeffs[0]
            if var_r<std_r:
                print("{0}: {1:.8f} ps/s @ {2:5.4f}MHz [R2={3:.3f}]".format(n, rate/(2.0*numpy.pi*f*1.0e-12), f/1.0e6, r_sq ))
                # before plotting wrap back to -pi,pi and transform to degrees
                dsref.extra = [ drawline_fn(n, dsref.xval, numpy.rad2deg(do_wrap(extray))) ]

cache = {}
def phasedbg(plotar, ms2mappings):
    global cache
    if plotar.plotType!='phatime':
        raise RuntimeError( "phasedbg() cannot run on plot type {0}".format(plotar.plotType) )
    store = len(cache)==0
    # iterate over all plots and all data sets within the plots
    for k in plotar.keys():
        for d in plotar[k].keys():
            # get a reference to the data set
            dsref = plotar[k][d]
            # get the full data set label - we have access to all the data set's properties (FQ, SB, POL etc)
            n     = plots.join_label(k, d)
            # fit a line through the unwrapped phase
            unw    = numpy.unwrap(numpy.deg2rad(dsref.yval))
            #coeffs = numpy.polyfit(dsref.xval, unw, 1)
            coeffs = numpy.polyfit(xrange(len(dsref.yval)), unw, 1)
            # evaluate the fitted polynomial at the x-loci
            extray = numpy.polyval(coeffs, dsref.xval)
            # here we could compute the reliability of the fit
            diff   = unw - extray
            # compare std deviation and variance in the residuals after fit
            std_r  = numpy.std(diff)
            var_r  = numpy.var(diff)
            coeffs = numpy.rad2deg(coeffs)
            if var_r<std_r:
                # decide what to do
                if store:
                    cache[n] = coeffs
                else:
                    # check if current key exists in cache; if so
                    # do differencing
                    otherCoeffs = cache.get(n, None)
                    if otherCoeffs is None:
                        print("{0}: not found in cache".format(n))
                    else:
                        delta = otherCoeffs - coeffs
                        print("{0.BL} {0.SB} {0.P}: dRate={1:5.4f} dOff={2:4.1f}".format(n, delta[0], delta[1]+360.0 if delta[1]<0.0 else delta[1]))
                # before plotting wrap back to -pi,pi and transform to degrees
                #dsref.extra = [ drawline_fn(n, dsref.xval, numpy.rad2deg(do_wrap(extray))) ]
    if not store:
        cache = {}
