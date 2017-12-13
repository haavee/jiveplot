# the possible plottypes are defined here,
import jenums, ms2util, hvutil, parsers, copy, re, inspect, math, numpy, operator, os, types
from label_v6 import label

AX       = jenums.Axes
YTypes   = jenums.enum("amplitude", "phase", "real", "imag", "weight")
Scaling  = jenums.enum("auto_global", "auto_local")
CKReset  = jenums.enum("newplot")

CP       = copy.deepcopy

# we seem to have had a circular dependency here - plots.py (this module) imports plotiterator
# but plotiterator uses plots.YTypes ... Just 'import plots' wouldn't work unless the import of
# plotiterator was deferred until here
import plotiterator

## Sometimes a plot annotation is None ...
M        = lambda x: '*' if x is None else x   # M is for M(aybe)
M2       = lambda k, v: k+"="+("*" if v is None else str(v))
UNIQ     = lambda x: x if x else ""
BASENAME = os.path.basename


## a real pgplot 'context', usable in 'with ...' constructions
class pgenv(object):
    def __init__(self, plt):
        self.plotter = plt

    def __enter__(self):
        self.plotter.pgsave()

    def __exit__(self, tp, val, tb):
        self.plotter.pgunsa()


## Keep track of the layout of a page in number-of-plots in X,Y direction
class layout(object):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def nplots(self):
        return self.nx * self.ny

    def __str__(self):
        return "{2} plots organized as {0} x {1}".format(self.nx, self.ny, self.nplots())



# Take two labels and join them - i.e. to go from separate plot/data set labels 
# to full data set label
def join_label(l1, l2):
    # create an empty label
    newlab = label({}, [])
    def attrval(a):
        # it's ok if _either_ of l1 or l2 has the attribute but not both
        aval = filter(operator.truth, [getattr(l1, a), getattr(l2, a)])
        if len(aval)>1:
            raise RuntimeError, "Duplicate attribute value {0}: {1} {2}".format(a, aval[0], aval[1])
        return None if len(aval)==0 else aval[0] 
    map(lambda a: setattr(newlab, a, attrval(a)), label._attrs)
    return newlab

# take one label and split it in two; the plot and data set label, based
# on the contents of the 'inplot' list [the rest ends up in the data set label]
def split_label(l, inplot):
    # two brand new empty labels
    mk_lab = lambda : label({}, [])
    def reductor((pl, dsl), attr):
        v = getattr(l, attr)
        setattr(pl, attr, v) if attr in inplot else setattr(dsl, attr, v)
        return (pl, dsl)
    return reduce(reductor, label._attrs, (mk_lab(), mk_lab()))

def label_splitter(inPlot):
    inDataset = set(label._attrs) - set(inPlot)
    def do_split(l):
        return (label(l, inPlot), label(l, inDataset))
    return do_split

## Colorkey functions. Take a data set label and produce a
## key such that for data sets having identical "keys"
## they will be drawn in identical colors.

# the default colour index function: each key gets their 
# own distinctive colour index
def ckey_builtin(label, keycoldict, **opts):
    key = str(label)
    ck  = keycoldict.get(key, None)
    if ck is not None:
        return ck
    if len(keycoldict)>=opts.get('ncol', 16):
        return 1
    # Never seen this label before - must allocate new colour!
    # find values in the keycoldict and choose one that isn't there already
    colours = sorted([v for (k,v) in keycoldict.iteritems()])
    # skip colour 0 and 1 (black & white)
    ck      = 2
    if colours:
        # find first unused colour index
        while ck<=colours[-1]:
            if ck not in colours:
                break
            ck = ck + 1
    keycoldict[key] = ck
    return ck

#### A function that returns functions that remap each subband's x-axis
#### such that they are put next to each other in subband (=meta data) order
#### The original x-axes will be *remapped* for plotting purposes.
#### If you want to plot against e.g. physical x-axis values don't set 'multiSubband' 
#### to true or plot against frequency (note: not implemented yet ...)
####
#### Returns tuple (dict(), new-xaxis-maximum-value)

#### 'datasets' = list of (label, data) pairs
def mk_offset(datasets, attribute):
    # for each subband we must record the (max) extent of the x-axis
    # (there could be data sets for the same subband from different
    #  sources with different number of data points)
    #
    # Then we sort them by subband, and update each subband with the current
    # offset, increment the offset by the current subband's size and proceed
    def range_per_sb(acc, ds):
        (label, data) = ds
        attributeval  = getattr(label, attribute)
        (mi, ma)      = (min(data.xval), max(data.xval))
        (ami, ama)    = acc.get(attributeval, (mi, ma))
        acc.update( {attributeval: (min(mi, ami), max(ma, ama))} )
        return acc

    # process one entry from the {attrval1:(mi, ma), attrval2:(mi, ma), ...} dict
    # the accumulator is (dict(), current_xaxis_length)
    def offset_per_sb(acc, kv):
        (offsetmap, curMax)  = acc
        (sb, (sbMin, sbMax)) = kv
        if curMax is None:
            # this is the first SB in the plot - this sets
            # the xmin/xmax and the x-axis transform is the identity transform
            offsetmap[sb] = lambda xarr: xarr
            # now initialize the actual max x value to current SB's max x
            sbOffset      = 0
        else:
            sbOffset      = (curMax - sbMin)
            # generate a function which transforms the x-axis for the current subband
            offsetmap[sb] = lambda xarr: xarr+sbOffset if isinstance(xarr, numpy.ndarray) \
                                                       else map(lambda z:z+sbOffset, xarr)
        # the new x-axis length is the (transformed) maximum x value of this subband
        return (offsetmap, sbMax+sbOffset)

    # we automatically get the new x-axis max value out of this
    return reduce(offset_per_sb,
                  sorted(reduce(range_per_sb, datasets, {}).iteritems(),
                         key=operator.itemgetter(0)),
                  ({}, None))

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

Drawers = jenums.enum("Lines", "Points", "Both")


##########################################################
####
####      Base class for plotter objects
####
##########################################################
noFilter = lambda x: True


class Plotter(object):
    #drawfuncs = { Drawers.Points: lambda dev, x, y, tp: dev.pgpt(x, y, tp),
    #              Drawers.Lines:  lambda dev, x, y, tp: dev.pgline(x, y) }

    def __init__(self, desc, xaxis, yaxis, lo, xscaling=None, yscaling=None, yheights=None, drawer=None):
        self.xAxis               = CP(xaxis)
        self.yAxis               = CP(yaxis)
        self.yHeights            = CP(yheights)
        self.Description         = CP(desc)
        self.defaultLayout       = CP(lo)
        self.defaultDrawer       = CP(Drawers.Points) if drawer is None else CP(drawer)
        self.defaultxScaling     = CP(xscaling) if xscaling else CP(Scaling.auto_global)
        self.defaultyScaling     = CP(yscaling) if yscaling else [Scaling.auto_global] * len(self.yAxis)
        self.defaultsortOrder    = lambda x: hash(x)
        self.defaultsortOrderStr = "(none)"
        self.defaultMarker       = [None] * len(self.yAxis)
        self.defaultMarkerStr    = ["(none)"] * len(self.yAxis)
        self.defaultCkFun        = ckey_builtin
        self.defaultCkFunS       = "ckey_builtin"
        self.defaultFilter       = [noFilter] * len(self.yAxis)
        self.defaultFilterS      = ["(none)"] * len(self.yAxis)

        # dict mapping a specific drawer to a method call on self
        self.drawfuncs = { Drawers.Points: lambda dev, x, y, tp: self.drawPoints(dev, x, y, tp),
                           Drawers.Lines:  lambda dev, x, y, tp: self.drawLines(dev, x, y) }
        # this dict maps a specific drawer setting to list-of-functioncalls-to-self
        self.drawDict  = { Drawers.Points: [self.drawfuncs[Drawers.Points]],
                           Drawers.Lines:  [self.drawfuncs[Drawers.Lines]],
                           Drawers.Both:   [self.drawfuncs[Drawers.Points], self.drawfuncs[Drawers.Lines]] }
        
        # reset ourselves - go back to default plot settings
        # some (if not all of these) can be modified by the user
        self.reset()

    def description(self):
        return self.Description

    def xaxis(self):
        return self.xAxis

    def yaxis(self):
        return self.yAxis

    def nYaxes(self):
        return len(self.yAxis)

    def reset(self):
        # There's at least three different y-scaling possibilities
        #    * externally given global scale
        #    * automatic global scale derived from the max across all plots
        #    * automatic scaling of each plot
        # currently support two subplots in y axis. both subplots share
        # the x-axis
        self.layOut       = CP(self.defaultLayout)
        self.xScale       = CP(self.defaultxScaling)
        self.yScale       = CP(self.defaultyScaling)
        self.sortOrder    = CP(self.defaultsortOrder)
        self.sortOrderStr = CP(self.defaultsortOrderStr)
        self.marker       = CP(self.defaultMarker)
        self.markerStr    = CP(self.defaultMarkerStr)
        self.ck_fun       = CP(self.defaultCkFun)
        self.ck_fun_s     = CP(self.defaultCkFunS)
        self.filter_fun   = CP(self.defaultFilter)
        self.filter_fun_s = CP(self.defaultFilterS)
        self.multiSubband = False
        self.lineWidth    = 2
        self.pointSize    = 4
        self.markerSize   = 6
        self.drawMethod   = CP([""]*len(self.yAxis))
        self.drawers      = CP([[]]*len(self.yAxis))
        self.setDrawer(*self.defaultDrawer.split())

    # self.drawers    = [ [drawers-for-yaxis0], [drawers-for-yaxis1], ... ]
    # self.drawMethod = "yaxis0:method yaxis1:method ...."
    #
    # allowed:
    #    1 arg => set this drawer for all yAxes
    #    n arg => set the drawer for a specific yAxis
    #             format of argN:
    #              (<yAx>:)method
    #             if "(<yAx>:)" prefix is omitted the method will
    #             be set for yAxis N in stead of yAxis "<yAx>"
    #  n > nr-of-yaxes is not allowed
    #
    # Only allow:
    #    - all arguments fully unqualified "draw lines points"
    #    - all arguments fully qualified   "draw amplitude:points phase:lines"
    # In order to prevent this:
    #      "draw phase:points lines" 
    def setDrawer(self, *args):
        if args:
            # divide the arguments into qualifieds or unqualifieds
            (qualifieds, unqualifieds)  =  \
                hvutil.partition(operator.methodcaller('group', 'yAx'),
                                 filter(operator.truth,
                                        map(re.compile(r"^((?P<yAx>[^:]+):)?(?P<method>lines|points|both)$", re.I).match, args)))
            # Make sure that every entry matched
            if (len(qualifieds)+len(unqualifieds))!=len(args):
                raise RuntimeError, "Invalid draw method(s) specified; use Lines, Points or Both"
            # Depending in which one is the empty list we do things
            # and we complain loudly + bitterly if they're both non-empty ...
            if qualifieds and not unqualifieds:
                for qual in qualifieds:
                    ax                    = qual.group('yAx')
                    if ax not in self.yAxis:
                        raise RuntimeError, "The current plot type has no panel for {0}".format( ax )
                    yIdx                  = self.yAxis.index( ax  )
                    dm                    = qual.group('method').capitalize()
                    self.drawers[yIdx]    = self.drawDict[ dm ]
                    self.drawMethod[yIdx] = dm
            elif unqualifieds and not qualifieds:
                # all unqualified. Only acceptable: 1 unqualified or nYAxis unqualifieds
                if len(unqualifieds)!=len(self.yAxis) and len(unqualifieds)!=1:
                    raise RuntimeError, "Incorrect number of drawing methods supplied for plot type (either 1 or {0})".format(len(self.yAxis))
                # if there's just one, replicate to len yAxis
                methods = unqualifieds if len(unqualifieds)==len(self.yAxis) else [unqualifieds[0]] * len(self.yAxis)
                for (idx, method) in enumerate(methods):
                    dm                   = method.group('method').capitalize()
                    self.drawers[idx]    = self.drawDict[ dm  ]
                    self.drawMethod[idx] = dm
            else:
                raise RuntimeError, "You cannot mix qualified axis drawing methods with unqualified ones" 
        return " ".join(map(":".join, zip(self.yAxis, self.drawMethod)))

    # want to fix the scale of the axes?
    #  can give either:
    #     Scaling.auto_global, Scaling.auto_local or [<min>, <max>]
    # There's no asserts being done - you'll find out at runtime if
    # it's been set to something "the system" doesn't grok ...
    def xscale(self, *args):
        if not args:
            return self.xScale
        self.xScale = args[0]

    # y-scale can be set per subplot
    def yscale(self, idx, *args):
        try:
            if not args:
                return self.yScale[idx]
            self.yScale[idx] = args[0]
        except IndexError:
            raise RuntimeError, "This plot type has no panel {0}".format(idx)

    # query or set the layout
    # the optional argument must be a plots.layout object
    def layout(self, *args):
        if not args:
            return self.layOut
        self.layOut = args[0]

    # set line width, point size or marker size
    def setLineWidth(self, *args):
        if args:
           self.lineWidth = args[0]
        return self.lineWidth

    def setPointSize(self, *args):
        if args:
           self.pointSize = args[0]
        return self.pointSize

    def setMarkerSize(self, *args):
        if args:
           self.markerSize = args[0]
        return self.markerSize

    # some plot types (notably those <quantity> vs frequency/channel)
    # may support plotting all subband *next* to each other in stead
    # of on top of each other. Use "True" or "False"
    def multisb(self, *args):
        if not args:
            return self.multiSubband
        self.multiSubband = args[0]

    # We can influence the sort order of the plots
    def sortby(self, *args):
        if not args:
            if self.sortOrder==self.defaultsortOrder:
                return "(none)"
            else:
                return self.sortOrderStr
        # args is a list-of-axes which we must transform into an attrgetter
        if len(args)==1 and args[0].lower()=="none":
            self.sortOrder    = CP(self.defaultsortOrder)
            self.sortOrderStr = CP(self.defaultsortOrderStr)
        else:
            self.sortOrder    = hvutil.attrgetter( *map(str.upper, args) )
            self.sortOrderStr = CP(" ".join(args))
        return self.sortOrderStr

    def mark(self, idx, *args):
        if not args:
            return self.markerStr[idx]
        if args[0].lower()=="none":
            self.marker[idx]    = None
            self.markerStr[idx] = CP("(none)")
        else:
            self.marker[idx]    = plotiterator.partitioner(args[0])
            self.markerStr[idx] = CP(args[0])

    def filter_f(self, idx, *args):
        if not args:
            return self.filter_fun_s[idx]
        if args[0].lower()=="none":
            self.filter_fun[idx]   = CP(noFilter)
            self.filter_fun_s[idx] = CP("(none)")
        else:
            self.filter_fun[idx]   = parsers.parse_filter_expr( args[0] )
            self.filter_fun_s[idx] = CP(args[0])

    def markedPointsForYAxis(self, idx, ds):
        return self.marker[idx](ds) if self.marker[idx] else []

    # Create a page label Dict with properties
    #   .left   .center and .right for page header display purposes
    def mk_pagelabel(self, plots):
        rv = Dict()
        rv.left = [
                self.description(),
                "unique: "+M(plots.uniq),
                ";".join([  M2("Pol" , plots.polarizations),
                             M2("Nsub", plots.freqsel),
                             #M2("Src" , plots.sources),
                             ""  if plots.weightThres is None else "[threshold weight<{0}]".format(plots.weightThres),
                             M2("Ch",plots.chansel) if self.xAxis!=AX.CH else "",
                             " ".join(map(lambda (i, m): "" if m is None else "[{0}: {1}]".format(self.yAxis[i], self.markerStr[i]), enumerate(self.marker)))])
        ]
        rv.right = [
                    "data: "+plots.msname + " [" + plots.column+"]",
                    plots.userathost + " " + plots.time,
                    "will be set below"
                ]
        rv.center = plots.project
        if plots.comment:
            rv.left.append( plots.comment )
        return rv

    def drawPoints(self, dev, x, y, tp):
        olw = dev.pgqlw()
        dev.pgslw(self.pointSize)
        dev.pgpt(x, y, tp)
        dev.pgslw(olw)

    def drawLines(self, dev, x, y):
        olw = dev.pgqlw()
        dev.pgslw(self.lineWidth)
        dev.pgline(x, y)
        dev.pgslw(olw)

    def doExtraCallbacks(self, dev, dataset, **kwargs):
        if hasattr(dataset, 'extra'):
            for cb in dataset.extra:
                cb(dev, **kwargs)

    #
    # Deal with colouring data sets based 
    # the data set label
    #

    # return a colour index for the label
    def colkey(self, dslab, **opts):
        return self.ck_fun(dslab, self.ck_dict, **opts)

    def coldict(self):
        return self.ck_dict

    # reset colour-key function to default or
    # install a custom one
    def colkey_fn(self, *args):
        if args:
            if args is None or args[0].lower()=="none":
                self.ck_fun   = CP(self.defaultCkFun)
                self.ck_fun_s = CP(self.defaultCkFunS)
            else:
                self.ck_fun   = parsers.parse_ckey_expr( args[0] )
                self.ck_fun_s = CP(args[0])
        return self.ck_fun_s

    # reset the mapping of label => colour
    # TODO: maybe, for consistent colouring across
    #       multiple plots, allow for *NOT* refreshing
    #       the mapping between plots?
    def colkey_reset(self, *args):
        # if 'auto' in args and ... then self.ck_dict = dict()
        self.ck_dict = dict()

    # the number of pages of plots this would produce
    def num_pages(self, plotar):
        return (len(plotar)/self.layout().nplots())+1

AllInOne = type("AllInOne", (), {})()

##########################################################
####
####          Plot two quantities versus time
####
##########################################################
class Quant2TimePlotter(Plotter):
    def __init__(self, ytypes, yscaling=None, yheights=None, **kwargs):
        super(Quant2TimePlotter, self).__init__("+".join(ytypes) + " versus time", jenums.Axes.TIME, ytypes, layout(1,4), yscaling=yscaling, yheights=yheights, **kwargs)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            if len(plotar):
                raise RuntimeError, "first plot ({0}) > #-of-plots ({1})".format(first, len(plotar))
            else:
                raise RuntimeError, "No plots to plot"

        # make sure that the layout is such that we accomodate all plots on one page!
        # for this style we allow the number of plots to grow in each direction
        layout = growlayout(self.layout(), len(plotar), expandy=True) if onePage is AllInOne else self.layout()

        # Verbatim from amptimedrawer.g cf Huib style 
        device.pgscf( 2 )
   
        # get the pagestyle
        n        = min(len(plotar), layout.nplots()) if onePage else len(plotar)
        eachpage = pagestyle(layout, n, expandy=True)

        # Now we know how many plots/page so we can compute how many plots to do
        last     = min(first + n, len(plotar))

        # Huib, modified to have hours of day
        sday        = 60.0*60.0*24.0
        xmin        = reduce(min, map(lambda x: plotar.limits[x].xlim[0], self.yAxis), 9999999999999)
        day0hr      = math.floor(xmin/sday)*sday

        pagelabel = self.mk_pagelabel(plotar)

        device.pgbbuf()
        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            #for (i, plotlabel) in hvutil.enumerateslice(plotar.keys(), first, last):
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                #print "plot {0} '{1}' (first={2})".format(i, plotlabel.key(), first)
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                # <pidx> is the number of the current plot on the current page
                #        (always in the range 0:<plots_per_page - 1>)
                pnum = (i - first)
                pidx = pnum % eachpage.layout.nplots()

                # new page?
                if pidx==0:
                    #print "issueing new page!"
                    # before we go to the next page, write the legend
                    # because the "colcode" mapping gets filled in as we go along,
                    # waiting with printing it makes sure that it contains at least
                    # all the colour-to-dataset mappings present on the current page
                    # (if the next page introduces new entries they'll be added on
                    #  *that* page)
                    printlegend(device, self.coldict(), eachpage)
                    issuenewpage( device )
                    pagelabel.right[2] = pagenostring(i, len(plotar), eachpage)
                    printlabel(device, pagelabel, eachpage.header);

                # get a reference to the plot we should be plotting now!
                pref  = plotar[plotlabel]

                # Set up the plotcoord for this plot, including
                # world coordinate limits
                cvp      = plotcoord(pidx, eachpage)
                drawxlab = lastrow(eachpage, cvp, pnum, last-first) or self.xScale == Scaling.auto_local

                #print "  drawxlab={0} {1},{2}".format(drawxlab, cvp.x, cvp.y)
                ## Loop over the subplots
                for (subplot, ytype) in enumerate(self.yAxis):
                    # filter the data sets with current y-axis type
                    # Keep the indices because we need them twice
                    datasets = filter(lambda x: x.TYPE==ytype and self.filter_fun[subplot](x), pref.keys())

                    # the type may have been removed due to an expression/selection
                    if not datasets:
                        continue
                    # drawing of the y-axis label depends on the current view port or
                    # the setting of the current panel
                    drawylab = cvp.x==0 or Scaling.auto_local in self.yScale

                    # get viewport and edit y coords if we need to draw y-axis labels
                    vp       = getviewport(device, eachpage, cvp, drawxlab and subplot==0, drawylab, subplot, self.yHeights)

                    # Now send to the device
                    device.pgsvp( *vp )
                    # and set character height
                    device.pgsch( 0.75 )

                    # get the limits of the plots in world coordinates
                    (xlims, ylims) = getXYlims(plotar, ytype, plotlabel, self.xScale, self.yScale[subplot])

                    # we subtract day0hr from all x-axis values so we must do that with
                    # the x-axis limits too
                    xlims = map(lambda x: x - day0hr, xlims)
                
                    dy = ylims[1] - ylims[0]

                    device.pgswin( xlims[0], xlims[1], ylims[0], ylims[1] )
                    setboxes(device, cvp, drawxlab and subplot==0, drawylab, self.xAxis==jenums.Axes.TIME)

                    # filter the data sets with type amplitude. 
                    # Keep the indices because we need them twice
                    device.pgsls( 1 )

                    # first use: draw the data sets in the plots
                    olw = device.pgqlw()
                    if olw!=3:
                        device.pgslw(8)
                    for ds in datasets:
                        dsref = pref[ds]
                        # get the colour key for this data set
                        device.pgsci( self.colkey(label(ds, plotar.dslabel), **opts) )
                        for d in self.drawers[subplot]:
                            d(device, dsref.xval - day0hr, dsref.yval, -2 )
                        mp = self.markedPointsForYAxis(subplot, dsref)
                        if mp:
                            lw = device.pgqlw()
                            device.pgslw(self.markerSize)
                            device.pgpt( dsref.xval[mp] - day0hr, dsref.yval[mp], 7)
                            device.pgslw(lw)
                        self.doExtraCallbacks(device, dsref, xoffset=day0hr)
                    device.pgslw(olw)

                    if subplot==0: 
                        # Add some more metadata, if appropriate
                        device.pgsci( 1 )
                        device.pgsch( 0.8 )
                        # 2nd use of the data sets: print the source names at the appropriate times
                        printsrcname(device, pref, datasets, day0hr, ylims[0] + 0.45*dy, 0.1*dy)
                        device.pgmtxt( 'T', -1.0, 0.5, 0.5, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )

            # last page with plots also needs a legend, doesn't it?
            printlegend(device, self.coldict(), eachpage)
        finally:
            device.pgebuf()



##########################################################
####
####      Generic X vs Y plot
####
##########################################################
class GenXvsYPlotter(Plotter):
    def __init__(self, xtype, ytype, yscaling=None, lo=None, colkey=None):
        super(GenXvsYPlotter, self).__init__(ytype+" versus "+xtype, xtype, [ytype], layout(2,4) if lo is None else lo, yscaling=yscaling)
        if colkey is not None:
            self.colkey_fn(colkey)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            raise RuntimeError, "first plot ({0}) > #-of-plots ({1})".format(first, len(plotar))

        # make sure that the layout is such that we accomodate all plots on one page!
        # for this style we allow the number of plots to grow in each direction
        layout = growlayout(self.layout(), len(plotar), expandx=True, expandy=True) if onePage is AllInOne else self.layout()

        # Verbatim from ampchandrawer.g cf Huib style 
        device.pgscf( 2 )
   
        # get the pagestyle
        n        = min(len(plotar), layout.nplots()) if onePage else len(plotar)
        eachpage = pagestyle(layout, n, expandy=True, expandx=True)

        # Now we know how many plots/page so we can compute how many plots to do
        last     = min(first + n, len(plotar))

        pagelabel = self.mk_pagelabel( plotar )

        device.pgbbuf()

        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            # we want the plots in baseline, subband order, if any
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                # <pidx> is the number of the current plot on the current page
                #        (always in the range 0:<plots_per_page - 1>)
                pnum = (i - first)
                pidx = pnum % eachpage.layout.nplots()

                # new page?
                if pidx==0:
                    # before we go to the next page, write the legend
                    # because the "colcode" mapping gets filled in as we go along,
                    # waiting with printing it makes sure that it contains at least
                    # all the colour-to-dataset mappings present on the current page
                    # (if the next page introduces new entries they'll be added on
                    #  *that* page)
                    printlegend(device, self.coldict(), eachpage)
                    issuenewpage( device )
                    pagelabel.right[2] = pagenostring(i, len(plotar), eachpage)
                    printlabel(device, pagelabel, eachpage.header);

                # get a reference to the plot we should be plotting now!
                pref = plotar[plotlabel]

                # filter the data sets with type yType
                # Keep the indices because we need them twice
                datasets = filter(lambda x: x.TYPE==self.yAxis[0] and self.filter_fun[0](x), pref.keys())

                # Set up the plotcoord for this plot, including
                # world coordinate limits
                cvp      = plotcoord(pidx, eachpage)
                drawxlab = lastrow(eachpage, cvp, pnum, last-first) or self.xScale == Scaling.auto_local
                drawylab = cvp.x==0 or self.yScale[0] == Scaling.auto_local

                # get viewport and edit y coords if we need to draw y-axis labels
                vp       = getviewport(device, eachpage, cvp, drawxlab, drawylab)

                # Now send to the device
                device.pgsvp( *vp )
                # and set character height
                device.pgsch( 0.75 )

                # get limits of the plot in world coordinates
                (xlims, ylims) = getXYlims(plotar, self.yAxis[0], plotlabel, self.xScale, self.yScale[0])

                # compute delta y for later use
                dy = ylims[1] - ylims[0]

                device.pgswin( xlims[0], xlims[1], ylims[0], ylims[1] )
                setboxes(device, cvp, drawxlab, drawylab, False)

                device.pgsls( 1 )
                # first use: draw the data sets in the plots
                # remember the lowest subband if the subband number is
                # actually in the data set labels
                for ds in datasets:
                    dsref = pref[ds]
                    # get the colour key for this data set
                    device.pgsci( self.colkey(label(ds, plotar.dslabel), **opts) )
                    # only one yAxis in this type of plot
                    for d in self.drawers[0]:
                        d(device, dsref.xval , dsref.yval, -2 )
                    mp = self.markedPointsForYAxis(0, dsref)
                    if mp:
                        lw = device.pgqlw()
                        device.pgslw(self.markerSize)
                        device.pgpt( dsref.xval[mp], dsref.yval[mp], 7)
                        device.pgslw(lw)
                    self.doExtraCallbacks(device, dsref)
                
                # Add some more metadata
                device.pgsci( 1 )
                device.pgsch( 0.8 )

                # first label with baseline
                device.pgmtxt( 'T', -1.0, 0.5, 0.5, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )

            # last page with plots also needs a legend, doesn't it?
            printlegend(device, self.coldict(), eachpage)
        finally:
            device.pgebuf()


##########################################################
####
####      Plot two quantities versus channel
####
##########################################################
class Quant2ChanPlotter(Plotter):
    def __init__(self, ytypes, yscaling=None, yheights=None, **kwargs):
        super(Quant2ChanPlotter, self).__init__("+".join(ytypes)+" versus channel", jenums.Axes.CH, ytypes, layout(2,4), yscaling=yscaling, yheights=yheights, **kwargs)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            if len(plotar):
                raise RuntimeError, "first plot ({0}) > #-of-plots ({1})".format(first, len(plotar))
            else:
                raise RuntimeError, "No plots to be plotted"

        # make sure that the layout is such that we accomodate all plots on one page!
        # for this style we allow the number of plots to grow in each direction
        layout = growlayout(self.layout(), len(plotar), expandx=True, expandy=True) if onePage is AllInOne else self.layout()

        # Verbatim from ampchandrawer.g cf Huib style 
        device.pgscf( 2 )
   
        # get the pagestyle
        n        = min(len(plotar), layout.nplots()) if onePage not in [None, AllInOne] else len(plotar)
        eachpage = pagestyle(layout, n, expandy=True, expandx=True)

        # Now we know how many plots/page so we can compute how many plots to do
        last     = min(first + n, len(plotar))

        # We need to have the real frequencies
        try:
            mysm = ms2util.makeSpectralMap( plotar.msname )
        except RuntimeError:
            mysm = None

        pagelabel = self.mk_pagelabel( plotar )

        device.pgbbuf()

        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            # we want the plots in baseline, subband order, if any
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                # <pidx> is the number of the current plot on the current page
                #        (always in the range 0:<plots_per_page - 1>)
                pnum = (i - first)
                pidx = pnum % eachpage.layout.nplots()

                # new page?
                if pidx==0:
                    # before we go to the next page, write the legend
                    # because the "colcode" mapping gets filled in as we go along,
                    # waiting with printing it makes sure that it contains at least
                    # all the colour-to-dataset mappings present on the current page
                    # (if the next page introduces new entries they'll be added on
                    #  *that* page)
                    printlegend(device, self.coldict(), eachpage)
                    issuenewpage( device )
                    pagelabel.right[2] = pagenostring(i, len(plotar), eachpage)
                    printlabel(device, pagelabel, eachpage.header);

                # get a reference to the plot we should be plotting now!
                pref = plotar[plotlabel]

                # Set up the plotcoord for this plot, including
                # world coordinate limits
                cvp      = plotcoord(pidx, eachpage)
                drawxlab = lastrow(eachpage, cvp, pnum, last-first) or self.xScale == Scaling.auto_local

                ## Loop over the subplots
                plotxlims = None
                for (subplot, ytype) in enumerate(self.yAxis):
                    # filter the data sets with current y-axis type
                    # Keep the indices because we need them twice
                    datasets = filter(lambda kv: kv[0].TYPE == ytype and self.filter_fun[subplot](kv[0]), pref.iteritems())

                    # the specific type may have been removed?
                    if not datasets:
                        continue

                    # drawing of the y-axis label depends on the current view port or
                    # the setting of the current panel
                    drawylab = cvp.x==0 or Scaling.auto_local in self.yScale

                    # get viewport and edit y coords if we need to draw y-axis labels
                    vp       = getviewport(device, eachpage, cvp, drawxlab and subplot==0, drawylab, subplot, self.yHeights)

                    # Now send to the device
                    device.pgsvp( *vp )
                    # and set character height
                    device.pgsch( 0.75 )

                    # get limits of the plot in world coordinates
                    (xlims, ylims) = getXYlims(plotar, ytype, plotlabel, self.xScale, self.yScale[subplot])

                    ## we support two types of plots, actually:
                    ##   * all subbands on top of each other
                    ##   * all subbands next to each other
                    xoffset  = {}
                    identity = lambda x: x

                    if self.multiSubband:
                        # make new x-axes based on the subband attribute of the label(s)
                        (xoffset, xlims[1]) = mk_offset(datasets, 'SB')

                    # Do x-limits bookkeeping. If this is subplot #0 store the current limits. Otherwise
                    # copy the x-limits to overwrite what we got from this data set
                    if subplot==0:
                        plotxlims = copy.deepcopy(xlims)
                    else:
                        xlims = copy.deepcopy(plotxlims)

                    # compute delta y for later use
                    dy = ylims[1] - ylims[0]

                    device.pgswin( xlims[0], xlims[1], ylims[0], ylims[1] )
                    setboxes(device, cvp, drawxlab and subplot==0, drawylab, self.xAxis==jenums.Axes.TIME, subplot)

                    device.pgsls( 1 )
                    # first use: draw the data sets in the plots
                    # remember the lowest subband if the subband number is
                    # actually in the data set labels
                    for (lab, data) in datasets:
                        # get the colour key for this data set
                        device.pgsci( self.colkey(label(lab, plotar.dslabel), **opts) )
                        # get the actual x-axis values for the data set
                        xvals = xoffset.get(lab.SB, identity)( data.xval )
                        for d in self.drawers[subplot]:
                            d(device, numpy.asarray(xvals), data.yval, -2 )
                        mp = self.markedPointsForYAxis(subplot, data)
                        if mp:
                            lw = device.pgqlw()
                            device.pgslw(self.markerSize)
                            device.pgpt( xvals[mp], data.yval[mp], 7)
                            device.pgslw(lw)
                        self.doExtraCallbacks(device, data)
                    
                    # Add some more metadata in subplot 0
                    if subplot==0:
                        device.pgsci( 1 )
                        device.pgsch( 0.8 )

                        # first label with baseline
                        device.pgmtxt( 'T', -1.0, 0.5, 0.5, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )

                        # Want to write real frequencies so we must have the actual FQ/SB
                        # It either FQ, SB are part of the data set label we're stuffed; the
                        # channel number does not map (uniquely) to frequency any more
                        # because the user has overplotted >1 subband in one plot
                        device.pgsch( 0.5 )
                        if plotlabel.FQ is not None and plotlabel.SB is not None:
                            if mysm is None:
                                frqedge = "no freq info"
                            else:
                                frqedge = "{0:f}MHz".format( mysm.frequencyOfFREQ_SB(plotlabel.FQ, plotlabel.SB)/1.0e6 )
                        else:
                            frqedge = "multi SB"
                        device.pgmtxt( 'B', -1, 0.01, 0.0, frqedge )

            # last page with plots also needs a legend, doesn't it?
            printlegend(device, self.coldict(), eachpage)
        finally:
            device.pgebuf()


#####
##### Utilities
#####

def pagestyle(layout, nplots, expandx=None, expandy=None):
    page        = type('', (), {})()
    page.layout = copy.deepcopy(layout)

    # if the number of plots to plot 
    # is (significantly) smaller than the
    # indicated layout, start making them bigger
    if nplots<page.layout.nplots() and (expandx or expandy):
        if expandx and expandy:
            sqrtN          = math.sqrt(nplots)
            page.layout.nx = int(math.floor(sqrtN))
            page.layout.ny = int(math.ceil(sqrtN))
        elif expandx:
            page.layout.nx = int(math.ceil(float(nplots)/page.layout.ny))
        else:
            page.layout.ny = int(math.ceil(float(nplots)/page.layout.nx))

        while page.layout.nplots()<nplots:
            if expandx and expandy:
                if page.layout.nx<page.layout.ny:
                    page.layout.nx = page.layout.nx + 1
                else:
                    page.layout.ny = page.layout.ny + 1
            elif expandx:
                page.layout.nx = page.layout.nx + 1
            else:
                page.layout.ny = page.layout.ny + 1

    page.header = 0.14
    page.footer = 0.04
    page.xl     = 0.01 #0.05
    page.xr     = 0.98 #0.95
    page.yb     = 0.04 + page.footer
    page.yt     = 1.0 - page.header
    return page

def growlayout(layout, nplots, expandx=None, expandy=None):
    # make sure the new layout is such that all plots will be on one page
    nlayout = copy.deepcopy( layout )
    while nlayout.nplots()<nplots:
        if expandx and expandy:
            if nlayout.nx<nlayout.ny:
                nlayout.nx = nlayout.nx + 1
            else:
                nlayout.ny = nlayout.ny + 1
        elif expandx:
            nlayout.nx = nlayout.nx + 1
        else:
            nlayout.ny = nlayout.ny + 1
    return nlayout

def issuenewpage(device):
   device.pgpage()
   device.pgsvp( 0.0, 1.0, 0.0, 1.0 )
   device.pgswin( 0.0, 1.0, 0.0, 1.0 )
   device.pgsci( 1 )

def pagenostring(i, nplots, eachpage):
    # python math slightly more sensitive than glish
    # "/" is _integer_ divide, not float!!!
    return "page: {0}/{1}".format( \
            int(math.ceil(float(i)/eachpage.layout.nplots())) + 1, \
            int(math.ceil(float(nplots)/eachpage.layout.nplots())) )

# return an unnamed object with attributes ".x" and ".y" indicating
# the column and row index of the indicated plot. Rows are filled first.
def plotcoord(plotnr, eachpage):
    return type('', (), {"x" : plotnr % eachpage.layout.nx,
                         "y" : plotnr / eachpage.layout.nx} )()

# set the actual viewport on the device
# (almost) verbatim from drawutils.g - only we do NOT send the viewport to the device
# 'subplot', if given, must be integer - index of the the n-th subplot
# 'heights' must be given - an array of the individual subplot heights, in fractions of the
#           viewport
def getviewport(device, page, cvp, drawxlab, drawylab, subplot=None, height=None):
    viewport = [0.0] * 4

    dx = (page.xr - page.xl)/page.layout.nx
    dy = (page.yt - page.yb)/page.layout.ny

    viewport[0] = page.xl + cvp.x * dx
    viewport[1] = page.xl + (cvp.x+1) * dx
    viewport[2] = page.yt - (cvp.y+1) * dy
    viewport[3] = page.yt - cvp.y * dy

    dvpx = viewport[1] - viewport[0]
    dvpy = viewport[3] - viewport[2]

    if subplot is not None:
        orgviewport = copy.deepcopy(viewport)

        if subplot>1:
            raise RuntimeError, "viewport only supports two subplots"

        # compute y-coordinate of division - it's at 60% of the plot height
        bot   = sum(height[0:subplot])
        top   = bot + height[subplot]
        vp2   = viewport[2]
        viewport[2] = vp2 + bot*dvpy
        viewport[3] = vp2 + top*dvpy
        #split = viewport[2] + 0.6 * dvpy
        # depending on which subplot this either becomes the new bottom coord
        # or the new top coord
        #if subplot==0:
        #    viewport[3] = split
        #else:
        #    viewport[2] = split

        # relax the fit for more comfort
        viewport[0] = viewport[0] + 0.04*dvpx
        viewport[1] = viewport[1] - 0.04*dvpx

#        if subplot==0:
#            viewport[2] = viewport[2] + 0.04*dvpy
#        else:
#            viewport[3] = viewport[3] - 0.04*dvpy

        # when drawing x-axis labels we only need to shrink
        # the bottom row plots. Plots who have 'drawxlabel' == True
        # and are not on the bottom row of the page have more than enough
        # room below (at least one (unused) plot below them) so we don't have
        # to shrink the plot in the y-direction in that case ...
        if drawxlab and subplot==0 and cvp.y==(page.layout.ny - 1):
	        viewport[2] = viewport[2] + 0.1*dvpy
    else:
        # no subplot, just relax the fit a bit
        viewport[0] = viewport[0] + 0.04*dvpx
        viewport[1] = viewport[1] - 0.04*dvpx
        viewport[2] = viewport[2] + 0.04*dvpy
        viewport[3] = viewport[3] - 0.04*dvpy

        # see above for why for some plots we don't have to shrink
        # the y-direction when plotting the x-axis labels
        if drawxlab and cvp.y==(page.layout.ny - 1):
            viewport[2] = viewport[2] + 0.1*dvpy

    # for all plots, if we must draw the y scale we must make room
    if drawylab:
        viewport[0] = viewport[0] + 0.05 * (viewport[1] - viewport[0])  # was 0.1 * (dx)
    return viewport

##
## Return tuple of (xlims, ylims) with
## xlims = [<xmin>, <xmax>], ylims=[<ymin>, <ymax>]
##
def getXYlims(plotarray, ytype, curplotlabel, xscaling, yscaling):
        ## Process X-axis limits
        fixed = False
        if xscaling == Scaling.auto_local:
            xlims = plotarray.meta[curplotlabel][ytype].xlim
        elif xscaling == Scaling.auto_global:
            xlims = plotarray.limits[ytype].xlim
        else:
            xlims = copy.deepcopy(xscaling)
            fixed = True
        # transform into writable object
        xlims = list(xlims)
        dx = xlims[1] - xlims[0]
        # if x-range too small, safeguard against that?
        # make sure dx is positive and non-zero. Note this
        if dx<1.0e-6:
            dx = 1
        if not fixed:
            xlims[0] = xlims[0] - 0.05*dx
            xlims[1] = xlims[1] + 0.05*dx

        ## Repeat for Y
        fixed = False
        if yscaling == Scaling.auto_local:
            ylims = plotarray.meta[curplotlabel][ytype].ylim
        elif yscaling == Scaling.auto_global:
            ylims = plotarray.limits[ytype].ylim
        else:
            ylims = copy.deepcopy(yscaling)
            fixed = True
        ylims = list(ylims)
        dy = ylims[1] - ylims[0]
        # id. for y range
        if dy<1.0e-6:
            dy = 1
        if not fixed:
            ylims[0] = ylims[0] - 0.05*dy
            ylims[1] = ylims[1] + 0.15*dy
        return (xlims, ylims)

# eachpage: contains the page layout
# plotlocation: row,column index of current plot
# plot:     plotnumber of current plot
# nplot:    total number of plots to plot
#
# we have a last-row-on-page if:
#    * the y-index of the plot IS the bottom row of the page
#    * there are less than one row of plot _remaining_ to plot
def lastrow(eachpage, plotlocation, plot, nplot):
    return (plotlocation.y+1)==eachpage.layout.ny or (nplot - plot) <= eachpage.layout.nx

# Sorry Huib, now in Harro coding style!
def printlabel(device, labels, room):
    # Puts a label on the page, assuming a fixed record structure
    # Save current character height so we can put it back when done
    ch = device.pgqch()

    # set the height we're going to use (somewhat smaller than normal)
    device.pgsch( 0.8 )
  
    # compute the skip between lines - we should use the max of 
    # left/right labels to make them line up properly
    nline    = max(len(labels.left), len(labels.right))
    lineskip = room / (nline+1)

    # compute an array of y-positions up to the maximum number 
    # of lines of text in either left or right
    ypos = map(lambda x: 1.0 - (1.9+x)*lineskip, xrange(0, nline))

    # plot the left lines
    for (txt, pos) in zip(labels.left, ypos):
        device.pgptxt(0.05, pos, 0.0, 0.0, txt)
    # id. for the right ones
    for (txt, pos) in zip(labels.right, ypos):
        device.pgptxt(0.98, pos, 0.0, 1.0, txt)

    # What's left is the center label, we print it LARGER than normal
    device.pgsch( 1.6 );
    device.pgptxt(0.5, 1.0 - 0.3*room, 0.0, 0.5, labels.center);

    # And put back the original character height
    device.pgsch( ch );

def setboxes(device, cvp, drawxlab, drawylab, xtime=False, subplot=None):
    #  Determine the boxstrings; i.e. determine wether or
    #  not to draw the labels. Only draw labels at the leftmost
    #  and the bottom plots
    device.pgsci( 1 )

    xboxstr = yboxstr = "ABCTS"
    yboxstr += "V"

    if xtime:
        xboxstr += "ZHO"

    if (drawxlab and subplot is None) or (drawxlab and subplot==0):
        xboxstr += "N"
        #print "drawing x-label for cvp {0},{1}".format( cvp.x, cvp.y )

    if cvp.x==0 or drawylab:
        yboxstr += "N"
    else:
        yboxstr += "M"

    yboxstr += "1"

    device.pgsch( 0.6 )
    device.pgtbox( xboxstr, 0.0, 0, yboxstr, 0.0, 0 )

## Use the footer of the page to print the color => data set key mapping
def printlegend(device, colcode, eachpage):
    # if the color code dict is empty, there isn't much we can do now, is there?
    if len(colcode)==0:
        return

    device.pgsls( 1 )
    och = device.pgqch()
    device.pgsch( 0.4 )

    # reset all viewports
    device.pgsvp( 0, 1, 0, 1 )
    device.pgswin( 0, 1, 0, 1 )

    (xsep, xoff, xline, xskip) = (0.01, 0.1, 0.03, 0.005)
    ysep                       = 0.45 * eachpage.footer
    nxleg                      = 4
    room                       = eachpage.footer
    # for drawing a horizontal line we need 2 x-coords and only 1 y-coord
    xpos                       = [0.0] * 2
    ypos                       = 0.0

    # Find the longest description and divide the space we have into an equal
    # number of positions. So we need the size of the longest key in device units
    (xsz, ysz) = device.pglen(max(colcode.keys(), key=len), 5)

    nxleg = int( math.floor( (1.0 - 2*xoff) / (xsz+xline+xskip+xsep) ) )
    if nxleg<1:
        nxleg = 1

    for (idx, (label, col)) in enumerate(colcode.iteritems()):
        # sometimes, when only one data set is plotted in a plot,
        # there are no labels left [either they're in the global, shared
        # label section or they're in the per-plot label but none reside
        # in the per-dataset section]. So there's no need to draw the legend
        # because there isn't any
        if not label:
            continue
        (ipos, jpos) = (idx % nxleg, idx / nxleg)
        
        xpos[0] = xoff + (xsz+xline+xskip+xsep) * ipos
        xpos[1] = xpos[0] + xline
        xcapt   = xpos[1] + xskip
        
        ypos  = room + ysep * (1 - jpos)
        ycapt = ypos - 0.005

        device.pgsci( col )
        olw = device.pgqlw()
        device.pgslw( 4 )
        device.pgline( numpy.asarray(xpos), numpy.asarray([ypos, ypos]) )
        device.pgslw( olw )
        device.pgsci( 1 )
        device.pgptxt( xcapt, ycapt, 0.0, 0.0, label )
    device.pgsch( och )

def printsrcname(device, plotref, datasets, day0hr, yval, ystep):
    # device is the PGPLOT object
    # plotarray is reference to the current plot
    # datasets is the (potential) subset of data set keys plotted in this plot
    #          (the plot could contain >1 type of data sets
    # day0hr is the time reference value; all values in the data sets are
    #        plotted wrt this value
    # yval is the y coordinate of the lable
    # ystep determines whether they are staggered
    device.pgsls( 1 )
    och = device.pgqch()
    device.pgsch( 0.4 )
    device.pgsci( 1 )

    seensrces = set()
    for dskey in datasets:
        # If no 'SRC' in data set identifier, do nothing
        if dskey.SRC is None:
            continue
        # Huib: Need to do only for first occurrence
        if dskey.SRC in seensrces:
            continue
        # stagger them in groups of 3, covers 3 source phase ref
        device.pgptxt( plotref[dskey].xval[0]-day0hr, yval + ystep * (len(seensrces)%3), 0.0, 0.0, dskey.SRC )
        seensrces.add( dskey.SRC )
    device.pgsch( och )

###### 
###### The list of defined plotters
######

Plotters  = {
        # by time
        'amptime': Quant2TimePlotter([YTypes.amplitude], yscaling=[Scaling.auto_global], yheights=[0.97]),
        'phatime': Quant2TimePlotter([YTypes.phase], yscaling=[[-185, 185]], yheights=[0.97]),
        'anptime': Quant2TimePlotter([YTypes.amplitude, YTypes.phase], yscaling=[Scaling.auto_global, [-185, 185]], \
                                     yheights=[0.58, 0.38]),
        'retime':  Quant2TimePlotter([YTypes.real], yscaling=[Scaling.auto_global], yheights=[0.97]),
        'imtime':  Quant2TimePlotter([YTypes.imag], yscaling=[Scaling.auto_global], yheights=[0.97]),
        'rnitime': Quant2TimePlotter([YTypes.real, YTypes.imag], yscaling=[Scaling.auto_global, Scaling.auto_global], \
                                     yheights=[0.48, 0.48]),
        'wt':      Quant2TimePlotter([YTypes.weight], yscaling=[Scaling.auto_global], yheights=[0.94]),
        # by channel
        'ampchan': Quant2ChanPlotter([YTypes.amplitude], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines),
        'phachan': Quant2ChanPlotter([YTypes.phase], yscaling=[[-185, 185]], yheights=[0.97], drawer=Drawers.Lines),
        'anpchan': Quant2ChanPlotter([YTypes.amplitude, YTypes.phase], yscaling=[Scaling.auto_global, [-185, 185]], \
                                     yheights=[0.58, 0.38], drawer=Drawers.Lines+" "+Drawers.Points),
        'rechan' : Quant2ChanPlotter([YTypes.real], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines),
        'imchan' : Quant2ChanPlotter([YTypes.imag], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines),
        'rnichan': Quant2ChanPlotter([YTypes.real, YTypes.imag], yscaling=[Scaling.auto_global, Scaling.auto_global],
                                     yheights=[0.48, 0.48], drawer=Drawers.Lines),
        # generic
        'uv'     : GenXvsYPlotter('U', 'V', lo=layout(2,2), colkey='src'),
        'ampuv'  : GenXvsYPlotter('UV distance', YTypes.amplitude, lo=layout(2,2))
}


Types = jenums.enum(*Plotters.keys())
