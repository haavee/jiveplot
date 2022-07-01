# the possible plottypes are defined here,
from   __future__ import print_function
from six          import iteritems
from functional   import zip_, map_, drap, reduce, range_
from plotutil     import *
import enumerations, jenums, ms2util, hvutil, parsers, copy, re
import inspect, math, numpy, operator, os, types, functional, functools

AX       = jenums.Axes
FLAG     = jenums.Flagstuff
Scaling  = enumerations.Enum("auto_global", "auto_local")
FixFlex  = enumerations.Enum("fixed", "flexible")
RowsCols = enumerations.Enum("rows", "columns")
CKReset  = enumerations.Enum("newplot")
FU       = functional
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
        self.nx   = nx
        self.ny   = ny
        self.rows = True

    def nplots(self):
        return self.nx * self.ny

    def __str__(self):
        return "{2} plots organized as {0} x {1}".format(self.nx, self.ny, self.nplots())


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
    colours = sorted([v for (k,v) in iteritems(keycoldict)])
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
        (mi, ma) = (0, 0) if data.xlims is None else data.xlims
        #(mi, ma) = (min(data.xval), max(data.xval))
        # if either mi/ma is None that means there is no data
        (ami, ama)    = acc.get(attributeval, (mi, ma))
        acc.update( {attributeval: (min(mi, ami), max(ma, ama))} )
        return acc

    # process one entry from the {attrval1:(mi, ma), attrval2:(mi, ma), ...} dict
    # the accumulator is (dict(), current_xaxis_length)
    def offset_per_sb(acc, kv):
        (sb, (sbMin, sbMax)) = kv
        # if the indicated subband has no min/max then no data was to be plotted for that one
        if sbMin is None or sbMax is None:
            return acc
        (offsetmap, curMax)  = acc
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
                                                       else map_(lambda z:z+sbOffset, xarr)
        # the new x-axis length is the (transformed) maximum x value of this subband
        return (offsetmap, sbMax+sbOffset)

    # we automatically get the new x-axis max value out of this
    return reduce(offset_per_sb,
                  sorted(iteritems(reduce(range_per_sb, datasets, {})),
                         key=operator.itemgetter(0)),
                  ({}, None))


def normalize_mask(m):
    if numpy.all(m):
        return True
    if numpy.any(m): # note: do not use "sum(m)" - dat's wikkit sl0w
        return m
    # not all && not any => none at all!
    return False


# After https://stackoverflow.com/questions/11731136/python-class-method-decorator-with-self-arguments
def check_attribute(attribute):
    def _check_attribute(f, *brgs, **kwbrgs):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if getattr(self, attribute) is None:
                raise RuntimeError("Called method {0} requires {1} to be not None".format(f.__name__, attribute))
            return f(self, *args, **kwargs)
        return wrapper
    return _check_attribute

def verify_argument(verifier):
    def _verify_argument(f, *brgs, **kwbrgs):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if not verifier(*args, **kwargs):
                raise RuntimeError("Called method {0} fails to verify called with {1} {2}".format(f.__name__, args, kwargs))
            return f(self, *args, **kwargs)
        return wrapper
    return _verify_argument

# A viewport describes an area on the draw surface with xLeft, xRight, yBottom,
# yTop coordinates in Normalized Device Coordinates

# If the viewport is a subplot /then/ it can be drawn on a device, decorated
# with axes, tick marks and labels The subplot state is managed from the parent
# page; if a viewport is requested from a page, the page knows wether the
# viewport is special one (leftmost, rightmost, bottom row) and sets the
# properties accordingly. This allows the viewport to correctly draw x,y axis
# values and labels, if set.

# The parent page does the layouting, taking care of reserving space for the
# values, labels and computing the offsets at which these should happen

# The viewport infers labels and scaling properties directly from the plotter
# object (through its parent page) whose data it is meant to display

class Viewport(object):

    # x=column, y=row, parent = parent _Page_
    def __init__(self, x, y, parent, subplot=None):
        self.viewport_= [0.0]*4  # xLeft xRight yBottom yTop
        self.window   = None
        self.x        = x
        self.y        = y
        self.page     = parent
        self.xDisp    = 0 # "DISP" parameter to PGMTXT for x-label
        self.yDisp    = 0 #  id. for y-label
        self.lastRow  = False
        self.lastCol  = False
        self.subPlot  = subplot

    @staticmethod
    def _is_ok(x):
        return 0<=x<=1
    @staticmethod
    def _is_not_ok(x):
        return x<0 or x>1

    def get_viewport(self):
        return self.viewport_
    def set_viewport(self, vp):
        if len(vp)!=4:
            raise RuntimeError("Viewport must be length 4")
        if functional.filter_(Viewport._is_not_ok, vp):
            raise RuntimeError("Attempt to set invalid viewport {0}".format(vp))
        self.viewport_ = vp
    viewport = property(get_viewport, set_viewport)

    # transform array-of-length-4 into four named properties
    def get0(self):
        return self.viewport_[0]
    def set0(self, v):
        self[0] = v
    def get1(self):
        return self.viewport_[1]
    def set1(self, v):
        self[1] = v
    def get2(self):
        return self.viewport_[2]
    def set2(self, v):
        self[2] = v
    def get3(self):
        return self.viewport_[3]
    def set3(self, v):
        self[3] = v

    xLeft   = property(get0, set0)
    xRight  = property(get1, set1)
    yBottom = property(get2, set2)
    yTop    = property(get3, set3)

    # also allow [0..3] based indexing
    def __getitem__(self, n):
        return self.viewport_[n]
    def __setitem__(self, n, v):
        if Viewport._is_not_ok(v):
            raise RuntimeError("Attempt to set invalid viewport[{0}] = {1} on ".format(n, v)+repr(self))
        self.viewport_[n] = v
        return self

    # return x/y size in NDC
    @property
    @check_attribute('subPlot')
    def dx(self):
        return self.xRight - self.xLeft
    @property
    @check_attribute('subPlot')
    def dy(self):
        return self.yTop - self.yBottom

    # return x/y-sizes in world coordinates
    @property
    @check_attribute('window')
    def dx_world(self):
        return self.window[1] - self.window[0]

    @property
    @check_attribute('window')
    def dy_world(self):
        return self.window[3] - self.window[2]


    # extract a new viewport which is an indexed subsection of the parent viewport
    def subplot(self, idx):
        # get a new viewport as height fraction of this one
        yHeights = self.page.plotter.yHeights
        bot   = sum(yHeights[0:idx])
        top   = bot + yHeights[idx]
        dvpY  = self.yTop - self.yBottom

        # construct with valid subplot and transfer logic values
        rv          = Viewport(self.x, self.y, self.page, subplot=idx)
        rv.xDisp    = self.xDisp
        rv.yDisp    = self.yDisp
        rv.lastRow  = self.lastRow
        rv.lastCol  = self.lastCol
        rv.viewport = [self.xLeft, self.xRight, self.yBottom + bot*dvpY, self.yBottom+top*dvpY]
        return rv

    @check_attribute('subPlot')
    def doXLabel(self):
        return self.page.plotter.xLabel and self.lastRow and self.subPlot==0
    @check_attribute('subPlot')
    def doYLabel(self):
        return self.page.plotter.yLabel[self.subPlot] and self.x == 0
    @check_attribute('subPlot')
    def doXTicks(self):
        return self.subPlot==0 and (self.lastRow or self.page.plotter.xScale == Scaling.auto_local)
    @check_attribute('subPlot')
    def doYTicks(self):
        return self.x == 0 or self.lastCol or self.page.plotter.yScale[self.subPlot] == Scaling.auto_local

    # Do our thang on the device!
    @check_attribute('subPlot')
    def svp(self, device):
        device.pgsvp( *self.viewport_ )

    @check_attribute('subPlot')
    def drawLabels(self, device):
        with pgenv(device):
            # all text drawn in black with page's label character size
            device.pgsci( 1 )
            device.pgsch( self.page.labelCharSz )
            if self.doXLabel():
                device.pgmtxt('B', self.xDisp, 0.9, 1.0, self.page.plotter.xLabel)
            if self.doYLabel():
                device.pgmtxt('L', self.yDisp, 0.9, 1.0, self.page.plotter.yLabel[self.subPlot])

    # sets window and draws boxes &cet
    @check_attribute('subPlot')
    def drawBox(self, device, xmi, xma, ymi, yma):
        if self.window is not None:
            raise RuntimeError("Can only set window once on viewport ({0.x}, {0.y}, subPlot={0.subPlot})".format(self))
        self.window = [xmi, xma, ymi, yma]
        with pgenv(device):
            # step one: set the window in world coordinates
            device.pgswin( *self.window )
            # Need to draw the box always, wih thin line
            device.pgsci( 1 )
            device.pgslw( 1 )
            device.pgsch( self.page.tickCharSz )

            # Form the XOPT/YOPT strings for PGPLOT's PG(T)BOX function
            xboxstr  = yboxstr = "ABCTS"
            yboxstr += "V"

            if self.page.plotter.xAxis == jenums.Axes.TIME:
                xboxstr += "ZHO"

            # do we need to draw the x-axis ticks?
            if self.doXTicks():
                xboxstr += "N"
            # y-axis values for the outermost column get drawn on the right-hand side
            if self.doYTicks():
                yboxstr += ("M" if self.lastCol else "N")
            # y values forced to decimal
            yboxstr += "1"

            #device.pgsch( 0.6 )
            device.pgtbox( xboxstr, 0.0, 0, yboxstr, 0.0, 0 )


    # sets viewport, window, draws boxes and labels
    @check_attribute('subPlot')
    def setLabelledBox(self, device, xmi, xma, ymi, yma):
        self.svp(device)
        self.drawBox(device, xmi, xma, ymi, yma)
        self.drawLabels(device)

    # Help in choosing which x limit to use; data sets these days have
    # flagged+unflagged data and we need to select based on which of these are
    # actually available what the lowest time is.
    # Key into the choice table is:
    #    ( <bool A>, <bool B> )
    # where:
    #      <bool A> = '.xval   is not None'  ("data set has unflagged data points")
    #      <bool B> = '.xval_f is not None'  ("data set has flagged   data points")
    choicetab  = {
            # we don't have to cover (True, True) because
            # that would mean that both .xval and .xval_f are None
            # and that case is filtered out; datasets with no data at all
            # are not sent to the drawing function ...
            (False, False): lambda x, xf: min(min(x), min(xf)),
            (True , False): lambda x, xf: min(xf),
            (False, True ): lambda x, xf: min(x)
            }

    # analyze dataset labels for unique source names
    # and display the first occurrence of each, cycling
    # in three heights. Use world coordinates for this
    @check_attribute('subPlot')
    def printSrcNameByTime(self, device, datasets, xform_x=lambda x: x):
        # if parent page's parent plot [that's ancestry for you ...] indicates
        # don't show source name(s) then let's not do that
        if not self.page.plotter.showSource:
            return
        # do each source only once
        seensrces        = set()
        # must be done in world coordinates
        y_offset, y_step = (0.45 * self.dy_world, 0.1*self.dy_world)
        with pgenv(device):
            # in black w/ small characters
            device.pgsci( 1 )
            device.pgsch( 0.4 )
            for (dskey,dsdata) in datasets:
                # If no 'SRC' in data set identifier or already seen: do nothing
                if dskey.SRC is None or dskey.SRC in seensrces:
                    continue
                # SRC available, not seen yet, find 'earliest time'
                x0   = Viewport.choicetab[(dsdata.xval is None, dsdata.xval_f is None)](dsdata.xval, dsdata.xval_f)
                # stagger them in groups of 3, covers 3 source phase ref
                device.pgptxt( xform_x(x0), y_offset + y_step * (len(seensrces)%3), 0.0, 0.0, dskey.SRC )
                seensrces.add( dskey.SRC )

    @check_attribute('subPlot')
    def drawMainLabel(self, device, mainlabel):
        with pgenv(device):
            device.pgsci( 1 )
            device.pgsch( 0.8 )
            device.pgmtxt( 'T', -1.0, 0.5, 0.5, mainlabel)

    #  Get a nice string representation of this objects's stat
    _fmt = ("Viewport(x={0.x:d},y={0.y:d} [x:{0.xLeft:.3f}-{0.xRight:.3f} y:{0.yBottom:.3f}-{0.yTop:.3f}] xDisp={0.xDisp:.3f} yDisp={0.yDisp:.3f} "+
            "lastRow/Col={0.lastRow}/{0.lastCol} subPlot={0.subPlot}) id={1}").format
    def __repr__(self):
        return Viewport._fmt(self, id(self))




########################################################################################
#
#  An object describing a page layout for a particular plotter and array of plots
#  to be displayed.
#
#  From the parent plotter properties the layout is copied and potentially
#  altered, based on the amount of plots that are to be generated. From the
#  labels and scaling options the Page class can compute the size of the plots
#  and decide how the viewports (where the actual data will be plotted)'s axes
#  are to be created (if x/y tick values must be drawn, if x/y labels must be
#  drawn).
#
#  From the parent plotter's properties it can also infer wether the metadata
#  (header, legend) must be drawn or wether that space needs to be allocated to
#  the plot area.
#
########################################################################################


class Page(object):
    charSize   = 1.0/40    # PGPLOT nominal char size in NDC [1/40th of y-height if charsize set to 1.0]
    tickString = "1.00e^8" # nominal-ish format for how tick values are displayed (it's the string length that counts)


    # come up with a good layout
    #
    #   onePage:
    #       is AllInOne => for animation, change layout to hold nplot
    #       is None     => unnavigable device so we may have to produce > 1 pages
    #       *           => navigable device (window) so draw one page
    #                      if nlot < layout, change layout to hold all
    def __init__(self, plotter, device, onePage, plotar, **kwargs):
        nplot        = len(plotar)
        # start by setting page boundaries
        self.header  = 0.14 if plotter.showHeader else 0.01
        self.footer  = 0.04 if plotter.showLegend else 0.00
        self.xl_     = 0.01
        self.xr_     = 0.98
        self.yb_     = self.footer
        self.yt_     = 1.0 - self.header
        self.plotter = plotter
        # the left hand side and bottom of the plot area
        # may be offset (to make room for x/y axis labels
        self.leftShift_   = 0
        self.rightShift_  = 0
        self.bottomShift_ = 0
        # and the given layout
        self.layout = copy.deepcopy(plotter.layout())
        # depending on if it's allowed and how to re-arrange the amount of plots come up with a new layout
        if onePage is AllInOne:
            # we must grow or shrink the layout such that everything fits on one page, note: this is non-negotiable
            if plotter.fixedLayout:
                print("Warning: fixed layout overridden by AllInOne requirement")
            if nplot > self.layout.nplots():
                self._growLayout(nplot, **kwargs)
            else:
                self._shrinkLayout(nplot, **kwargs)
        elif not plotter.fixedLayout:
            nplot = min(nplot, self.layout.nplots()) if onePage else nplot
            # shrinkage is allowed if number of plots < layout
            # AND (either nx,ny==1 OR spillage > 25%)
            # otherwise we leave the layout well alone I guess
            if nplot<self.layout.nplots() and (self.layout.nx==1 or self.layout.ny==1 or (float(nplot)/self.layout.nplots())<=0.75):
                self._shrinkLayout(nplot, **kwargs)
            else:
                self._updateLayout(nplot, **kwargs)
        # Now we can seed dx,dy for the plot panels
        # and ddx, ddy for room for the tick values, if necessary
        self.dx, self.dy   = (0, 0)
        self.ddx, self.ddy = (0, 0)
        self._updateDxy()

        # page has tickCharSz and labelCharSz (in normalized character heights)
        # as well as the 'offset outside viewport in character heights' in case a label needs to be drawn
        # if we need to do x/y labels we just shift the xleft and ybottom accordingly
        # zip the panel heights with their labels and filter those who are not empty
        # set the property on the page in units of the normal character height
        if self.plotter.charSize > 0:
            self.labelCharSz = self.plotter.charSize
        else:
            self.labelCharSz = [h/n for (h,n) in
                    zip_(map(lambda fraction: self.dy * fraction, self.plotter.yHeights),
                         map(len, self.plotter.yLabel))+
                    [(self.dx, len(self.plotter.xLabel))] if n>0]
            self.labelCharSz = min(0.8*Page.charSize, 0 if not self.labelCharSz else min(self.labelCharSz)) / Page.charSize

        # already compute the tickCharSz with the same current settings for consistency
        # Allow user to override the character size?
        if self.plotter.charSize > 0:
            self.tickCharSz = self.plotter.charSize
        else:
            self.tickCharSz = map_(lambda fraction: (self.dy * fraction)/len(Page.tickString),
                                   self.plotter.yHeights) + [ self.dx/len(Page.tickString) ]
            self.tickCharSz = min(0.6*Page.charSize, min(self.tickCharSz)) / Page.charSize

        # the important values to keep are: tickwitdh (measured string length) and label x/y height and tick char height
        with pgenv(device):
            device.pgsch( self.labelCharSz )
            (self.lXCH, self.lYCH) = device.pgqcs( 0 )
            device.pgsch( self.tickCharSz )
            (self.tXCH, self.tYCH) = device.pgqcs( 0 )
            (self.tickWidth, _   ) = map_(lambda l: 1.2*l, device.pglen(Page.tickString, 0))
            self.tickHeight        = 2 * self.tYCH

        if self.labelCharSz>0:
            # if we need to do x labels, shift bottom of page up by ~2 chY units
            # (xlabels are drawn with horizontal baseline)
            if self.plotter.xLabel:
                self.bottomShift = self.bottomShift + 1.5*self.lYCH
            # if there is/are y labels, shift the left side of the page to the right
            # make room for one and a bit label character-heights-with-vertical-baseline
            if any(self.plotter.yLabel):
                self.rightShift = self.rightShift + 1.5*self.lXCH

        # Depending on x/y scaling we have to make room for x/y axis tick values on left row/bottom row
        # or on all plots
        if self.plotter.xScale == Scaling.auto_local:
            # local x-axis scaling: bottom of each plot must be raised
            self.ddy = self.tickHeight
        else:
            # common x-axis so we can just raise the bottom to make room
            self.bottomShift = self.bottomShift + self.tickHeight

        # repeat for y-axis
        # tick values are drawn with horizontal baseline
        if Scaling.auto_local in self.plotter.yScale:
            # each plot will have own y axis so shift viewport to the right
            self.ddx = self.tickWidth
        else:
            # common/global y-axis, just shift the left hand side of the page's view area
            self.rightShift = self.rightShift + self.tickWidth
            # also make a /little/ more room on the right hand side by shifting
            # the left side back in case we have > 1 columns of plots because
            # the right-most plots will get their global y-axis tick values
            # displayed on the right. (When doing local y-axis scale, all plots
            # get their y-axis values on the left hand side)
            if self.layout.nx > 1:
                self.leftShift  = self.leftShift  + 0.5*self.tickWidth

        # From the parent plotter + plot array we can form the page header, but only if needed
        self.pageLabel = None if not self.plotter.showHeader else self.plotter.mk_pagelabel( plotar )

    def _updateDxy(self):
        self.dx = (self.xr_ - self.xl_ - self.rightShift_ - self.leftShift_)/self.layout.nx
        self.dy = (self.yt_ - self.yb_ - self.bottomShift_)/self.layout.ny

    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setLeftShift(self, ls):
        self.leftShift_ = ls
        self._updateDxy()
    def _getLeftShift(self):
        return self.leftShift_
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setRightShift(self, rs):
        self.rightShift_ = rs
        self._updateDxy()
    def _getRightShift(self):
        return self.rightShift_
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setBottomShift(self, bs):
        self.bottomShift_ = bs
        self._updateDxy()
    def _getBottomShift(self):
        return self.bottomShift_
    leftShift   = property(_getLeftShift,   _setLeftShift)
    rightShift  = property(_getRightShift,  _setRightShift)
    bottomShift = property(_getBottomShift, _setBottomShift)

    # whenever xl, xr, yt, yb is updated, recompute dx, dy
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setxl(self, xl):
        self.xl_ = xl
        self._updateDxy()
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setxr(self, xr):
        self.xr_ = xr
        self._updateDxy()
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setyb(self, yb):
        self.yb_ = yb
        self._updateDxy()
    @verify_argument(lambda *args, **kwargs: args[0]>=0 and args[0]<=1)
    def _setyt(self, yt):
        self.yt_ = yt
        self._updateDxy()
    def _getxl(self):
        return self.xl_
    def _getxr(self):
        return self.xr_
    def _getyb(self):
        return self.yb_
    def _getyt(self):
        return self.yt_

    xl = property(_getxl, _setxl)
    xr = property(_getxr, _setxr)
    yb = property(_getyb, _setyb)
    yt = property(_getyt, _setyt)

    def plotIndex(self, pnum):
        return pnum % self.layout.nplots()

    def viewport(self, pnum, nplot):
        # figure out the index of the plot on the page
        pidx = self.plotIndex(pnum)
        # the viewport coords - take care of filling rows or columns first
        if self.layout.rows:
            rv          = Viewport(pidx % self.layout.nx, pidx // self.layout.nx, self)
        else:
            rv          = Viewport(pidx // self.layout.ny, pidx % self.layout.ny, self)
        rv.lastRow  = (rv.y+1)==self.layout.ny or (nplot - pnum)<=self.layout.nx
        rv.lastCol  = (self.layout.nx>1 and (rv.x+1)==self.layout.nx and not Scaling.auto_local in self.plotter.yScale) or (pnum == nplot)
        xl,yt       = (self.xl + self.rightShift + rv.x*self.dx, self.yt - rv.y * self.dy)
        # if there's multiple columns, rows, relax the fit in those directions
        x_scale     = 0.98 if self.layout.nx > 1 else 1.0
        y_scale     = 0.98 if self.layout.ny > 1 else 1.0
        #              left           right               bottom                       top
        #              0              1                   2                            3
        rv.viewport = [xl + self.ddx, xl+x_scale*self.dx, yt-y_scale*self.dy+self.ddy, yt]
        # we now have the basic viewport for the current plot

        # set label offsts in units of the label character height
        rv.xDisp   = 2
        rv.yDisp   = (1.5*self.tickWidth  / self.lYCH)  if self.lYCH else 0
        return rv

    # if amount of plots to draw << current layout, then we automatically
    # rescale to a layout which maximizes to the actual amount of plots.
    # that is: if rescaling is allowed at all
    def _updateLayout(self, nplots, expandx=None, expandy=None, **kwargs):
        if self.layout.nplots()<nplots or not (expandx or expandy):
            return
        # start with an approximation
        if expandx and expandy:
            sqrtN          = math.sqrt(nplots)
            self.layout.nx = int(math.floor(sqrtN))
            self.layout.ny = int(math.ceil(sqrtN))
        elif expandx:
            self.layout.nx = int(math.ceil(float(nplots)/self.layout.ny))
        else:
            self.layout.ny = int(math.ceil(float(nplots)/self.layout.nx))
        # expand in allowed dimensions until fit
        self._growLayout(nplots, expandx=expandx, expandy=expandy)


    def _growLayout(self, nplots, expandx=None, expandy=None, **kwargs):

        if nplots>self.layout.nplots() and not (expandx or expandy):
            raise RuntimeError("Request to grow layout from {0} to {1} plots but not allowed to expand!".format(self.layout.nplots(), nplots))
        # make sure the new layout is such that all plots will be on one page
        while self.layout.nplots()<nplots:
            if expandx and expandy:
                if self.layout.nx<self.layout.ny:
                    self.layout.nx = self.layout.nx + 1
                else:
                    self.layout.ny = self.layout.ny + 1
            elif expandx:
                self.layout.nx = self.layout.nx + 1
            else:
                self.layout.ny = self.layout.ny + 1


    def _shrinkLayout(self, nplots, expandx=None, expandy=None, **kwargs):
        # if people say expandy = True that means they value the y direction
        # so for shrinking we're going to reverse that
        expandx = not expandx
        expandy = not expandy
        lo      = self.layout

        while True:
            # we want to minimize spillage
            # if < 0 we must grow again
            spill         = lo.nplots() - nplots
            # no spillage means we're /definitely/ done!
            if spill==0:
                break
            (comp, delta) = (operator.gt, +1) if spill < 0 else (operator.lt, -1)

            # if neither nor both directions are allowed to expand ...
            # note: there is no 'xor' for 'objects' in Python only bitwise xor ...
            if (expandx and expandy) or (not expandx and not expandy):
                if lo.nx>1 and lo.ny>1:
                    # if both > 1 prefer a balanced shrinkage
                    if comp(lo.ny,lo.nx):
                        lo.nx += delta
                    else:
                        lo.ny += delta
                elif comp(1, lo.nx):
                    lo.nx += delta
                elif comp(1, lo.ny):
                    lo.ny += delta
                else:
                    raise RuntimeError("Failed to shrink (expandx and expandy and neither nx,ny>1)")
            elif expandx:
                # prefer lowering x, until we can't anymore
                if comp(1, lo.nx):
                    lo.nx += delta
                else:
                    lo.ny += delta
            else:
                # prefer lowering y, until we can't anymore
                if comp(1, lo.ny):
                    lo.ny += delta
                else:
                    lo.nx += delta
            # check new spillage?
            nspill = lo.nplots() - nplots
            if nspill < spill:
                # less spillage is gooder
                continue
            # we may only terminate if spill >= 0
            if spill>=0:
                break


    def show(self, device):
        with pgenv(device):
            device.pgslw( 2 )
            device.pgsls( 1 )
            device.pgsvp( 0, 1, 0, 1)
            device.pgsci( 2 )
            device.pgbox( "BC", 0, 0, "BC", 0, 0)
            device.pgsvp( self.xl_, self.xr_, self.yb_, self.yt_ )
            device.pgsci( 1 )
            device.pgbox( "BC", 0, 0, "BC", 0, 0)

    def printlegend(self, device):
        # if the color code dict is empty, there isn't much we can do now, is there?
        # note: we filter out the ones that do not have a label:
        # sometimes, when only one data set is plotted in a plot,
        # there are no labels left [either they're in the global, shared
        # label section or they're in the per-plot label but none reside
        # in the per-dataset section]. So there's no need to draw the legend
        # because there isn't any
        coldict = dict(filter(functional.compose(operator.truth, operator.itemgetter(0)),
                       iteritems(self.plotter.coldict())))
        if not coldict or not self.plotter.showLegend:
            return
        with pgenv(device):
            device.pgsvp( self.xl_, self.xr_, 0, self.yb_ )
            device.pgswin( 0, 1, 0, 1 )
            device.pgsch(0.4)
            # Find the longest description and divide the space we have into an equal
            # number of positions. So we need the size of the longest key in device units
            (xsz, ysz)                 = device.pglen(max(coldict.keys(), key=len), 0)
            # the values below look like NDC but they are world coordinates for PGPTXT.
            # However, we'll map the footer area to world (0,1,0,1) so it's "NDC" inside the footer ;-)
            (xsep, xoff, xline, xskip) = (0.01, 0.01, 0.03, 0.005)
            (txt_off,)                 = (0.10,) # drop text baseline by this factor to align nicely with the line
            nxleg                      = max(1, int(math.floor((1.0 - 2*xoff) / (xsz+xline+xskip+xsep))))
            # for drawing a horizontal line we need 2 x-coords and only 1 y-coord
            xpos                       = [0.0] * 2
            ypos                       = 0.0
            nyleg                      = int(math.ceil(float(len(coldict))/nxleg))
            dy                         = 1.0/(nyleg+1)
            for (idx, cmap) in enumerate(iteritems(coldict)):
                (label, col) = cmap
                (ipos, jpos) = (idx % nxleg, idx // nxleg)

                xpos[0] = xoff + (xsz+xline+xskip+xsep) * ipos
                xpos[1] = xpos[0] + xline
                xcapt   = xpos[1] + xskip

                ypos  = 1.0 - (jpos + 1)*dy
                ycapt = ypos - txt_off*dy

                device.pgsci( col )
                device.pgslw( 4 )
                device.pgline( numpy.asarray(xpos), numpy.asarray([ypos, ypos]) )
                device.pgsci( 1 )
                device.pgslw( 1 )
                device.pgptxt( xcapt, ycapt, 0.0, 0.0, label )

    def printPageLabel(self, device, curPlot, nPlot):
        if self.pageLabel is None:
            return
        # Puts a label on the page, assuming a fixed record structure.
        # take the data from the parent plotter

        # compute the skip between lines - we should use the max of
        # left/right labels to make them line up properly
        nline    = max(len(self.pageLabel.left), len(self.pageLabel.right))
        # set current pageno based on settings
        self.pageLabel.right[2] = "page: {0}/{1}".format( int(math.ceil(float(curPlot)/self.layout.nplots())) + 1,
                                                          int(math.ceil(float(nPlot)/self.layout.nplots())) )
        # When using giza as PGPLOT backend, then '_' is suddenly "subscript" a-la TeX ffs
        noUnderscore = functools.partial(re.sub, r"_", r"\\_") if 'giza' in device.pgqinf('VERSION').lower() else functional.identity
        with pgenv(device):
            # set the viewport to the header area and map to world coordinates 0,1 0,1
            device.pgsvp( self.xl, self.xr, self.yt, 1 )
            device.pgswin( 0, 1, 0, 1 )

            device.pgsch( 0.8 )

            # compute y position of succesive lines
            ypos      = map_(lambda lineno: 1.0 - (lineno+1.5)/(nline+1), range_(nline))

            # plot the left lines
            for (txt, pos) in zip(self.pageLabel.left, ypos):
                device.pgptxt(0, pos, 0.0, 0.0, noUnderscore(txt))
            # id. for the right ones
            for (txt, pos) in zip(self.pageLabel.right, ypos):
                device.pgptxt(1, pos, 0.0, 1.0, noUnderscore(txt))

            # What's left is the center label, we print it LARGER than normal
            device.pgsch( 1.6 );
            device.pgptxt(0.5, 0.75, 0.0, 0.5, self.pageLabel.center);

    def nextPage(self, device, curPlot, nPlot):
        self.printlegend(device)
        with pgenv(device):
           device.pgpage()
           device.pgsvp( 0.0, 1.0, 0.0, 1.0 )
           device.pgswin( 0.0, 1.0, 0.0, 1.0 )
           device.pgsci( 1 )
        self.printPageLabel(device, curPlot, nPlot)


    _fmt = ("Page({0.layout.nx}x{0.layout.ny} [x={0.xl:.3f}-{0.xr:.3f} y={0.yb:.3f}-{0.yt:.3f}] dx/y={0.dx:.3f} {0.dy:.3f} ddx/y={0.ddx:.3f} {0.ddy:.3f} " +
           "lbl={0.labelCharSz:.3f} tick={0.tickCharSz:.3f} rs={0.rightShift_:.3f} bs={0.bottomShift_:.3f}").format
    def __repr__(self):
        return Page._fmt(self)


Drawers  = enumerations.Enum("Lines", "Points", "Both")
AllInOne = type("AllInOne", (), {})()


##########################################################
####
####      Base class for plotter objects
####
##########################################################
noFilter = lambda x: True
# The label regex's match groups:
#                                          non-capturing 3
#                                          |
#                       1                  v
rxLabel  = re.compile(r'([^ \t\v]+)\s*:\s*((?<![\\])[\'"])((?:.(?!(?<![\\])\2))*.?)\2')
# Standard x/y[n] axis description
#                        1   2
rxXYAxis = re.compile(r'^(x|y([0-9]*))$').match


class Plotter(object):
    #drawfuncs = { Drawers.Points: lambda dev, x, y, tp: dev.pgpt(x, y, tp),
    #              Drawers.Lines:  lambda dev, x, y, tp: dev.pgline(x, y) }

    def __init__(self, desc, xaxis, yaxis, lo, xscaling=None, yscaling=None, yheights=None, drawer=None, **kwargs):
        # determine the number of subplots from the yaxis parameter we limit
        # ourselves to accepting list(...) or anything else (if we would e.g.
        # test wether 'yaxis' was a sequence than a yaxis of "hello" would
        # result in five subplots: ['h', 'e', 'l', 'l', 'o'])
        if not isinstance(yaxis, list):
            yaxis = [yaxis]
        self.xAxis               = CP(xaxis)
        self.yAxis               = CP(yaxis)
        self.yHeights            = [0.97]*len(self.yAxis) if yheights is None else CP(yheights)
        self.Description         = CP(desc)
        self.defaultLayout       = CP(lo)
        self.defaultFixedLayout  = False
        self.defaultDrawer       = CP(Drawers.Points) if drawer is None else CP(drawer)
        self.defaultxScaling     = CP(xscaling) if xscaling is not None else CP(Scaling.auto_global)
        self.defaultyScaling     = CP(yscaling) if yscaling is not None else [Scaling.auto_global] * len(self.yAxis)
        self.defaultsortOrder    = lambda x: hash(x)
        self.defaultsortOrderStr = "(none)"
        self.defaultMarker       = [None] * len(self.yAxis)
        self.defaultMarkerStr    = ["(none)"] * len(self.yAxis)
        self.defaultCkFun        = ckey_builtin
        self.defaultCkFunS       = "ckey_builtin"
        self.defaultFilter       = [noFilter] * len(self.yAxis)
        self.defaultFilterS      = ["(none)"] * len(self.yAxis)
        self.defaultxLabel       = ""
        self.defaultyLabel       = [""] * len(self.yAxis)
        self.defaultShowHeader   = True
        self.defaultShowLegend   = True
        self.defaultShowSource   = True

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
        self.fixedLayout  = CP(self.defaultFixedLayout)
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
        self.charSize     = 0 # zero = auto-scale, otherwise use this value
        self.drawMethod   = CP([""]*len(self.yAxis))
        self.drawers      = CP([[]]*len(self.yAxis))
        self.setDrawer(*str(self.defaultDrawer).split())
        self.xLabel       = CP(self.defaultxLabel)
        self.yLabel       = CP(self.defaultyLabel)
        self.showHeader   = CP(self.defaultShowHeader)
        self.showLegend   = CP(self.defaultShowLegend)
        self.showSource   = CP(self.defaultShowSource)

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
                raise RuntimeError("Invalid draw method(s) specified; use Lines, Points or Both")
            # Depending in which one is the empty list we do things
            # and we complain loudly + bitterly if they're both non-empty ...
            if qualifieds and not unqualifieds:
                for qual in qualifieds:
                    ax                    = qual.group('yAx')
                    if ax not in self.yAxis:
                        raise RuntimeError("The current plot type has no panel for {0}".format( ax ))
                    yIdx                  = self.yAxis.index( ax  )
                    dm                    = qual.group('method').capitalize()
                    self.drawers[yIdx]    = self.drawDict[ dm ]
                    self.drawMethod[yIdx] = dm
            elif unqualifieds and not qualifieds:
                # all unqualified. Only acceptable: 1 unqualified or nYAxis unqualifieds
                if len(unqualifieds)!=len(self.yAxis) and len(unqualifieds)!=1:
                    raise RuntimeError("Incorrect number of drawing methods supplied for plot type (either 1 or {0})".format(len(self.yAxis)))
                # if there's just one, replicate to len yAxis
                methods = unqualifieds if len(unqualifieds)==len(self.yAxis) else [unqualifieds[0]] * len(self.yAxis)
                for (idx, method) in enumerate(methods):
                    dm                   = method.group('method').capitalize()
                    self.drawers[idx]    = self.drawDict[ dm  ]
                    self.drawMethod[idx] = dm
            else:
                raise RuntimeError("You cannot mix qualified axis drawing methods with unqualified ones")
        return " ".join(map(":".join, zip(map(str,self.yAxis), self.drawMethod)))

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
            raise RuntimeError("This plot type has no panel {0}".format(idx))

    # query or set the layout.
    # args is either nothing or a list of strings which are two numbers optionally followed by a list of options:
    #  'fixed'/'flexible' , 'rows'/'columns'
    def layout(self, *args):
        if not args:
            return self.layOut
        nxy     = slice(0,2) if len(args)>=2 else None
        opts    = slice(2,None) if len(args)>2 else None
        if nxy:
            old_rows    = self.layOut.rows
            self.layOut = layout(*map_(int, args[nxy]))
            # New layout object so copy over the existing setting
            # If there is an option overwriting it, that will be
            # handled below
            self.layOut.rows = old_rows
        if opts is not None:
            seen   = set()
            opts   = args[opts]
            for opt in set(map(str.lower, opts)):
                if opt in FixFlex:
                    if FixFlex in seen:
                        raise RuntimeError("Cannot pass mutual exclusive fixed/flexible option at the same time")
                    self.fixedLayout = (opt == 'fixed')
                    seen.add( FixFlex )
                    continue
                if opt in RowsCols:
                    if RowsCols in seen:
                        raise RuntimeError("Cannot pass mutual exclusive rows/columns option at the same time")
                    self.layOut.rows = (opt == 'rows')
                    seen.add( RowsCols )
                    continue
                raise RuntimeError("Unrecognized option '{0}' passed to layout".format(opt))

    def layoutStyle(self):
        return ",".join(["fixed" if self.fixedLayout else "flexible", "rows" if self.layOut.rows else "columns"])

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

    def setCharSize(self, *args):
        if args:
           self.charSize = args[0]
        return self.charSize

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
            self.sortOrder    = hvutil.attrgetter( *map_(str.upper, args) )
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

    def markedPointsForYAxis(self, idx, x, y):
        return self.marker[idx](x, y) if self.marker[idx] else []

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
                             " ".join(map(lambda i_m: "" if i_m[1] is None else "[{0}: {1}]".format(self.yAxis[i_m[0]], self.markerStr[i_m[0]]), enumerate(self.marker)))])
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
        och = dev.pgqch()
        dev.pgsch(self.pointSize)
        dev.pgpt(x, y, tp)
        dev.pgsch(och)

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

    # deal with labelling of the axes
    def haveXLabel(self):
        return self.xLabel
    def haveYLabel(self):
        return functional.filter_(operator.truth, self.yLabel)

    def setLabel(self, *args):
        if args:
            # verify that the whole string was valid (replace all recognized labels with empty string
            # and check if we have anything left that's not whitespace)
            if reduce(operator.add, map(FU.compose(str.strip, operator.itemgetter(0), FU.partial(rxLabel.subn, "")), args)):
                raise RuntimeError("Syntax error in label string(s), not all are formed like <axis> : '<label text>'")
            # if there are arguments given extract "<axis>:'<label text>'" entries
            # "label x:channel  y1: 'Phase (deg)' amplitude:'Flux (Jy)' y:'Oh (noes/fortnight)'"
            # the rxLabel will yield (<axis>, detected quote, <label text>)
            for axis, text in iteritems(dict(map(FU.m_itemgetter(0, 2),
                                                 reduce(operator.add, map(rxLabel.findall, args))))):
                # need to find which axis this is: 'x', 'y[n]' or '<quantity>'
                xyAxis = rxXYAxis(axis.lower())
                if xyAxis:
                    if (xyAxis.group(1) == 'x'):
                        # 'k itsda x-axis
                        self.xLabel = CP(text)
                    else:
                        # default to y-axis #0 if none given
                        yNumber = xyAxis.group(2)
                        yNumber = int(yNumber) if yNumber else 0
                        if yNumber >= len(self.yLabel):
                            raise RuntimeError("y-axis #{0} is out of range (at most {1} available)".format(yNumber, len(self.yLabel)))
                        self.yLabel[ yNumber ] = CP(text)
                else:
                    # ok not simple x/y[n] but actual quantity
                    if axis.lower() == str(self.xAxis).lower():
                        self.xLabel = CP(text)
                    else:
                        # test if quantity is one of the defined y-axes?
                        lyax = map_(FU.compose(str.lower, str), self.yAxis)
                        if axis not in lyax:
                            raise RuntimeError("The indicated y-axis {0} does not apply to this plot".format(axis))
                        self.yLabel[ lyax.index(axis) ] = CP(text)
        # ok return overview of labels (only return the labels that are defined)
        rv = []
        if self.xLabel:
            rv.append( ('x ', self.xAxis, self.xLabel) )
        for n, tp in enumerate(self.yAxis):
            if self.yLabel[n]:
                rv.append( ("y{0}".format(n), tp, self.yLabel[n]) )
        return rv

    # prepare the full page layout based on:
    #    - current layout
    #    - all plots on one page?
    #    - number of plots?
    #    - x/y tick values drawn?
    #    - x/y labels need to be drawn?
    # return:
    #   Page(...)
    def pagestyle(self, device, onePage, nplots, **kwargs):
        # Start with a basic page
        return Page(self, device, onePage, nplots, **kwargs)



##########################################################
####
####          Plot two quantities versus time
####
##########################################################
class Quant2TimePlotter(Plotter):
    def __init__(self, ytypes, yscaling=None, yheights=None, **kwargs):
        super(Quant2TimePlotter, self).__init__("+".join(map(str,ytypes)) + " versus time", jenums.Axes.TIME, ytypes, layout(1,4), yscaling=yscaling, yheights=yheights, **kwargs)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            raise RuntimeError(("first plot ({0}) > #-of-plots ({1})" if len(plotar) else "No plots to plot").format(first, len(plotar)))

        page     = self.pagestyle(device, onePage, plotar, expandy=True)

        # Now we know how many plots/page so we can compute how many plots to actually draw in this iteration
        n        = min(len(plotar), page.layout.nplots()) if onePage else len(plotar)
        last     = min(first + n, len(plotar))

        # Huib, modified to have hours of day
        sday     = 60.0*60.0*24.0
        xmin     = reduce(min, map(lambda x: plotar.limits[x].xlim[0], self.yAxis), float('inf'))
        day0hr   = math.floor(xmin/sday)*sday
        xform_x  = lambda x: x - day0hr

        device.pgbbuf()
        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                pnum = i - first
                if (pnum % page.layout.nplots()) == 0:
                    # before we go to the next page, write the legend
                    page.nextPage(device, i, len(plotar))

                # request the plot area for this plot number
                cvp  = page.viewport(pnum, last-first)

                # get a reference to the plot we should be plotting now
                pref  = plotar[plotlabel]

                ## Loop over the subplots
                for (subplot, ytype) in enumerate(self.yAxis):
                    # filter the data sets with current y-axis type
                    # Keep the indices because we need them twice
                    datasets = functional.filter_(lambda kv: kv[0].TYPE == ytype and self.filter_fun[subplot](kv[0]),
                                                  iteritems(pref))

                    # the type may have been removed due to an expression/selection
                    if not datasets:
                        continue
                    # get the limits of the plots in world coordinates
                    (xlims, ylims) = getXYlims(plotar, ytype, plotlabel, self.xScale, self.yScale[subplot])

                    # get the viewport for this particular subplot
                    vp = cvp.subplot(subplot)

                    # we subtract day0hr from all x-axis values so we must do that with
                    # the x-axis limits too
                    xlims = map_(xform_x, xlims)

                    # tell viewport to draw a framed box - initializes the world coordinates
                    # and based on the current plot's x-axis type we get time or normal x-axis
                    vp.setLabelledBox(device, xlims[0], xlims[1], ylims[0], ylims[1])

                    with pgenv(device):
                        # draw with lines
                        device.pgsls( 1 )
                        for (lab, data) in datasets:
                            # get the colour key for this data set
                            device.pgsci( self.colkey(label(lab, plotar.dslabel), **opts) )
                            # we know there's stuff to display so let's do that then
                            # Any marked data points to display?
                            (mu, mf) = (None, None)
                            # Any unflagged data to display?
                            if data.xval is not None:
                                drap(functional.ylppa(device, xform_x(data.xval), data.yval, -2), self.drawers[subplot])
                                mu = self.markedPointsForYAxis(subplot, data.xval, data.yval)
                            # Any flagged data to display?
                            if data.xval_f is not None:
                                drap(functional.ylppa(device, xform_x(data.xval_f), data.yval_f, 5), self.drawers[subplot])
                                mf = self.markedPointsForYAxis(subplot, data.xval_f, data.yval_f)
                            # draw markers if necessary, temporarily changing line width(markersize)
                            lw = device.pgqlw()
                            device.pgslw(self.markerSize)
                            if mu:
                                device.pgpt( xform_x(data.xval[mu]), data.yval[mu], 7)
                            if mf:
                                device.pgpt( xform_x(data.xval_f[mf]), data.yval_f[mf], 27)
                            device.pgslw(lw)
                            # Any extra drawing commands?
                            self.doExtraCallbacks(device, data, xoffset=day0hr)
                    # do some extra work for panel (subplot) number 0
                    if subplot==0:
                        # 2nd use of the data sets: print the source names at the appropriate times
                        vp.printSrcNameByTime(device, datasets, xform_x)
                        # and draw the main plot label
                        vp.drawMainLabel(device, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )

            # last page with plots also needs a legend, doesn't it?
            page.printlegend(device)
        finally:
            device.pgebuf()



##########################################################
####
####      Generic X vs Y plot
####
##########################################################
class GenXvsYPlotter(Plotter):
    def __init__(self, xtype, ytype, yscaling=None, lo=None, colkey=None):
        super(GenXvsYPlotter, self).__init__(str(ytype)+" versus "+str(xtype), xtype, [ytype], layout(2,4) if lo is None else lo, yscaling=yscaling)
        if colkey is not None:
            self.colkey_fn(colkey)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            raise RuntimeError ("first plot ({0}) > #-of-plots ({1})" if len(plotar) else "No plots to plot").format(first, len(plotar))

        page     = self.pagestyle(device, onePage, plotar, expandy=True)

        # Now we know how many plots/page so we can compute how many plots to actually draw in this iteration
        n        = min(len(plotar), page.layout.nplots()) if onePage else len(plotar)
        last     = min(first + n, len(plotar))

        xmin     = reduce(min, map(lambda x: plotar.limits[x].xlim[0], self.yAxis), float('inf'))
        xform_x  = lambda x: x

        device.pgbbuf()
        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                pnum = i - first
                if (pnum % page.layout.nplots()) == 0:
                    # before we go to the next page, write the legend
                    page.nextPage(device, i, len(plotar))

                # request the plot area for this plot number
                cvp  = page.viewport(pnum, last-first)

                # get a reference to the plot we should be plotting now
                pref  = plotar[plotlabel]

                # filter the data sets with current y-axis type
                # Keep the indices because we need them twice
                datasets = functional.filter_(lambda kv: kv[0].TYPE == self.yAxis[0] and self.filter_fun[0](kv[0]),
                                              iteritems(pref))

                # the type may have been removed due to an expression/selection
                if not datasets:
                    continue
                # get the limits of the plots in world coordinates
                (xlims, ylims) = getXYlims(plotar, self.yAxis[0], plotlabel, self.xScale, self.yScale[0])
                xlims          = map_(xform_x, xlims)

                # get the viewport for this particular subplot
                vp = cvp.subplot(0)

                # tell viewport to draw a framed box - initializes the world coordinates
                # and based on the current plot's x-axis type we get time or normal x-axis
                vp.setLabelledBox(device, xlims[0], xlims[1], ylims[0], ylims[1])

                with pgenv(device):
                    # draw with lines
                    device.pgsls( 1 )
                    for (lab, data) in datasets:
                        # get the colour key for this data set
                        device.pgsci( self.colkey(label(lab, plotar.dslabel), **opts) )
                        # we know there's stuff to display so let's do that then
                        # Any marked data points to display?
                        (mu, mf) = (None, None)
                        # Any unflagged data to display?
                        if data.xval is not None:
                            drap(functional.ylppa(device, xform_x(data.xval), data.yval, -2), self.drawers[0])
                            mu = self.markedPointsForYAxis(0, data.xval, data.yval)
                        # Any flagged data to display?
                        if data.xval_f is not None:
                            drap(functional.ylppa(device, xform_x(data.xval_f), data.yval_f, 5), self.drawers[0])
                            mf = self.markedPointsForYAxis(0, data.xval_f, data.yval_f)
                        # draw markers if necessary, temporarily changing line width(markersize)
                        lw = device.pgqlw()
                        device.pgslw(self.markerSize)
                        if mu:
                            device.pgpt( xform_x(data.xval[mu]), data.yval[mu], 7)
                        if mf:
                            device.pgpt( xform_x(data.xval_f[mf]), data.yval_f[mf], 27)
                        device.pgslw(lw)
                        # Any extra drawing commands?
                        self.doExtraCallbacks(device, data)
                # and draw the main plot label
                vp.drawMainLabel(device, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )

            # last page with plots also needs a legend, doesn't it?
            page.printlegend(device)
        finally:
            device.pgebuf()


##########################################################
####
####      Plot two quantities versus channel
####
##########################################################
class Quant2ChanPlotter(Plotter):
    def __init__(self, ytypes, yscaling=None, yheights=None, **kwargs):
        xt = kwargs.get('xtype', jenums.Axes.CH)
        super(Quant2ChanPlotter, self).__init__("+".join(map(str,ytypes))+" versus "+('channel' if xt is jenums.Axes.CH else xt), xt, ytypes, layout(2,4), yscaling=yscaling, yheights=yheights, **kwargs)

    def drawfunc(self, device, plotar, first, onePage=None, **opts):
        # onePage == None? I.e. plot all. I.E. start from beginning!
        if onePage in [None, AllInOne]:
            first = 0

        # Check for sensibility in caller.
        if first>=len(plotar):
            raise RuntimeError ("first plot ({0}) > #-of-plots ({1})" if len(plotar) else "No plots to plot").format(first, len(plotar))

        page     = self.pagestyle(device, onePage, plotar, expandy=True)

        # Now we know how many plots/page so we can compute how many plots to actually draw in this iteration
        n        = min(len(plotar), page.layout.nplots()) if onePage else len(plotar)
        last     = min(first + n, len(plotar))

        xmin     = reduce(min, map(lambda x: plotar.limits[x].xlim[0], self.yAxis), float('inf'))
        identity = lambda x: x

        # We may need to have the real frequencies
        try:
            mysm = ms2util.makeSpectralMap( plotar.msname )
        except RuntimeError as E:
            mysm = None
            print("Failed to make spectral map: ",E)

        device.pgbbuf()
        try:
            # reset the colour-key mapping; indicate we start a new plot
            self.colkey_reset(CKReset.newplot)

            # retrieve the plotlabels. We count the plots numerically but address them in the plotar
            # (which, in reality, is a Dict() ...) by their key
            for (i, plotlabel) in hvutil.enumerateslice(sorted(plotar.keys(), key=self.sortOrder), first, last):
                # <pnum> is the actual counter of the plots we're creating
                #        (potentiall spans > 1 page)
                pnum = i - first
                if (pnum % page.layout.nplots()) == 0:
                    # before we go to the next page, write the legend
                    page.nextPage(device, i, len(plotar))

                # request the plot area for this plot number
                cvp  = page.viewport(pnum, last-first)

                # get a reference to the plot we should be plotting now
                pref      = plotar[plotlabel]
                plotxlims = None # keep track of altered x-limits (for multisubband)
                ## Loop over the subplots
                for (subplot, ytype) in enumerate(self.yAxis):
                    # filter the data sets with current y-axis type
                    # Keep the indices because we need them twice
                    datasets = functional.filter_(lambda kv: kv[0].TYPE == ytype and self.filter_fun[subplot](kv[0]),
                                                  iteritems(pref))

                    # the type may have been removed due to an expression/selection
                    if not datasets:
                        continue
                    # get the limits of the plots in world coordinates
                    (xlims, ylims) = getXYlims(plotar, ytype, plotlabel, self.xScale, self.yScale[subplot])

                    ## we support two types of plots, actually:
                    ##   * all subbands on top of each other
                    ##   * all subbands next to each other
                    xoffset  = {}

                    if self.multiSubband:
                        # make new x-axes based on the subband attribute of the label(s)
                        (xoffset, xlims[1]) = mk_offset(datasets, 'SB')

                    # Do x-limits bookkeeping. If this is subplot #0 store the current limits. Otherwise
                    # copy the x-limits to overwrite what we got from this data set
                    if subplot==0:
                        plotxlims = copy.deepcopy(xlims)
                    else:
                        xlims = copy.deepcopy(plotxlims)

                    # get the viewport for this particular subplot
                    vp = cvp.subplot(subplot)

                    # tell viewport to draw a framed box - initializes the world coordinates
                    # and based on the current plot's x-axis type we get time or normal x-axis
                    vp.setLabelledBox(device, xlims[0], xlims[1], ylims[0], ylims[1])

                    with pgenv(device):
                        # draw with lines
                        device.pgsls( 1 )
                        for (lab, data) in datasets:
                            # get the colour key for this data set
                            device.pgsci( self.colkey(label(lab, plotar.dslabel), **opts) )
                            # get the x-axis transformation for the current data set
                            xform_x = xoffset.get(lab.SB, identity)
                            # we know there's stuff to display so let's do that then
                            # Any marked data points to display?
                            (mu, mf) = (None, None)
                            # Any unflagged data to display?
                            if data.xval is not None:
                                drap(functional.ylppa(device, xform_x(data.xval), data.yval, -2), self.drawers[subplot])
                                mu = self.markedPointsForYAxis(subplot, data.xval, data.yval)
                            # Any flagged data to display?
                            if data.xval_f is not None:
                                drap(functional.ylppa(device, xform_x(data.xval_f), data.yval_f, 5), self.drawers[subplot])
                                mf = self.markedPointsForYAxis(subplot, data.xval_f, data.yval_f)
                            # draw markers if necessary, temporarily changing line width(markersize)
                            lw = device.pgqlw()
                            device.pgslw(self.markerSize)
                            if mu:
                                device.pgpt( xform_x(data.xval[mu]), data.yval[mu], 7)
                            if mf:
                                device.pgpt( xform_x(data.xval_f[mf]), data.yval_f[mf], 27)
                            device.pgslw(lw)
                            # Any extra drawing commands?
                            self.doExtraCallbacks(device, data)
                    # do some extra work for panel (subplot) number 0
                    if subplot==0:
                        # and draw the main plot label
                        vp.drawMainLabel(device, "{0:s}".format(label.format(plotlabel.attrs(plotar.plotlabel))) )
                        # indicate multi subband if set
                        with pgenv(device):
                            device.pgsch( 0.5 )
                            frqedge = None
                            if all(map(functools.partial(operator.ne, None), [plotlabel.FQ, plotlabel.SB, mysm])):
                                frqedge = "{0:.3f}MHz".format( mysm.frequencyOfFREQ_SB(plotlabel.FQ, plotlabel.SB)/1.0e6 )
                            elif self.multiSubband and len(xoffset)>1:
                                frqedge = "multi SB"
                            if frqedge:
                                device.pgmtxt( 'B', -1, 0.01, 0.0, frqedge )

            # last page with plots also needs a legend, doesn't it?
            page.printlegend(device)
        finally:
            device.pgebuf()



######
###### Utilities
######

#  19 Dec 2017 MarkK comes to me saying the plots are borkened!
#              Plotting ratios of two data sets that are basically equal (i.e.
#              ratio==1.00000000...)  some of them are displayed as
#              0.14000000xyz. First thought it was NaN clogging up the display
#              but that didn't fix anything.
#
#              Traced to here - "dy" (ylims[1] - ylims[0]) was ~3.147e-6 which
#              is larger than the "dy" limit of 1e-6 (which I had arbitrarily
#              set, apparently) but small enough to break PGPLOT plotting. So
#              now we try to work out the machine epsilon and check if "dy" >
#              few tens of epsilons and if it isn't then make the y-range
#              artificially larger than the span of the data in the y-direction.
#
#              PGPLOT pgswin() takes "REAL" (aka float32) as parameters so
#              probably we're running into some machine precision probs here if
#              the range between Y2 and Y1 is too small. Of course this applies
#              to X1 and X2 as well
#
#              Finding epsilon for a given IEEE floating point type:
#              http://rstudio-pubs-static.s3.amazonaws.com/13303_daf1916bee714161ac78d3318de808a9.html
F       = numpy.float32
epsilon = abs((F(7)/F(3)) - (F(4)/F(3)) - F(1))

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
        # make sure dx is positive and non-zero. SEE NOTE ABOVE
        # Apparently x-axis of ~1E-6 is still OK.
        # Apparently not, go to 100*epsilon in stead
        dx = max(dx, 100*epsilon) #1e-6)
        if not fixed:
            mid      = (xlims[1] + xlims[0])/2
            xlims[0] = mid - 0.55*dx
            xlims[1] = mid + 0.65*dx

        ## Repeat for Y
        fixed = False
        if yscaling == Scaling.auto_local:
            ylims = list(plotarray.meta[curplotlabel][ytype].ylim)
        elif yscaling == Scaling.auto_global:
            ylims = list(plotarray.limits[ytype].ylim)
        else:
            ylims = list(copy.deepcopy(yscaling))
            fixed = True
        dy = ylims[1] - ylims[0]
        # id. for y range  (see NOTE above)
        dy = max(dy, 100*epsilon)
        if not fixed:
            mid      = (ylims[1] + ylims[0])/2
            ylims[0] = mid - 0.55*dy
            ylims[1] = mid + 0.65*dy
        return (xlims, ylims)


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
        'ampfreq': Quant2ChanPlotter([YTypes.amplitude], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines, xtype='frequency'),
        'phachan': Quant2ChanPlotter([YTypes.phase], yscaling=[[-185, 185]], yheights=[0.97], drawer=Drawers.Lines),
        'phafreq': Quant2ChanPlotter([YTypes.phase], yscaling=[[-185, 185]], yheights=[0.97], drawer=Drawers.Lines, xtype='frequency'),
        'anpchan': Quant2ChanPlotter([YTypes.amplitude, YTypes.phase], yscaling=[Scaling.auto_global, [-185, 185]], \
                                     yheights=[0.58, 0.38], drawer=Drawers.Lines.value+" "+Drawers.Points.value),
        'anpfreq': Quant2ChanPlotter([YTypes.amplitude, YTypes.phase], yscaling=[Scaling.auto_global, [-185, 185]], \
                                     yheights=[0.58, 0.38], drawer=Drawers.Lines.value+" "+Drawers.Points.value, xtype='frequency'),
        'rechan' : Quant2ChanPlotter([YTypes.real], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines),
        'imchan' : Quant2ChanPlotter([YTypes.imag], yscaling=[Scaling.auto_global], yheights=[0.97], drawer=Drawers.Lines),
        'rnichan': Quant2ChanPlotter([YTypes.real, YTypes.imag], yscaling=[Scaling.auto_global, Scaling.auto_global],
                                     yheights=[0.48, 0.48], drawer=Drawers.Lines),
        # generic
        'uv'     : GenXvsYPlotter('U', 'V', lo=layout(2,2), colkey='src'),
        'ampuv'  : GenXvsYPlotter('UV distance (lambda)', YTypes.amplitude, lo=layout(2,2)),
        'phauv'  : GenXvsYPlotter('UV distance (lambda)', YTypes.phase, yscaling=[[-185, 185]], lo=layout(2,2))
}


Types = enumerations.Enum(*Plotters.keys())
