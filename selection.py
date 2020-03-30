# holds the user selection
# $Id: selection.py,v 1.10 2014-05-14 17:02:02 jive_cc Exp $
# $Log: selection.py,v $
# Revision 1.10  2014-05-14 17:02:02  jive_cc
# HV: * Weight thresholding implemented - but maybe I'll double the code
#       to two different functions, one with weight thresholding and one
#       without because weight thresholding is sloooooow
#
# Revision 1.9  2014-04-14 14:46:05  jive_cc
# HV: * Uses pycasa.so for table data access waiting for pyrap to be fixed
#     * added "indexr" + scan-based selection option
#
# Revision 1.8  2014-04-10 21:14:41  jive_cc
# HV: * I fell for the age-old Python trick where a default argument is
#       initialized statically - all data sets were integrating into the
#       the same arrays! Nice!
#     * Fixed other efficiency measures: with time averaging data already
#       IS in numarray so no conversion needs to be done
#     * more improvements
#
# Revision 1.7  2013-12-12 14:10:17  jive_cc
# HV: * another savegame. Now going with pythonic based plotiterator,
#       built around ms2util.reducems
#
# Revision 1.6  2013-06-19 12:28:44  jive_cc
# HV: * making another savegame
#
# Revision 1.5  2013-03-31 17:17:56  jive_cc
# HV: * another savegame
#
# Revision 1.4  2013-03-09 16:59:07  jive_cc
# HV: * another savegame
#
# Revision 1.3  2013-02-19 16:53:29  jive_cc
# HV: * About time to commit - make sure all edits are safeguarded.
#       Making good progress. baselineselection, sourceselection and
#       timeselection working
#
# Revision 1.2  2013-01-29 12:23:45  jive_cc
# HV: * time to commit - added some more basic stuff
#
from   six        import iteritems
from   functools  import reduce
from   functional import compose, is_not_none, map_
import jenums, hvutil, copy, operator

# how to format a time range "(start, end)" as TaQL
fmt_time_cond = "(TIME>={0[0]:.7f} && TIME<={0[1]:.7f})".format
fmt_dd_select = "{0}/{1}/{2}:{3}".format
range_repr    = compose(hvutil.range_repr, hvutil.find_consecutive_ranges)

class selection:
    def __init__(self):
        self.init()

    def init(self):
        self.chanSel         = None
        self.polSel          = None
        self.scanSel         = []
        self.plotType        = None
        self.solint          = None # AIPS legacy :D
        self.solchan         = None # AIPS legacy :D
        self.averageTime     = jenums.Averaging.NoAveraging
        self.averageChannel  = jenums.Averaging.NoAveraging
        self.taqlString      = None
        self.weightThreshold = None
        # We keep the baseline selection as human readable format (is an array of
        # strings) and a TaQL version. This seems a bit overdone but the one is
        # much easier to read (guess which one) and the other is waaaaay more
        # efficient when selecting the data (that'll be the other one, but I trust
        # you to have guessed that one...)
        self.baselines          = None
        self.baselinesTaql      = None
        self.sources            = None
        self.sourcesTaql        = None
        self.timeRange          = None
        self.timeRangeTaql      = None
        # ddSelection = [ (fq, sb, polid, [product index, ...]), ... ]
        self.ddSelection        = None
        self.ddSelectionTaql    = None
        # when to start new plots. By default each baseline + freqgroup
        # ends up in a different plot
        self.newPlot = dict([(x, False) for x in jenums.Axes])
        self.newPlot[jenums.Axes.FQ] = True
        self.newPlot[jenums.Axes.BL] = True

    # FIXME XXX FIXME
    # should check in jplotter.py if selected time range is larger than the 
    # whole data set because then we can decide to NOT add a time condition, making
    # the query faster
    def selectTimeRange(self, timeranges):
        self.timeRange     = copy.deepcopy(timeranges)
        self.timeRangeTaql = "(" + " OR ".join(map(fmt_time_cond, self.timeRange)) + ")"

    @classmethod
    def group_sb(cls, acc, f_s_p_prods):
        (f, s, p, prods) = f_s_p_prods
        key = (f, p, str(prods))
        acc.setdefault(key, set()).add(s)
        return acc
    @classmethod
    def proc_sb(cls, f_p_ps_v):
        ((f,p,ps), v) = f_p_ps_v
        return (f, list(v), p, eval(ps))
    @classmethod
    def group_fq(cls, acc, f_s_p_l):
        (f,s,p,l) = f_s_p_l
        key = (str(s), p, str(l))
        acc.setdefault(key, set()).add(f)
        return acc
    @classmethod
    def proc_fq(cls, ss_p_ps_v):
        ((ss, p, ps), v) = ss_p_ps_v
        return (list(v), eval(ss), p, eval(ps))
    @classmethod
    def fmter(cls, pMap):
        def do_it(f_s_p_l):
            (f,s,p,l) = f_s_p_l
            pols      = ",".join(hvutil.itemgetter(*l)(pMap.getPolarizations(p)))
            return fmt_dd_select(range_repr(f), range_repr(s), p, pols)
        return do_it

    # the ddSelection as Human Readable Format. Needs a 
    # polarization map to unmap indices to string
    def ddSelectionHRF(self, pMap):
        if not self.ddSelection:
            return ["No frequency selection yet"]
        # For the Human Readable Format we have to do two more reductions ...
        #    ddSelection = [(fq, sb, polid, [prods]), ...]
        # 1) group all subbands from the same fq, polid, prods together to form:
        #    ddSelection = [(fq, [sb], polid, [prods]), ...]
        # 2) then check if there are multiple fq's that have the same
        #    subband/polid/prods selection - such that the list can be
        #    written as:
        #    ddSelection = [([fq], [sb], polid, [prods]), ...]

        # Group together all the subbands for the same (fq, pol, [prods]) 
        # such that we end up with 
        #    ddSelection = [(fs, [sb], pol, [prods]), ....]
        # once more dict + set to the rescue
        hrf = map(selection.proc_sb, iteritems(reduce(selection.group_sb, self.ddSelection, {})))

        # each entry can already be listed as one, abbreviated, line Let's see
        # if there are multiple FQs who have the same [sb], pol, [prods]
        # selection and group them together. Again: dict + set to the rescue
        hrf = map(selection.proc_fq, iteritems(reduce(selection.group_fq, hrf, {})))

        # now we can produce the output
        return map_(selection.fmter(pMap), hrf)

    def selectionTaQL(self):
        # If an explicit TaQL string is set, return that one
        if self.taqlString:
            return self.taqlString

        # No? Ok, build the query
        return " AND ".join(filter(operator.truth, 
                                   [self.baselinesTaql, self.sourcesTaql, self.timeRangeTaql, self.ddSelectionTaql]))
        #return reduce(lambda acc, x: (acc+" AND "+x if acc else x) if x else acc, \
        #              filter(is_not_none, \
        #                     [self.baselinesTaql, self.sourcesTaql, self.timeRangeTaql, self.ddSelectionTaql]), \
        #              "")

    def mkCPPNewplot(self):
        # return a list of True/False values for all plot axes in the order the C++
        # expects them
        return map_(lambda x: self.newPlot[x], [jenums.Axes.P, jenums.Axes.CH, jenums.Axes.SB,jenums.Axes.FQ,
                                               jenums.Axes.BL, jenums.Axes.SRC, jenums.Axes.TIME, jenums.Axes.TYPE])



