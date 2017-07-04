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
import jenums, hvutil, copy

class selection:
    def __init__(self):
        self.init()

    def init(self):
        self.chanSel         = None
        self.polSel          = None
        self.scanSel         = []
        self.plotType        = None
        self.solint          = None # AIPS legacy :D
        self.averageTime     = jenums.Averaging.None
        self.averageChannel  = jenums.Averaging.None
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

    def selectTimeRange(self, timeranges):
        self.timeRange     = copy.deepcopy(timeranges)
        self.timeRangeTaql = "(" + " OR ".join(map(lambda (s,e): "(TIME>={0:.7f} && TIME<={1:.7f})".format(s,e), self.timeRange)) + ")"

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
        def group_sb(acc, (f, s, p, prods)):
            key = (f, p, str(prods))
            acc[key].add(s) if key in acc else acc.update({key:set([s])})
            return acc
        hrf = map(lambda ((f,p,ps), v): (f,list(v),p,eval(ps)), \
                  reduce(group_sb, self.ddSelection, {}).iteritems())

        # each entry can already be listed as one, abbreviated, line Let's see
        # if there are multiple FQs who have the same [sb], pol, [prods]
        # selection and group them together. Again: dict + set to the rescue
        def group_fq(acc, (f,s,p,l)):
            key = (str(s), p, str(l))
            acc[key].add(f) if key in acc else acc.update({key:set([f])})
            return acc
        hrf = map(lambda ((ss, p, ps), v): (list(v), eval(ss), p, eval(ps)), \
                  reduce(group_fq, hrf, {}).iteritems())

        # now we can produce the output
        def fmter( (f,s,p,l) ):
            return "{0}/{1}/{2}:{3}".format( \
                    hvutil.range_repr(hvutil.find_consecutive_ranges(f)), \
                    hvutil.range_repr(hvutil.find_consecutive_ranges(s)), \
                    p, \
                    ",".join(hvutil.itemgetter(*l)(pMap.getPolarizations(p))) \
                    )
        return map(fmter, hrf)

    def selectionTaQL(self):
        # If an explicit TaQL string is set, return that one
        if self.taqlString:
            return self.taqlString

        # No? Ok, build the query
        return reduce(lambda acc, x: (acc+" AND "+x if acc else x) if x else acc, \
                      filter(lambda z: z is not None, \
                             [self.baselinesTaql, self.sourcesTaql, self.timeRangeTaql, self.ddSelectionTaql]), \
                      "")

    def mkCPPNewplot(self):
        # return a list of True/False values for all plot axes in the order the C++
        # expects them
        return map(lambda x: self.newPlot[x], [jenums.Axes.P, jenums.Axes.CH, jenums.Axes.SB,jenums.Axes.FQ,
                                               jenums.Axes.BL, jenums.Axes.SRC, jenums.Axes.TIME, jenums.Axes.TYPE])



