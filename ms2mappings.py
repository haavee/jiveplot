# keeps mappings of metadata names to numbers for metadata in a measurement set
# $Id: ms2mappings.py,v 1.8 2017-01-27 13:50:28 jive_cc Exp $
# $Log: ms2mappings.py,v $
# Revision 1.8  2017-01-27 13:50:28  jive_cc
# HV: * jplotter.py: small edits
#         - "not refresh(e)" => "refresh(e); if not e.plots ..."
#         - "e.rawplots.XXX" i.s.o. "e.plots.XXX"
#     * relatively big overhaul: in order to force (old) pyrap to
#       re-read tables from disk all table objects must call ".close()"
#       when they're done.
#       Implemented by patching the pyrap.tables.table object on the fly
#       with '__enter__' and '__exit__' methods (see "ms2util.opentable(...)")
#       such that all table access can be done in a "with ..." block:
#          with ms2util.opentable(...) as tbl:
#             tbl.getcol('DATA') # ...
#       and then when the with-block is left, tbl gets automagically closed
#
# Revision 1.7  2015-09-23 12:28:36  jive_cc
# HV: * Lorant S. requested sensible requests (ones that were already in the
#       back of my mind too):
#         - option to specify the data column
#         - option to not reorder the spectral windows
#       Both options are now supported by the code and are triggered by
#       passing options to the "ms" command
#
# Revision 1.6  2014-04-08 22:41:11  jive_cc
# HV: Finally! This might be release 0.1!
#     * python based plot iteration now has tolerable speed
#       (need to test on 8M row MS though)
#     * added quite a few plot types, simplified plotters
#       (plotiterators need a round of moving common functionality
#        into base class)
#     * added generic X/Y plotter
#
# Revision 1.5  2013-08-20 18:23:50  jive_cc
# HV: * Another savegame
#       Got plotting to work! We have stuff on Screen (tm)!
#       Including the fance standardplot labels.
#       Only three plot types supported yet but the bulk of the work should
#       have been done I think. Then again, there's still a ton of work
#       to do. But good progress!
#
# Revision 1.4  2013-06-19 12:28:44  jive_cc
# HV: * making another savegame
#
# Revision 1.3  2013-02-19 16:53:29  jive_cc
# HV: * About time to commit - make sure all edits are safeguarded.
#       Making good progress. baselineselection, sourceselection and
#       timeselection working
#
# Revision 1.2  2013-01-29 12:23:45  jive_cc
# HV: * time to commit - added some more basic stuff
#
import sys, ms2util, jenums

class FailedToLoadMS(Exception):
    def __init__(self, ms):
        self.measurementset = ms

    def __str__(self):
        return "Failed to load measurementset {0}".format(self.measurementset)

class mappings:
    def __init__(self):
        self.reset()

    def valid(self):
        return self.spectralMap and self.baselineMap and self.polarizationMap \
               and self.fieldMap and self.timeRange and self.domain is not None \
               and self.project

    def reset(self):
        self.spectralMap     = None
        self.baselineMap     = None
        self.polarizationMap = None
        self.fieldMap        = None
        self.timeRange       = None
        self.domain          = None
        self.project         = None
        self.numRows         = 0

    # load mappings from a path, allegedly pointing at a MeasurementSet
    def load(self, nm, **kwargs):
        # join local default(s) with user settings
        options = dict({"unique": True}, **kwargs)
        limit   = 500000

        # raises in case of fishy
        ms2util.assertMinimalMS(nm)

        # test length of MS. If we think it's too long (say >500000 rows) and
        # 'unique' is True, then warn + ask 
        with ms2util.opentable(nm) as m:
            if False: #len(m)>limit and defaults['unique']:
                print
                print "===> WARNING: this MS contains {0} rows and you requested".format(len(m))
                print "     'unique' meta data values - i.e. only meta data which"
                print "     is actually used in the main table."
                print
                print "     Finding this out on a MS this size usually takes (a lot of)"
                print "     time. If you do not want this, cancel when you're asked and"
                print "     restart jplotter.jcli() with 'unique=False' keyword:"
                print "         .... "
                print "         >>> jplotter.jcli(unique=False)"
                print "     otherwise the code will continue as normal"
                print
                ans = None
                while not ans:
                    a = raw_input("Do you wish to continue? [y]/n ").lower()
                    if not a:
                        a = "n"
                    if a in "yn":
                        ans = a
                if ans=="n":
                    raise RuntimeError, "Opening of MS cancelled by user"

            def logit(x):
                sys.stdout.write("{0} ... {1}                            \r".format(nm, x))
                sys.stdout.flush()

            # start reading the mappings
            logit("spectral map       1/5")
            spectralMap     = ms2util.makeSpectralMap(nm, **options)
            logit("baseline map       2/5")
            baselineMap     = ms2util.makeBaselineMap(nm, **options)
            logit("polarization map   3/5")
            polarizationMap = ms2util.makePolarizationMap(nm, **options)
            logit("field map          4/5")
            fieldMap        = ms2util.makeFieldMap(nm, **options)
            logit("timerange          5/5")
            timeRange       = ms2util.getTimeRange(nm, **options)
            domain          = ms2util.getDataDomain(nm, **options)
            project         = ms2util.getProject(nm, **options)
            sys.stdout.write("                                                        \r")
            sys.stdout.flush()

            # clear ourselves
            self.reset()
            self.spectralMap     = spectralMap
            self.baselineMap     = baselineMap
            self.polarizationMap = polarizationMap
            self.fieldMap        = fieldMap
            self.timeRange       = timeRange
            self.domain          = domain
            self.project         = project
            self.numRows         = len(m)

