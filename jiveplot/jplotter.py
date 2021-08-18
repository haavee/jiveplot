#  Interface to the plotiterator.
#
#  Well... it's a sort of a high level interface so the user does not
#  have to worry about having to translate DATA_DESC_IDs etc.
#
#  The info will be presented in a sort of HRF (Human Readable Format)
#
#  Rewrite from Ancient glish code to Python by H. Verkouter
#   Dec 2012
#
#   $Id: jplotter.py,v 1.73 2017/08/29 11:27:36 jive_cc Exp $
#
#   $Log: jplotter.py,v $
#   Revision 1.73  2017/08/29 11:27:36  jive_cc
#   HV: * Added post-processing-pre-display filtering of datasets to be plotted
#         This allows e.g. per subplot (think: amp+phase) selection of things to
#         plot - users wanted to display phase of parallel pols but amp of
#         everything:
#           pt anpchan; filter phase: p in [ll,rr];
#
#   Revision 1.72  2017-04-26 12:27:17  jive_cc
#   HV: * ZsoltP + BenitoM: "in amplitude+phase plots it would be much
#         better to have phase drawn with points and amplitude with lines"
#         So now it is possible to change the drawing style of subplots
#         individually in stead of just globally for all subplots.
#         See "help draw" for examples.
#
#   Revision 1.71  2017-04-24 10:10:26  jive_cc
#   HV: * Based on BenitoM's comment added the 'x' and 'y' commands as separate
#         commands such that you can now type "help x" or "help y" and the
#         axis-scaling help comes up like any sane person would expect
#
#   Revision 1.70  2017-03-20 14:54:55  jive_cc
#   HV: * The "add column name" patch broke UV plotting. Now it should work OK
#
#   Revision 1.69  2017-03-10 13:32:28  jive_cc
#   HV: * feature request (IlsevB) "put data column name in plot metadata"
#
#   Revision 1.68  2017-01-27 13:50:28  jive_cc
#   HV: * jplotter.py: small edits
#           - "not refresh(e)" => "refresh(e); if not e.plots ..."
#           - "e.rawplots.XXX" i.s.o. "e.plots.XXX"
#       * relatively big overhaul: in order to force (old) pyrap to
#         re-read tables from disk all table objects must call ".close()"
#         when they're done.
#         Implemented by patching the pyrap.tables.table object on the fly
#         with '__enter__' and '__exit__' methods (see "ms2util.opentable(...)")
#         such that all table access can be done in a "with ..." block:
#            with ms2util.opentable(...) as tbl:
#               tbl.getcol('DATA') # ...
#         and then when the with-block is left, tbl gets automagically closed
#
#   Revision 1.67  2016-01-11 15:30:30  jive_cc
#   HV: * channel averaging display 'crashed' if no channel selection was
#         present (chanSel==None => no channels selected => i.e. all channels)
#
#   Revision 1.66  2016-01-11 13:27:19  jive_cc
#   HV: * "goto page N" was broken in interactive mode. Now just type
#         <digits><enter> to go to a specific page
#
#   Revision 1.65  2015-12-09 07:02:11  jive_cc
#   HV: * big change! the plotiterators now return a one-dimensional dict
#         of label => dataset. The higher level code reorganizes them
#         into plots, based on the 'new plot' settings. Many wins in this one:
#           - the plotiterators only have to index one level in stead of two
#           - when the 'new plot' setting is changed, we don't have to read the
#             data from disk again [this is a *HUGE* improvement, especially for
#             larger data sets]
#           - the data set expression parser is simpler, it works on the
#             one-dimensional 'list' of data sets and it does not have to
#             flatten/unflatten any more
#       * The code to deal with refreshing the plots has been rewritten a bit
#         such that all necessary steps (re-organizing raw plots into plots,
#         re-running the label processing, re-running the post processing,
#         re-running the min/max processing) are executed only once; when
#         necessary. And this code is now shared between the "pl" command and
#         the "load/store" commands.
#
#   Revision 1.64  2015-10-02 14:47:56  jive_cc
#   HV: * multi subband plotting:
#           - default is now 'False' in stead of 'None'
#           - multi command does not accept "1" or "0" any more
#           - multi command accepts t(rue) or f(alse)
#
#   Revision 1.63  2015-09-23 12:28:36  jive_cc
#   HV: * Lorant S. requested sensible requests (ones that were already in the
#         back of my mind too):
#           - option to specify the data column
#           - option to not reorder the spectral windows
#         Both options are now supported by the code and are triggered by
#         passing options to the "ms" command
#
#   Revision 1.62  2015-09-21 11:36:22  jive_cc
#   HV: * The big push. The 'command.py' interface long left something
#         to be desired if a command specified an 'args()' method to transform
#         the line of input into multiple arguments and call the callback function
#         with those arguments [hint: it wouldn't].
#         Now the command.py object calls the callback as:
#           "cmd.callback( *cmd.args(input) )"
#       * all commands have been visited and modified to be specified as
#            "def xyz( .., *args)"
#         and do the right thing if called with wrong number of arguments.
#         Many command's regex's got simpler.
#       * fixed some error messages to be more readable
#       * extended the documentation of x, y scaling of plots ("xyscale")
#       * clear error message if adressing an invalid panel in a plot
#       * the 'new plot' specification was widened: can now use comma-separated
#         list of axes to set the new plot value in one go:  "new p,sb true"
#       * the "cd" command now does '~' expansion at the beginning of the path
#       * fixed bug in pgplot device name generation in case details were left out
#
#   Revision 1.61  2015-09-16 14:56:26  jive_cc
#   HV: * Feature request by Lorant to be able to show currently defined variables
#
#   Revision 1.60  2015-09-02 12:48:53  jive_cc
#   HV: * go to absolute page number is now also 1-based, to be consistent with
#         page numbering as displayed on the plot page(s)
#
#   Revision 1.59  2015-09-02 07:49:40  jive_cc
#   HV: * plots on screen are now black-on-white too
#       * helptext for 'ckey' was not linked to full, extented, helpfile text
#
#   Revision 1.58  2015-07-08 07:56:54  jive_cc
#   HV: * added integration time as global symbol in time selection 't_int'
#       * updated and corrected "time" documentation
#
#   Revision 1.57  2015-07-07 10:29:29  jive_cc
#   HV: * opening a different MS clears the selection, as it should've been from
#         the start
#
#   Revision 1.56  2015-06-26 12:46:11  jive_cc
#   HV: * let os.system() handle "ls" command
#
#   Revision 1.55  2015-04-08 14:35:24  jive_cc
#   HV: * prevent memory usage explosion
#
#   Revision 1.54  2015-03-26 20:10:08  jive_cc
#   HV: * flatfilter() sorts the data sets by increasing x-value. This is
#         very important when executing data set expressions ("load", "store")
#       * help text for "load" and "store" now actually refers to the full text,
#         not the one-line summary.
#       * add support for the "ckey" command - programmable data set colouring
#         is now possible. data sets can be assigned explicit colours and/or
#         based on a diferent set of attribute values than the default.
#
#   Revision 1.53  2015-03-11 13:54:20  jive_cc
#   HV: * when loading/storing an expression producing plots
#         any installed post-processing is also run
#       * dataset expression parser changes:
#         - loose the '$' for variable dereferencing; the
#           name resolving has become smarter
#         - prepare to support calling functions-with-arguments
#           a name followed by function-call operator '(' <args> ')'
#
#   Revision 1.52  2015-02-02 08:55:21  jive_cc
#   HV: * support for storing/loading plots, potentially manipulating them
#         via arbitrary arithmetic expressions
#       * helpfile layout improved
#
#   Revision 1.51  2015-01-09 14:27:57  jive_cc
#   HV: * fixed copy-paste error in weight-thresholded quantity-versus-time fn
#       * sped up SOLINT processing by factor >= 2
#       * output of ".... took    XXXs" consistentified & beautified
#       * removed "Need to convert ..." output; the predicted expected runtime
#         was usually very wrong anyway.
#
#   Revision 1.50  2015-01-09 00:02:27  jive_cc
#   HV: * support for 'solint' - accumulate data in time bins of size 'solint'
#         now also in "xxx vs time" plots. i.e. can be used to bring down data
#         volume by e.g. averaging down to an arbitrary amount of seconds.
#       * solint can now be more flexibly be set using d(ays), h(ours),
#         m(inutes) and/or s(econds). Note that in the previous versions a
#         unitless specification was acceptable, in this one no more.
#
#   Revision 1.49  2015-01-06 13:53:25  jive_cc
#   HV: * improved error handling when trying to "cd" whilst in a closed
#         plot environment
#
#   Revision 1.48  2015-01-06 12:09:53  jive_cc
#   HV: * fixed lost precision issue in generating TIME part of TaQL qry when
#         selecting data from scan(s)
#
#   Revision 1.47  2014-12-19 14:25:35  jive_cc
#   HV: * cleaned up "mark" command a bit; now more consistent
#       * index now must have ":" to separate it from the expression
#       * edited built-in help to reflect this syntax change
#
#   Revision 1.46  2014-12-19 11:24:06  jive_cc
#   HV: * Ability to close current plot device. Note that your environment goes with it ...
#
#   Revision 1.45  2014-12-19 10:33:31  jive_cc
#   HV: * can now load post-processing code using absolute path
#
#   Revision 1.44  2014-12-19 10:08:29  jive_cc
#   HV: * Plotting can now safely be interrupted using ^C, w/o exiting the whole program
#
#   Revision 1.43  2014-12-11 21:25:03  jive_cc
#   HV: * define extra colours, if the device supports more colours. we now have
#         48 colours available.
#
#   Revision 1.42  2014-12-09 16:37:51  jive_cc
#   HV: * Added possibility of adding post-processing of data sets -
#         can load/run MODULE.FUNCTION() to operate on the generated datasets.
#         The function can add extra drawfunctions to the data set such that
#         extra things can be drawn or information can be printed to stdout.
#       * The page legend code has been modified such that the space below the
#         plot is used more efficiently, taking into account the size of the
#         longest data set label, so the layout is more dynamic.
#
#   Revision 1.41  2014-11-28 14:31:08  jive_cc
#   HV: * fixed two locations where "self.dirty" was not handled properly
#
#   Revision 1.40  2014-09-02 16:20:09  jive_cc
#   HV: * fix crash/untimely demise if trying to plot to a file which failed to
#         open
#       * support for setting line width, point size and marker size for those
#         respective drawing elements (and querying them, ofc).
#
#   Revision 1.39  2014-08-08 15:38:41  jive_cc
#   HV: * Added 'scripted' to command.py - a command-source for the commandline
#         object, assuming the object generates commands. Can be generator or
#         iterable.
#       * jplotter gets '-h' and '-v' and '-d' command line flags
#       * 'jcli' is now called 'run_plotter' and takes a commandline instance
#         as parameter; the caller determines where the commands for the plotter
#         are read from
#       * modifications to 'jcli/run_plotter' to properly support plotting to
#         file directly ... this is a bit of a hidden trick; the name of
#         the environment is kept the same but the pgplot device in it is
#         changed to a file. Nasty, but it works. PGPLOT only allows one
#         file to be open at any point in time *sigh*.
#       * support for "/cps" or "/vcps" ending in the postscript file names
#         for 'save' and 'file' commands
#       * added first version of pythonified standardplots!!!!!!!!!!
#         Finally!!! After having been on the RnDM agenda for YEARS, the
#         moment is finally there! Yay!
#
#   Revision 1.38  2014-08-07 10:15:46  jive_cc
#   HV: * Prevent NaN being present in the data breaking the plots
#
#   Revision 1.37  2014-05-21 18:12:59  jive_cc
#   HV: * Make failure to open PGPLOT device not fatal for the whole
#         plotting app ...
#
#   Revision 1.36  2014-05-15 20:58:21  jive_cc
#   *** empty log message ***
#
#   Revision 1.35  2014-05-15 09:56:09  jive_cc
#   HV: * Moar beautyfication
#
#   Revision 1.34  2014-05-15 08:27:27  jive_cc
#   HV: * Add 'new plot' and 'solint' settings to output of the 'pp' command
#         to display the full set of plot properties
#
#   Revision 1.33  2014-05-15 08:19:43  jive_cc
#   HV: * Beautified output of plot properties and selection / everything
#         now lines up nicely when printed using 'sl' or 'pp'
#
#   Revision 1.32  2014-05-14 17:35:14  jive_cc
#   HV: * if weight threshold is applied this is annotated in the plot
#       * the plotiterators have two implementations now, one with weight
#         thresholding and one without. Until I find a method that is
#         equally fast with/without weight masking
#
#   Revision 1.31  2014-05-14 17:02:01  jive_cc
#   HV: * Weight thresholding implemented - but maybe I'll double the code
#         to two different functions, one with weight thresholding and one
#         without because weight thresholding is sloooooow
#
#   Revision 1.30  2014-05-12 21:22:41  jive_cc
#   HV: * multiple plot environments now remember (and switch to) their current
#         working directory
#
#   Revision 1.29  2014-05-11 11:01:38  jive_cc
#   HV: * Initial support for multiple independant plots/windows/files; each
#         device has their own MS + selection + plot type. Not everything works
#         perfectly yet.
#
#   Revision 1.28  2014-05-06 14:32:12  jive_cc
#   HV: * Scan selection accepts a number of scan numbers and/or ranges and
#         rewrites it into a query
#
#   Revision 1.27  2014-05-06 14:20:39  jive_cc
#   HV: * Added marking capability
#
#   Revision 1.26  2014-04-25 15:15:46  jive_cc
#   HV: * support drawing of lines, points or both
#
#   Revision 1.25  2014-04-25 12:22:29  jive_cc
#   HV: * deal with lag data correctly (number of channels/number of lags)
#       * there was a big problem in the polarization labelling which is now fixed
#
#   Revision 1.24  2014-04-24 20:13:15  jive_cc
#   HV: * 'scan' and 'time' selection now use the expression parsers rather than
#         the regex-based implementation
#       * trigger use of persistent macros in CommanLine object
#
#   Revision 1.23  2014-04-23 14:17:46  jive_cc
#   HV: * Saving this old version which deals with "scans" - to be replaced with
#         something far more powerful
#
#   Revision 1.22  2014-04-15 12:23:11  jive_cc
#   HV: * deleted C++ makePlots method
#
#   Revision 1.21  2014-04-15 12:21:56  jive_cc
#   HV: * pagelabel is now centralized computed in the base class
#         we now have maximum consistency between all plots
#
#   Revision 1.20  2014-04-15 07:53:16  jive_cc
#   HV: * time averaging now supports 'solint' = None => average all data in
#         each time-range selection bin
#
#   Revision 1.19  2014-04-14 22:08:01  jive_cc
#   HV: * add support for accessing scan properties in time selection
#
#   Revision 1.18  2014-04-14 21:04:43  jive_cc
#   HV: * Information common to all plot- or data set labels is now stripped
#         and displayed in the plot heading i.s.o in the plot- or data set label
#
#   Revision 1.17  2014-04-14 14:46:05  jive_cc
#   HV: * Uses pycasa.so for table data access waiting for pyrap to be fixed
#       * added "indexr" + scan-based selection option
#
#   Revision 1.16  2014-04-10 21:14:40  jive_cc
#   HV: * I fell for the age-old Python trick where a default argument is
#         initialized statically - all data sets were integrating into the
#         the same arrays! Nice!
#       * Fixed other efficiency measures: with time averaging data already
#         IS in numarray so no conversion needs to be done
#       * more improvements
#
#   Revision 1.15  2014-04-08 22:41:11  jive_cc
#   HV: Finally! This might be release 0.1!
#       * python based plot iteration now has tolerable speed
#         (need to test on 8M row MS though)
#       * added quite a few plot types, simplified plotters
#         (plotiterators need a round of moving common functionality
#          into base class)
#       * added generic X/Y plotter
#
#   Revision 1.14  2014-04-02 17:55:30  jive_cc
#   HV: * another savegame, this time with basic plotiteration done in Python
#
#   Revision 1.13  2013-12-12 14:10:15  jive_cc
#   HV: * another savegame. Now going with pythonic based plotiterator,
#         built around ms2util.reducems
#
#   Revision 1.12  2013-09-03 17:34:30  jive_cc
#   HV: * Amazing! All plot types do work now. Can save plots to file.
#         Still some details left to do obviously ...
#
#   Revision 1.11  2013-08-20 18:23:50  jive_cc
#   HV: * Another savegame
#         Got plotting to work! We have stuff on Screen (tm)!
#         Including the fance standardplot labels.
#         Only three plot types supported yet but the bulk of the work should
#         have been done I think. Then again, there's still a ton of work
#         to do. But good progress!
#
#   Revision 1.10  2013-06-19 12:28:44  jive_cc
#   HV: * making another savegame
#
#   Revision 1.9  2013-03-31 17:17:56  jive_cc
#   HV: * another savegame
#
#   Revision 1.8  2013-03-11 18:23:39  jive_cc
#   HV: * freqsel now works properly (yay!)
#       * updated source selection to be consistent with baseline selection,
#         i.e.:
#            selector = expr | -expr   # -expr = remove from selection
#            expr = name | !name       # !name = negative match on name
#            name = regex
#
#   Revision 1.7  2013-03-10 14:33:36  jive_cc
#   HV: * freqsel working (first version)
#
#   Revision 1.6  2013-03-09 16:59:07  jive_cc
#   HV: * another savegame
#
#   Revision 1.5  2013-03-08 21:30:08  jive_cc
#   HV: * making a savegame
#
#   Revision 1.4  2013-02-19 16:53:29  jive_cc
#   HV: * About time to commit - make sure all edits are safeguarded.
#         Making good progress. baselineselection, sourceselection and
#         timeselection working
#
#   Revision 1.3  2013-02-11 09:40:33  jive_cc
#   HV: * saving work done so far
#         - almost all mapping-functionality is in place
#         - some of the API functions are starting to appear
#
#   Revision 1.2  2013-01-29 12:23:45  jive_cc
#   HV: * time to commit - added some more basic stuff
#
from   __future__ import print_function
from   six        import iteritems
import copy, re, math, operator, itertools, ppgplot, datetime, os, subprocess, numpy, imp, time
import pyrap.quanta, sys, pydoc, collections, functools
from   jiveplot.functional import compose, const, identity, map_, filter_, drap, range_, reduce, partial
from jiveplot import hvutil, plotiterator, parsers, jenums, selection, selection, ms2mappings, plots, ms2util, gencolors

if '-d' in sys.argv:
    print("PPGPLOT=",repr(ppgplot))

# Monkeypatch ppgplot if it don't seem to have pgqwin()
if not hasattr(ppgplot, 'pgqwin'):
    # the associative array from device id to window
    pg_windows = dict()
    # need to wrap pgswin to remember which window set for which device
    old_pgswin = ppgplot.pgswin
    def pgswin_new(w0, w1, w2, w3):
        pg_windows[ ppgplot.pgqid() ] = [w0, w1, w2, w3]
        return old_pgswin(w0, w1, w2, w3)
    def pgqwin():
        return pg_windows.get(ppgplot.pgqid(), None)
    ppgplot.pgswin = pgswin_new
    ppgplot.pgqwin = pgqwin


CP   = copy.deepcopy
NOW  = time.time
AVG  = jenums.Averaging
FLAG = jenums.Flagstuff

pwidth = lambda p, w: "{0:{1}}".format(p, w)
ppfx   = lambda p: pwidth(p, 10)
pplt   = lambda p: pwidth(p, 25)
prng   = lambda p: pwidth(p, 14)


class jplotter:
    def __init__(self, ms=None, **kwargs):
        # set construction defaults
        d = {'ms': ms, 'unique': True}
        # update with user input
        d.update( kwargs )

        # wether to show ALL meta data (selectables) 'unique==False')
        # or only show selectables that are actually used in the main
        # table of a MS ('unique==True')
        self.unique = d['unique']

        # Our data members
        self.reset()

        # finally, see if we were constructed with a ms argument
        if d['ms']:
            self.ms(d['ms'])

    def ms(self, *args):
        if args:
            rxYesNo = re.compile(r"^(?P<yes>t(rue)?)|f(alse)?$", re.I)
            # provide transformation(s) for options on the command line
            def as_bool(x):
                mo = rxYesNo.match(x)
                if mo:
                    return bool( mo.group('yes') )
                raise RuntimeError("{0} is not a valid boolean expression".format(x))
            (arguments, options) = hvutil.split_optarg(*args, **{'unique':as_bool, 'readflags':as_bool})
            if len(arguments)>1:
                raise RuntimeError("ms() takes only one or no parameters")
            # attempt to open the measurement set
            try:
                # Update our defaults with what the user specified on the command line
                self.mappings.load(arguments[0], **dict({'unique': self.unique}, **options))
                self.msname    = arguments[0]
                self.scanlist  = []
                self.dirty     = True
                self.selection = selection.selection()
                self.readFlags = options.get('readflags', True)
            except Exception as E:
                print(E)
        if self.msname:
            m = self.mappings
            print("ms: Current MS is '{0}' containing {1} rows of {2} data for {3} [{4}]".format( \
                    self.msname, m.numRows, m.domain.domain, m.project, m.domain.column))
        else:
            print("ms: no MS opened yet")
        return True

    def uniqueMetaData(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("This command supports only one argument")
            if args[0] in "tT":
                self.unique = True
            elif args[0] in "fF":
                self.unique = False
            else:
                self.unique = bool(int(args[0]))
        print("unique meta data: ",self.unique)

    def haveMS(self):
        return self.msname

    def listScans(self, *args):
        pfx = "listScans:"
        if not self.msname:
            print(pfx,"No MS loaded yet")
            return None
        if not self.scanlist:
            print(pfx, "No scans (yet) - have you run the 'indexr' task?")
            return None
        # print list of scan id's + text
        lines = "\n".join( map(str, self.scanlist) )
        if args and '-p' in args:
            pydoc.pager( lines )
        else:
            print(lines)

    def listBaselines(self):
        pfx = "listBaselines:"
        if not self.msname:
            print(pfx,"No MS loaded yet")
            return None
        mk_output(prng(pfx), self.mappings.baselineMap.baselineNames(), 80)

    def listFreqs(self):
        if not self.msname:
            print("listFreqs: No MS loaded yet")
            return
        pmap   = self.mappings.polarizationMap
        smap   = self.mappings.spectralMap
        pfx    = 'listFreqs:'
        # deal with the frequency setups
        for frqid in smap.freqIds():
            print("{0} FREQID={1} [{2}]".format(prng(pfx), frqid, smap.freqGroupName(frqid)))
            # Loop over the subbands
            for (sbidx, sb) in smap.subbandsOfFREQ(frqid):
                print("{0}   SB{1:2d}:".format(prng(pfx), sbidx),sb.__str__(polarizationMap=self.mappings.polarizationMap))

    def freqIDs(self):
        if not self.msname:
            return []
        else:
            return self.spectralMap.freqIds()

    def nSubband(self, frqid):
        if not self.msname:
            return 0
        else:
            return self.spectralMap.spectralWindows(frqid)

    def nPolarizations(self, fid):
        if not self.msname:
            return 0
        else:
            return len(self.polarizationMap.getCorrelationTypes(fid))

    def Polarizations(self, pid):
        if not self.msname:
            return []
        else:
            return self.polarizationMap.getPolarizations(pid)

    def listSources(self):
        pfx = "listSources:"
        if not self.msname:
            print(pfx, "No MS loaded yet")
            return None
        mk_output(prng(pfx+" "), self.mappings.fieldMap.getFields(), 80)

    def listAntennas(self):
        pfx = "listAntennas:"
        if not self.msname:
            print(pfx,"No MS loaded yet")
            return None
        # baselineMap.antennas() = [ (id, name), ... ]
        mk_output(prng(pfx+" "), map_("{0[0]} ({0[1]: >2}) ".format,
                                 self.mappings.baselineMap.antennas()), 80)

    def listTimeRange(self):
        pfx = "listTimeRange:"
        if not self.msname:
            print(pfx,"No MS loaded yet")
            return None
        # call upon the mighty casa quanta to produce
        # human readable format
        tr      = self.mappings.timeRange
        print(prng(pfx),ms2util.as_time(tr.start),"->",ms2util.as_time(tr.end)," dT:"," ".join(map("{0:.3f}".format, tr.inttm)))

    def dataDomain(self):
        return self.mappings.domain.domain

    def indexr(self):
        pfx = "indexr:"
        if not self.msname:
            print(pfx,"No MS loaded yet")
            return []
        # need to remember old ms name
        lmsvar = 'indexr_lastms'
        if not hasattr(self, lmsvar):
            setattr(self, lmsvar, None)

        if not self.scanlist or self.indexr_lastms!=self.msname:
            # need to recompute scan list.
            # From indexr() we only get field_ids in the scan object
            # so we immediately transform them into field names
            print("Running indexr. This may take some time.")
            unmapFLD = self.mappings.fieldMap.field
            def unmapFLDfn(x):
                x.field = unmapFLD(x.field_id)
                return x
            self.scanlist = map_(unmapFLDfn, ms2util.indexr(self.msname))
            # and store the current ms
            self.indexr_lastms = CP(self.msname)
        print("indexr: found ",len(self.scanlist)," scans. (use 'listr' to inspect)")
        return self.scanlist

    ## display or select time-range + source via scan
    _scanTaQL = "(TIME>={0.start:.7f} AND TIME<={0.end:.7f} AND FIELD_ID={0.field_id} AND ARRAY_ID={0.array_id} AND SCAN_NUMBER={0.scan_number})".format

    def scans(self, *args):
        pfx = "scan:"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")
        # shorthand to selection object
        sel_  = self.selection
        map_  = self.mappings
        # were we called with argument(s)?
        if args:
            if not self.scanlist:
                return errf("No scanlist yet - please run 'indexr' first")

            oldscanSel = CP(sel_.scanSel)

            # if args[0] looks like a range of scan numbers, then rewrite the argument
            qry      = args[0]
            rxIrange = re.compile(r"^\d+(-\d+(:\d+)?)?(\s+\d+(-\d+(:\d+)?)?)*$")
            if rxIrange.match(qry):
                qry = "start to end where scan_number in [{0}]".format(",".join(qry.split()))

            # Get the modifier function and scan-selection function
            tr = map_.timeRange
            (mod_f, filter_f) = parsers.parse_scan(qry, start=tr.start, end=tr.end, length=tr.end-tr.start, mid=(tr.start+tr.end)/2.0)

            # We create copies of the selected scans and modify them (such that the original definitions
            # remain unaltered). We mainly copy all attributes for display purposes. Behind the scenes it's
            # just a time range selection but that doesn't read too well on the screen
            def copy_and_modify(acc, scan):
                nScan = CP(scan)
                (nStart, nEnd) = mod_f(scan)
                nScan.start = nStart
                nScan.end   = nEnd
                acc.append(nScan)
                return acc
            # Only override scan selection if any scans were selected
            scanSel = reduce(copy_and_modify, filter_f(self.scanlist), [])
            if not scanSel:
                return errf("Your selection criteria did not match any scans")
            sel_.scanSel = scanSel
            self.dirty   = self.dirty or (sel_.scanSel!=oldscanSel)

            # Construct the TaQL from the scan selection
            if self.dirty:
                # clear out time/source selection
                sel_.sources       = []
                sel_.timeRange     = []
                sel_.sourcesTaql   = None
                # replace time range taql only if there are any scans selected
                scanTaQL           = " OR ".join(map(jplotter._scanTaQL, sel_.scanSel))
                sel_.timeRangeTaql = "("+scanTaQL+")" if scanTaQL else None

                # and we must construct the timerange list.
                # also compress the time ranges; if there are overlapping regions, join them
                def reductor(acc, xxx_todo_changeme4):
                    # if current start time <= previous end-time we has an overlaps
                    (s, e) = xxx_todo_changeme4
                    if s<= acc[-1][1]:
                        acc[-1] = (acc[-1][0], e)
                    else:
                        acc.append( (s,e) )
                    return acc
                # transform selected scans into list of (start,end) tuples and sort them ascending by start time
                sel_.timeRange = sorted(map(operator.attrgetter('start', 'end'), sel_.scanSel), key=operator.itemgetter(0))
                # Now compress them
                if sel_.timeRange:
                    sel_.timeRange = reduce(reductor, sel_.timeRange[1:], [sel_.timeRange[0]])
                
        # Display selected scans
        if sel_.scanSel:
            print("\n".join(map(str, sel_.scanSel)))
        else:
            print("No scans selected{0}".format( "" if self.scanlist else " (and could not have; 'indexr' must be run first)" ))

    ##
    ## Here come interesting API methods
    ##

    ## channels()    => show current selection of channels
    ## channels(str) => select new set of channels"
    def channels(self, *args):
        pfx  = "channels:"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")

        # either no args or a single string
        if args:
            sel_   = self.selection
            spmap_ = self.mappings.spectralMap

            # ah, let's see what the usr gave us.
            # let's see if there's a freqid selected.
            # If there's only one freqid in the MS we default to
            # that one. se scream if we're using a relative
            # selection like 'mid', 'all', 'first', 'last'
            if args[0]=="none":
                sel_.chanSel = None
            else:
                # get the current datadescription id's and verify the
                # number of channels is equal for all
                nchan = None

                if sel_.ddSelection:
                    # ddSelection is [(fq, sb, pol, [idx, ..]), ... ]
                    nch = set(map(lambda f_s_p_l: spmap_.numchanOfFREQ_SB(f_s_p_l[0], f_s_p_l[1]), sel_.ddSelection))
                    if len(nch)==1:
                        nchan = list(nch)[0]
                else:
                    # no freq selection yet, see if all the spectral windows in all the 
                    # freqids have the same number of channels. If they do we can easily
                    # use that "nch" as symbolic value
                    chset = reduce( \
                        lambda acc, fid: acc.update(map(spmap_.numchanOfSPW, spmap_.spectralWindows(fid))) or acc,\
                        spmap_.freqIds(), \
                        set())
                    if len(chset)==1:
                        # I don't know how else to extract the single element out of a set
                        [nchan] = list(chset)
                # Rite. We may have a value for nch!
                #  Scream loudly if one of the 'relative' channels (other than first)
                #  is mentioned in the argument and we didn't find an actual value
                found = lambda x : args[0].find(x)!=-1
                if any([found(x) for x in ["mid","last"]]) and not nchan:
                    return errf("Your selection contains relative channels (mid, last) but no value for them could be deduced")

                # Now we know we can safely replace strings -> values (and remove whitespace)
                if nchan:
                    replacer = lambda x : hvutil.sub(x, [("\s+", ""), (re.compile(r"all"), "first:last"), \
                                                         ("first", "0"), ("mid", repr(nchan/2)), ("last", repr(nchan-1))])
                else:
                    replacer = lambda x : hvutil.sub(x, [("\s+", "")])
                expander = lambda x : hvutil.expand_string_range(replacer(x))

                # Reduce and build the actual channelselection - we gather all the sub-selections (the comma separated
                # entries) into a set [such that duplicates are automatically handled] and transform it into a list
                # for subsequent use, which expects the chanSel to be a list
                sel_.chanSel = list(reduce(lambda acc, x: acc.update(expander(x)) or acc, args, set()))
            self.dirty   = True

        # print the selection
        cstr  = "No channels selected yet"
        if self.selection.chanSel:
            cstr = hvutil.range_repr(hvutil.find_consecutive_ranges(self.selection.chanSel))
        print("{0} {1}".format(ppfx(pfx), cstr))

    ##
    ## Print or set the source selection
    ##
    def sources(self, *args):
        pfx  = "sources:"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")

        sel_ = self.selection
        map_ = self.mappings

        if args:
            if "none" in args:
                sel_.sources     = None
                sel_.sourcesTaql = None
            else:
                #   sources = select | -select
                #   select  = expr | !expr
                #   expr    = shell like regex, meaning
                #              "*" for wildcard [0 or more characters]
                #              "?" for one character
                #
                # take [(sourcename, flag),..] and [selectors(),...]
                #  and map each selector over that list of (source,flag) tuples.
                #  the selector function will return a new list of (source,flag) tuples
                #  with the flags updated according to its selection result.
                #  Return the list of sources for which the flag is true
                #
                selector = lambda tuplst, selectors: \
                        [src for (src, flag) in reduce(lambda acc, sel: map(sel, acc), selectors, tuplst) if flag]

                def mkselector(src):
                    srcs   = copy.deepcopy(src)
                    # 0) Check if it was an expression with an explicit add/subtract
                    add = True
                    mo  = re.match("^\s*(?P<add>[-+])(?P<expr>.*)", srcs)
                    if mo:
                        add  = (mo.group('add')=='+')
                        srcs = mo.group('expr')

                    # a) remove whitespace and replace multiple wildcards by a single one
                    srcs   = hvutil.sub(srcs, [("\s+", ""), ("\*+", "*"), ("^all$", "*")])

                    # b) was a negation given? If so, strip the negation character but remember it
                    neg    = re.match("^!", srcs)
                    if neg:
                        srcs = srcs[1:]

                    # c) escape re special characters: "+" => "\+", "." => "\."
                    #    so they won't be interpreted *as* regex characters - they're
                    #    wont to appear in source names
                    #    Also replace the question mark by "." - do this _after_ the 
                    #    literal dots have been escaped ...
                    srcs = hvutil.sub(srcs, [("\+", "\+"), ("\.", "\."), ("\?",".")])

                    # d) replace wildcards (*) by (.*) 
                    srcs = hvutil.sub(srcs, [("\*", ".*")])

                    # Now we can make a regex
                    rx = re.compile("^"+srcs+"$", re.I)

                    # and select only the matching sources
                    return lambda src_flag: (src_flag[0], (src_flag[1] if neg else add) if rx.match(src) else (add if neg else src_flag[1]))

                # now build the list of selectors, based on comma-separated source selections 
                # and run all of them against the sourcelist
                sel_.sources = selector([(x, False) for x in map_.fieldMap.getFields()], map(mkselector, args))

                # set the TaQL accordingly - if necessary
                if len(sel_.sources)==len(map_.fieldMap.getFields()):
                    # all sources selected - no need for TaQL
                    sel_.sourcesTaql = ""
                else:
                    sel_.sourcesTaql = "(FIELD_ID IN {0})".format( map(map_.fieldMap.unfield, sel_.sources) )
            self.dirty   = True
            # new source selection overrides scan selection
            sel_.scanSel = []

        sstr = ["No sources selected yet"]
        if sel_.sources:
            sstr = sel_.sources
        mk_output(ppfx(pfx), sstr, 80)

    ##
    ## print/edit the baseline selection
    ##
    def baselines(self, *args):
        pfx  = "baselines:"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")

        sel_   = self.selection
        blmap_ = self.mappings
        if args:
            # remember current TaQL for the baselines so we can detect if the 
            # selection changed
            oldTaql = copy.deepcopy(sel_.baselinesTaql)
            
            # regex 'code' for the specials 'auto' and 'cross'
            #  so we can reuse them
            auto   = r"^(\w+)(\1)$"
            cross  = r"^(((.{2})(?!\3)(.{2}))|((.{3})(?!\6)(.{3})))$"
            simple = len(args)==1

            # process settings of just "auto" and "cross" independantly because then
            # we can really easily set the taql and selection
            if args[0]=="none" and simple:
                sel_.baselines     = None
                sel_.baselinesTaql = ''
            elif args[0]=="all" and simple:
                sel_.baselines     = copy.deepcopy(blmap_.baselineMap.baselineNames())
                sel_.baselinesTaql = ''
            elif args[0]=="auto" and simple:
                sel_.baselines     = filter_(lambda x: re.match(auto, x), blmap_.baselineMap.baselineNames())
                sel_.baselinesTaql = "(ANTENNA1==ANTENNA2)"
            elif args[0]=="cross" and simple:
                sel_.baselines     = filter_(lambda x: not re.match(auto, x), blmap_.baselineMap.baselineNames())
                sel_.baselinesTaql = "(ANTENNA1!=ANTENNA2)"
            else:
                # have to bloody do them all!
                # what do we support?
                #   aliases:  'all' 'auto' 'cross'
                #   literals: [:alpha:]+    (no wildcards or ()'s)
                #   regex like:  
                #       baseline = name* | *name | (name)name | name(name) | (name)(name)
                #       name     = namestr | name "|" name
                #       namestr  = [a-zA-Z0-9]+
                filterer = lambda tuplst, filters: \
                        [bl for (bl, flag) in reduce(lambda acc, filt: map(filt, acc), filters, tuplst) if flag]

                # pre-compile the used regexes
                rxp0  = re.compile(r"[\(\)]")
                rxp1  = re.compile(r"^(\([^\)]+\))([^(]+)$")
                rxp2  = re.compile(r"^(\([^\)]+\))(\([^\)]+\))$")
                rxp3  = re.compile(r"^([^\(]+)(\([^\)]+\))$")
                rxwc0 = re.compile(r"\*")
                # note: grouping here has an effect on reversing the baselines below!
                rxwc1 = re.compile(r"^\^?(\*|(\*)(\w+)|(\w+)(\*$))\$?$") 
                subst =  [(re.compile(x), y) for (x,y) in \
                                [("\s+", ""), ("\*+", "*"), ("^all$", "*"), \
                                  ("^auto$", r"^(\w+)(\\1)$"), \
                                  ] \
                                  #("^cross$", r"^(((.{2})(?!\\3)(.{2}))|((.{3})(?!\\6)(.{3})))$") \
                                ]

                def mkfilter(bl):
                    #print "mkfilter === ",bl
                    org = copy.deepcopy(bl)
                    # Before any extra processing, replace the word 'cross' by '!auto'
                    bls = re.sub(r"cross", r"!auto", copy.deepcopy(bl))
                    #print "bls=",bls
                    add = True
                    # a) was an explicit add/subtract given?
                    mo  = re.match("^\s*(?P<add>[-+])(?P<expr>.*)", bls)
                    if mo:
                        add = (mo.group('add')=='+')
                        bls = mo.group('expr')
                    # b) was a negation given? If so, strip the negation character but remember it
                    neg    = re.match(r"^\s*(?P<neg>!+)(?P<expr>.+)$", bls)
                    if neg:
                        bls = neg.group('expr')
                        neg = (len(neg.group('neg')) % 2)==1 # odd number of negations
                    #print neg,bls
                    # a) remove whitespace and replace multiple wildcards by a single one
                    #    and substitute aliases
                    bls   = hvutil.sub(bls, subst)
                    #print bls

                    # see if we have a parenthesized expression
                    # p0 => any parenthesis at all? if so either
                    #       p1, p2 or p3 must match otherwise syntax error
                    p0 = rxp0.search(bls)
                    p1 = rxp1.match(bls)
                    p2 = rxp2.match(bls)
                    p3 = rxp3.match(bls)
                    #def p(x):
                    #    print x
                    #map(lambda (x,y): p("p{0}:{1}".format(x,y)), zip(range(4), [p0,p1,p2,p3]))

                    if not (bls==auto) and not (bls==cross) and p0 and not (p1 or p2 or p3):
                        raise SyntaxError("the parenthesised baseline expression {0} is invalid".format(bl))

                    # Do we have "chars*" or "*chars"?
                    # note: we only really need to look at these if
                    #       no parenthesisation was found: if the baseline
                    #       expression was properly parenthesised the
                    #       wildcard stuff is irrelevant; we can already
                    #       properly reverse the expression
                    wc0 = rxwc0.search(bls)
                    #print bls,wc0
                    # if no wildcard we add one at the end
                    if not ((bls==auto) or (bls==cross) or wc0):
                        bls = bls+"*"
                        wc0 = rxwc0.search(bls)
                    #print bls,wc0
                    wc1 = rxwc1.match(bls)
                    #print bls,wc0,wc1

                    # If we are (1) parenthesized or (2) have a wildcard + non-wildcard
                    # and (3) are NOT equal to the 'auto' definition
                    # we must 'reverse' the expression since sometimes baselines are mentioned
                    # B->A rather than A->B
                    revbls = None
                    
                    if not (bls==auto) and not (bls==cross):
                        # no parenthesis but wildcards?
                        if not p0 and wc0:
                            # then it'd better be a valid wildcardexpression
                            if not wc1:
                                raise SyntaxError("the wildcarded baseline expression {0} is invalid".format(bl))
                            # now only need to reverse if it was "*chars" or "chars*"
                            fstgrp = 4 if wc1.group(4) else 2
                            revbls = "{1}{0}".format(wc1.group(fstgrp), wc1.group(fstgrp+1))
                        # Ok parenthesized expression. find out which one matched
                        elif p0:
                            if p1:
                                mg = p1
                            elif p2:
                                mg = p2
                            elif p3:
                                mg = p3
                            revbls = "{1}{0}".format(mg.group(1), mg.group(2))

                    # d) replace wildcards (*) by (.*) 
                    bls = hvutil.sub(bls, [("\*", ".*")])
                    if revbls:
                        revbls = hvutil.sub(revbls, [("\*", ".*")])
                    #print "compiling: bls={0}  revbls={1}".format(bls, revbls)

                    # e) now we can make a regex
                    rx  = re.compile(bls, re.I)
                    rxr = rx
                    if revbls:
                        rxr = re.compile(revbls, re.I)

                    # and filter only the matching sources
                    def f( bl_flag ):
                        (bl,flag) = bl_flag
                        # only perform reverse match if it is a (physically)
                        # different rx object
                        m  = rx.match(bl)
                        mr = rxr.match(bl) if rxr is not rx else False
                        return (bl, (flag if neg else add) if (m or mr) else (add if neg else flag))
                    return f
                # now build the list of filters, based on comma-separated source selections 
                # and run all of them against the sourcelist
                sel_.baselines  = filterer([(x, False) for x in blmap_.baselineMap.baselineNames()], map(mkfilter, args))

                if not sel_.baselines:
                    print("No baselines matched your selection")
                    sel_.baselines     = None
                    sel_.baselinesTaql = ''
                else:
                    # Now that we've selected the baselines, we must generate the TaQL to go with it
                    sel_.baselinesTaql = "((1000*ANTENNA1+ANTENNA2) IN {0})".format( \
                        map_(lambda x_y: 1000*x_y[0] + x_y[1], \
                            map(blmap_.baselineMap.baselineIndex, sel_.baselines)))
            self.dirty = self.dirty or (oldTaql != sel_.baselinesTaql)
        blstr = ["No baselines selected yet"]
        if sel_.baselines:
            blstr = sel_.baselines
        mk_output(ppfx(pfx), blstr, 80)

    ##
    ## print/set the timerange(s)
    ##
    def times(self, *args):
        pfx  = "time:"
        errf = hvutil.mkerrf("{0} ".format(pfx, str(self.msname)))
        if not self.msname:
            return errf("No MS loaded yet")
        if len(args)>1:
            raise RuntimeError("Please call this method with zero or one argument")

        sel_ = self.selection

        if args:
            if "none" in args:
                sel_.timeRange     = None
                sel_.timeRangeTaql = None
            else:
                tr = self.mappings.timeRange
                # Parse the time ranges
                l  = parsers.parse_time_expr(args[0], start=tr.start, end=tr.end, length=tr.end-tr.start, \
                                                      mid=(tr.start+tr.end)/2.0, t_int=tr.inttm[0])

                # Sort by start time & compress; overlapping time ranges get merged
                sel_.timeRange = sorted(l, key=operator.itemgetter(0))

                def reductor(acc, s_e):
                    # if current start time <= previous end-time we has an overlaps
                    if s_e[0]<= acc[-1][1]:
                        acc[-1] = (acc[-1][0], s_e[1])
                    else:
                        acc.append( s_e )
                    return acc
                if sel_.timeRange:
                    sel_.selectTimeRange( reduce(reductor, sel_.timeRange[1:], [sel_.timeRange[0]]) )

            self.dirty   = True
            sel_.scanSel = []

        # and show the current timeselection
        if sel_.timeRange:
            mk_output(ppfx(pfx),  map(lambda s_e: "{0} -> {1}".format(ms2util.as_time(s_e[0]), ms2util.as_time(s_e[1])), sel_.timeRange), 80)
        else:
            print("{0} No timerange(s) selected yet".format(ppfx(pfx)))

    ##
    ## Support frequency selection
    ##
    def freqsel(self, *args):
        pfx  = "freqsel:"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")

        sel_  = self.selection
        spMap = self.mappings.spectralMap
        pMap  = self.mappings.polarizationMap 

        #  [<freqidsel>/]<subbandselection>[/<polselection>]
        #
        #  Where we will accept the following for the separate entries:
        #
        #  freqidsel:    * OR 
        #				ranges (eg: '1:5[:2]') AND/OR separate entries (commaseparated)
        #
        #               Note: freqidsel may be omitted if and only if there is
        #                     only one freqid!
        #
        #  subbandselection: same as freqidsel
        #
        #  polselection: It is optional. If no polsel, the first polid is used and
        #                all combis are used.
        #                Syntax:
        #                    [polid:]pols
        #                polid = if desired, the polarization selection
        #                        can be prefixed with a specific polarization
        #                        ID to draw the selected polarizations from.
        #                        If not specified, the first polarization id
        #                        that yields positive match(es) on _all_
        #                        specified pols is used
        #                pols = comma separated list of:
        #                    [rl]{2}, *[lr], [lr]*, X, P, A
        #                       X = select crosspols
        #                       P = select parallel polarizations
        #                or just 
        #                    *
        #
        #  Examples:
        #   */P  selects all parallel hand polarization combinations from
        #        all subbands from the default freqid. Only allowed iff
        #        there is only one freqid! Note: if the subband(s) are
        #        correlated with multiple polarization ids then the first
        #        polarization id with parallel polarizations will be 
        #        selected.
        #   */*  selects all polarizations from the first polarization ID
        #        for all subbands. This is allowed only iff there is one
        #        freqid!
        #   */1:P selects all parallel hand polarization combinations from
        #         polarization id #1 for all subbands for the default freqid.
        #         Only iff there is only one freqid. Each subband must
        #         be (at least) correlated with polarization id #1
        #   1,3,5/2:X selects the crosspols from polarization ID 2 for
        #             subbands 1, 3 and 5. Also only allowed iff there is
        #             only one freqid
        #   2/0:3/r* selects polarizations RR and/or RL (if they exist)
        #            from the first polarization ID that has either or both
        #            for subbands 0,1,2,3 from freqid #2
        #   0/1:4/0:rr,ll explicitly select (require) from polarization ID #0
        #                 RR and LL for subbands 1,2,3,4 from freqid #0
        #
        # We need to end up with a list of
        #    [(DATA_DESC_ID, [CORR_TYPE,...]), ...]
        # or, in English:
        #    for each DATA_DESCRIPTION_ID ("DDId") we select we must record which products
        #    out of the polarization products associated with that DDId are selected.

        # regexes
        #rxPol = re.compile(r"^((?P<polid>\d+):)?(?P<pexpr>\*|(([RrlL]{2}|\*[RrLl]|[RrLl]\*|[XP])(,([RrLl]{2}|\*[RrLl]|[RrLl]\*|[XP]))*))$")
        rxPol = re.compile(r"^((?P<polid>\d+):)?(?P<pexpr>\*|(([rl]{2}|\*[rl]|[rl]\*|[XP])(,([rl]{2}|\*[rl]|[rl]\*|[XP]))*))$", re.I)
        rxRng = re.compile(r"^\*|(\d+|\d+:\d+)(,(\d+|\d+:\d+))*$")
        if args:
            #if re.match("^\s*$", args[0]):
            #    raise RuntimeError, "Empty frequency selection string?"

            if "none" in args:
                sel_.ddSelection     = None
                sel_.ddSelectionTaql = None
            else:
                # eventually we'll end up with a list of selected datadescription IDs
                # each selection entry will add the DDids it selects to the current set
                def mkselector(arg):
                    # the three potential bits in the selection
                    fqs = sbs = pols = None

                    # first things first: split at the slashes.
                    # depening on how many we find we know what the parts mean
                    parts = arg.split("/")
                    if len(parts)==0 or len(parts)>3:
                        raise SyntaxError("The frequency selection '{0}' is invalid".format(arg))

                    if len(parts)==3:
                        [fqs, sbs, pols] = parts
                    elif len(parts)==2:
                        # could be fq/sb or sb/pol
                        # look at the last bit: if it matches a polarization
                        # expression than we assume it's that interpretation.
                        # This is only acceptable if only one FQ.
                        if rxPol.match(parts[1]):
                            # assert only one FQ group
                            if spMap.nFreqId()!=1:
                                raise RuntimeError("There are multiple freqids so the selector '{0}' is ambiguous".format(arg))
                            [sbs, pols] = parts
                            fqs         = repr(spMap.freqIds()[0])
                        else:
                            [fqs, sbs] = parts
                    elif len(parts)==1:
                        # only subbands. Also only acceptable if only one freqid
                        if spMap.nFreqId()!=1:
                            raise RuntimeError("There are multiple freqids so the selector '{0}' is ambiguous".format(arg))
                        sbs = parts[0]
                        fqs = repr(spMap.freqIds()[0])

                    #print "==> fq,sb,p = ",fqs,sbs,pols

                    # expand the selection strings to numerical values
                    if fqs:
                        if not rxRng.match(fqs):
                            raise SyntaxError("Invalid freqgroup selection in '{0}'".format(arg))
                        if fqs=="*":
                            fqs = spMap.freqIds()
                        else:
                            fqs = hvutil.expand_string_range(fqs)
                    if sbs:
                        if not rxRng.match(sbs):
                            raise SyntaxError("Invalid subband selection in '{0}'".format(arg))
                        if sbs != "*":
                            sbs = hvutil.expand_string_range(sbs)
                    if pols:
                        polmatch = rxPol.match(pols)
                        if not polmatch:
                            raise SyntaxError("Invalid polarization selection in '{0}'".format(arg))
                        # before starting to compile the polarization regexen, we
                        # must first extract the, potential, leading polid
                        pid  = polmatch.group('polid')
                        if pid:
                            pid = int(pid)
                        pols = polmatch.group('pexpr')

                        # pols is a (potentially) comma-separated list of regexes
                        # (or at least, that's how we treat them).
                        # Substitute aliases with their corresponding regexes and compile
                        # them into a list of regexes
                        pols = map_(lambda x:re.compile("^"+x+"$", re.I), \
                                 map(lambda x: \
                                     hvutil.sub(x, [("\*", ".*"), (re.compile("P", re.I), r"(.)\\1"), \
                                                    (re.compile("X", re.I), r"(.)(?!\\1).")]), \
                                     pols.split(",")))
                        # and hoopla!
                        pols = (pid, pols)
                    else:
                        # no explicit polarization selection, select
                        # everything from the first polid we encounter
                        pols = (None, [re.compile(r"^.+$")] )

                    # Excellent! Now we can write a function that decides if the selection 
                    # can be honoure
                    #
                    # acc should change.
                    # it now is [ (data_desc_id, [prod_idx,..]) ]
                    # but inside the loop we have
                    #   FQ, SB, POLID (thus also DATA_DESC_ID) and [prod_idx,..]
                    # better to keep all four bits of info in a simple
                    # object [makes it easier later on]
                    def selector(acc):
                        # keep track of how many entries this selector added
                        # if none we should yell out!
                        olen = len(acc)
                        # Construct a polarization id filter (someone
                        # could have explicitly selected an ID).
                        # This needs to be done only once for each selector.  Also
                        # extract the regexes for the polarization matching
                        (polid, rxs) = pols
                        idfilter = lambda x: True if polid is None else x==polid

                        # The adder function checks if ALL regexes in the list 
                        # yielded a result [if not, the POLID under consideration
                        # could not honour the requested Polarizations].
                        # Then, because each match could have matched >1 polarization
                        # (think 'r*') we must add the individual matches (hence the
                        # double map())
                        def adder(x):
                            if len(x)==len(rxs):
                                # these map()'s are done for their sideeffect
                                # so it's safe to use "drap()" - and a single
                                # call to that is enough to drain all iterables
                                drap(lambda z: drap(lambda y: acc.append(y), z), x)

                        for fq in fqs:
                            # Dynamically expand the "all subbands for this freqid"
                            # (if selected)
                            if sbs=="*":
                                subbands = range_(spMap.nSubbandOfFREQ(fq))
                            else:
                                subbands = sbs
                            for sb in subbands:
                                # inner mapping function: map the regexen over the
                                # polarization products to see which match
                                def dopolid(pid):
                                    # extract the list of numbered polarization products
                                    plist = [x for x in enumerate(pMap.getPolarizations(pid))]
                                    # for every rx, compile a list of matching indices
                                    # and do a double filter:
                                    #   filter out the entries which didn't match
                                    #   and filter out the rx's that didn't have any
                                    #   matches at all
                                    return \
                                      filter_(operator.truth, \
                                        map(lambda rx: \
                                          map(operator.itemgetter(0), \
                                            filter(operator.itemgetter(1), \
                                              map(lambda i_p: \
                                                    ((fq,sb,pid,i_p[0]), rx.match(i_p[1])), plist))), rxs))
                                    return rv
                                # Now run the pol id processor over all feasable polids
                                # and add the succesfull polid(s), if anything
                                drap(adder, \
                                     map(lambda x: dopolid(x), \
                                         filter(idfilter, spMap.polarizationIdsOfFREQ_SB(fq, sb))))
                        if len(acc)==olen:
                            raise RuntimeError("the selection '{0}' did not match anything".format(arg))
                        return acc
                    return selector 

                # Loop over all selections and for each selection build the
                # selector function, execute the selector function and finally
                # tally up the total selection
                sel_.ddSelection = \
                    reduce(lambda acc, sel: sel(acc), map(mkselector, args), [])

                # Now there's some normalization to be done: it could be that
                # different expressions selected different products out of the same
                # fq/sb/polid combination so we want to group those together. Dict +
                # set to the rescue!  
                # After the reduction we have:
                #   ddSelection = [ (fq, sb, polid, [prods]), .... ]
                def reductor(acc, f_s_p_prod):
                    (f, s, p, prod) = f_s_p_prod
                    key = (f,s,p)
                    acc[key].add(prod) if key in acc else acc.update({key:set([prod])})
                    return acc
                # fsp_v == ((f,s,p), v)
                sel_.ddSelection = \
                        map_(lambda fsp_v: (fsp_v[0][0], fsp_v[0][1], fsp_v[0][2], list(fsp_v[1])), \
                             iteritems(reduce(reductor, sel_.ddSelection, {})))

                # Must update the TaQL to select the current DDIds
                # fspl = (f, s, p, l)
                sel_.ddSelectionTaql = "(DATA_DESC_ID in {0})".format( \
                        map_(lambda fspl: spMap.datadescriptionIdOfFREQ_SB_POL(fspl[0], fspl[1], fspl[2]), \
                             sel_.ddSelection))
                #print "ddSelectionTaql:",sel_.ddSelectionTaql

            self.dirty = True

        # Print the Human Readable Format
        print("\n".join( \
                map(lambda x: ppfx(pfx)+" "+x, \
                    self.selection.ddSelectionHRF(self.mappings.polarizationMap))))

    #####
    #####  Next category of functions
    #####
    def weightThreshold(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("weightThreshold() takes only one or no parameters")
            if args[0].lower()=="none":
                self.selection.weightThreshold = None
            else:
                try:
                    self.selection.weightThreshold = float(args[0])
                except:
                    raise RuntimeError("'{0}' is not a valid weight threshold (float)".format(args[0]))
            self.dirty = True
        print("{0} {1}".format(pplt("weightThreshold:"), self.selection.weightThreshold))

    # in Py3 we can't have an attribute by the name of None apparently;
    # so jenums.Averaging.None yields SyntaxError
    # But we don't want to disrupt the interface to the user too much
    # so "None" must be mapped to "NoAveraging"
    _avgMapFn = { 'None': const('NoAveraging') }
    def averageTime(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("averageTime() takes only one or no parameters")
            # 'scalar' => 'Scalar', 'vector' => 'Vector', 'none' => 'NoAveraging'
            s = args[0].capitalize()
            s = jplotter._avgMapFn.get(s, identity)(s)
            if s not in AVG:
                raise RuntimeError("'{0}' is not a valid time averaging method".format(args[0]))
            self.selection.averageTime = AVG[s]
            self.dirty = True
        print("{0} {1}".format(pplt("averageTime:"),self.selection.averageTime))

    def averageChannel(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("averageChannel() takes only one or no parameters")
            # 'scalar' => 'Scalar', 'vector' => 'Vector', 'none' => 'None'
            s = args[0].capitalize()
            s = jplotter._avgMapFn.get(s, identity)(s)
            if s not in AVG:
                raise RuntimeError("'{0}' is not a valid channel averaging method".format( args[0] ))
            self.selection.averageChannel = AVG[s]
            self.dirty = True
        print("{0} {1}".format(pplt("averageChannel:"),self.selection.averageChannel))

    def solint(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("solint() takes only one or no parameters")
            if args[0].lower()=="none":
                self.selection.solint = None
            else:
                try:
                    self.selection.solint = parsers.parse_duration(args[0])
                except:
                    raise RuntimeError("'{0}' is not a valid solution interval".format( args[0] ))
            self.dirty = True
        print("{0} {1}{2}".format(pplt("solint:"), self.selection.solint, "" if self.selection.solint is None else "s"))

    def nchav(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("nchav() takes only one or no parameters")
            if args[0].lower()=="none":
                self.selection.solchan = None
            else:
                try:
                    tmp = int(args[0])
                    assert tmp > 1, "Invalid nchav value - cannot bin by less than one channel"
                    self.selection.solchan  = tmp
                except Exception as E:
                    raise RuntimeError("'{0}' is not a valid channel averaging number ({1})".format( args[0], str(E) ))
            self.dirty = True
        print("{0} {1}".format(pplt("nchav:"), self.selection.solchan))

    def getNewPlot(self):
        return copy.deepcopy(self.selection.newPlot)

    def newPlotChanged(self):
        return self.npmodify>0

    def newPlot(self, *args):
        if args:
            # expect 'args' to be:
            #   <axis> T|F(\s+(<axis> T|F))*
            # <axis> = <ax>(,<ax>)*
            # <ax>   = P|CH|SB|FQ|SRC|BL|TIME|all

            # for each pair in np we must verify the axis is valid, the argument
            # is true/false and update it in the settings
            rxVal     = re.compile(r"(?P<yes>1|t(rue)?|y(es)?)|(0|f(alse)?|n(o)?)", re.I)

            def verifyAccumulate(acc, ax):
                ax2 = ax.upper()
                if ax2!="ALL" and ax2!=jenums.Axes.TYPE and ax2 not in jenums.Axes:
                    raise RuntimeError("'{0}' is not a valid plot axis".format( ax ))
                # expand to all axes if 'ALL', otherwise just append verified axis
                return acc+filter_(lambda a: a!=jenums.Axes.TYPE, jenums.Axes) if ax2=="ALL" else acc+[ax2]

            def proc_pair( ax_val ):
                (ax,val) = ax_val
                # phase one: verify and accumulate (==expand 'all' to all axes) the axis part
                ax2      = reduce(verifyAccumulate, ax.split(','), [])

                # check if the value can be made sense of
                if val is None:
                    raise RuntimeError("Missing newplot setting for axis {0}".format( ax ))

                valid    = rxVal.match(val)
                _newPlot = self.selection.newPlot

                if not valid:
                    raise RuntimeError("{0} is not a valid true/false setting".format( val ))

                # Good, now we can proceed to modifying the settings
                val = (valid.group('yes')!=None)
                def applicator(axis):
                    oval = _newPlot[axis]
                    _newPlot[axis] = val
                    return 0 if oval==val else 1
                self.npmodify = self.npmodify + sum(map(applicator, ax2))

            ## extract pairs + process
            ## Note: add sentinel element None
            args = list(args)+[None]
            # the call to map() was only done for its sideeffects
            # so drain+map() should be a good replacement
            drap(proc_pair, zip(args[::2], args[1::2]))
            #self.dirty = True

        # Display the settings, only for the axes for which the setting is true
        print("{0} {1}".format(pplt("new plots on:"), 
                hvutil.dictfold(lambda ax_val, acc: acc+"{0} ".format(ax_val[0]) if ax_val[1] else acc, "", self.selection.newPlot)))

    _isHeaderLegend = re.compile(r'^(?P<not>no)?(?P<what>header|legend|source)$', re.I).match
    def showSetting(self, *args):
        curPlotType                      = self.getPlotType()
        curPlot                          = None if curPlotType is None else plots.Plotters[curPlotType]
        args                             = filter_(operator.truth, args)
        # default: no arguments given? display everything (if there is a current plot type) otherwise show nothing
        # note: we cannot return early if curPlot is None because the Flagged/UnFlagged setting is
        #       not a plot-type dependent setting. If there is a current plot type selected then
        #       the plot specific settings are displayed/can be changed
        getFlag   = lambda: "{0} {1}{2}".format(pplt("show:"), self._showSetting, " ({0:s} + {1:s})".format(FLAG.Flagged, FLAG.Unflagged) if self._showSetting is FLAG.Both else "")
        getAttr   = lambda which, nm: "" if curPlot is None else "{0}{1}".format("" if getattr(curPlot, which) else "No", nm)
        getHeader = lambda          : getAttr('showHeader', 'Header')
        getLegend = lambda          : getAttr('showLegend', 'Legend')
        getSource = lambda          : getAttr('showSource', 'Source')
        setAttr   = lambda which    : (lambda _: None) if curPlot is None else (lambda f: setattr(curPlot, which, f is None))
        setFlag   = {'header': (setAttr('showHeader'), getHeader, lambda s, v: setattr(s, 'Header', v)),
                     'legend': (setAttr('showLegend'), getLegend, lambda s, v: setattr(s, 'Legend', v)),
                     'source': (setAttr('showSource'), getSource, lambda s, v: setattr(s, 'Source', v))}

        # In order to be let the proc_arg() function below modify values
        # outside of its scope (i.e. the values at /this/ stack frame) we must
        # either put them in a list or make them attributes of an object.
        # I chose the 2nd approach
        show      = type('',(), {'FLAG':"", 'Header':"", 'Legend':"", 'Source':""} if args else {'FLAG':getFlag(), 'Header':getHeader(), 'Legend':getLegend(), 'Source':getSource()})()
        def proc_arg(acc, arg):
            # expect 'arg' to be one of FLAG enums or (no)?(header|legend)
            newSetting = arg.capitalize()
            if newSetting in FLAG:
                if FLAG[newSetting] is not self._showSetting:
                    acc._showSetting = FLAG[newSetting]
                    acc.npmodify     = acc.npmodify + 1
                show.FLAG = getFlag()
            else:
                # Match to (no)(header|legend)
                mo = jplotter._isHeaderLegend(newSetting)
                if mo is None:
                    raise RuntimeError("Unknown show setting {0}".format(arg))
                # check if there is a current plot type whose' show setting we can manipulate
                if curPlot is None:
                    raise RuntimeError("No plot type selected to operate on")
                # kool. update showing header/legend
                fns = setFlag.get(mo.group('what').lower(), None)
                if fns is None:
                    raise RuntimeError("Unhandled show setting case '{0}'".format(mo.group('what')))
                # entry in dict is tuple of set-in-plot + get-current-from-plot and set-in-show-object functions
                (setf, curf, showf) = fns
                # get previous value from plot
                prev = curf()
                # set the target flag in the plotter
                setf( mo.group('not') )
                # get new value from plot
                new  = curf()
                # update the current value in the 'show' object (such that we collect
                # all the updated flags so we can display them later)
                showf( show, new )
                # and indicate that something has changed if the value has, in fact, changed
                acc.npmodify = acc.npmodify + (0 if new==prev else 1)
            return acc
        # process all arguments
        reduce(proc_arg, args, self)
        # decide what to print
        if show.FLAG:
            print(show.FLAG)
        hl = filter_(operator.truth, [show.Header, show.Legend, show.Source])
        if hl:
            print("{0} {1}".format(pplt("show[{0}]:".format(curPlotType)), " ".join(hl)))

    ## raw TaQL string manipulation ..
    def taqlStr(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("taqlStr accepts only one or zero arguments")
            if args[0]=="none":
                self.selection.taqlString = None
            else:
                self.selection.taqlString = args[0]
            self.dirty = True
        #print "taqlStr: ",self.selection.taqlString
        print("taqlStr: ",self.selection.selectionTaQL())

    def runTaQL(self):
        pfx  = "runTaQL"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")
        taql = CP(self.selection.selectionTaQL())
        if not taql:
            return errf("No actual selection so running query would be no-op")
        tab  = list()
        ## run the query
        with ms2util.opentable(self.msname) as tbl:
            print("Running query:", taql)
            s   = NOW()
            tab = tbl.query( taql )
            e   = NOW()
            print("Querying took\t{0:.3f}s".format( e-s ))
        print("runTaQL: selection has", len(tab), "rows")

    def plotType(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("plotType accepts only one or zero arguments")
            pt = args[0].lower()
            if pt not in plots.Types:
                raise RuntimeError("{0} not defined as valid plot type".format(pt))
            self.selection.plotType = pt
            self.dirty = True
        pt = self.selection.plotType
        print("{0} {1}{2}".format(pplt("plotType:"), pt, "" if pt is None else " ["+plots.Plotters[pt].description()+"]"))

    def getPlotType(self):
        return self.selection.plotType

    def makePlots(self):
        pfx  = "makePlots"
        errf = hvutil.mkerrf("{0} ".format(pfx, self.msname))
        if not self.msname:
            return errf("No MS loaded yet")

        sel_ = self.selection

        if not sel_.plotType:
            return errf("No plot type selected yet")

        ## Cannot do both time AND frequency averaging at the moment :-(
        #if sel_.averageTime!=AVG.NoAveraging and sel_.averageChannel!=AVG.NoAveraging:
        #    raise RuntimeError, "Unfortunately, we cannot do time *and* channel averaging at the same time at the moment. Please disable one (or both)"

        ## Create the plots!
        with plotiterator.Iterators[self.selection.plotType] as p:
            s = NOW()
            pl = p.makePlots(self.msname, self.selection, self.mappings, readflags=self.readFlags)
            e = NOW()
        print("Data munching took\t{0:.3f}s".format( e-s ))

        ## Make a new 'record' where we keep meta data + plots/data sets 
        ## with unmapped labels
        plotar2          = plots.Dict()
        plotar2.meta     = plots.Dict()
        plotar2.limits   = plots.Dict()

        E   = os.environ
        S   = E.get #lambda env, deflt: E[env] if env in E else deflt 

        plotar2.msname        = CP(self.msname)
        plotar2.column        = CP(self.mappings.domain.column)
        plotar2.uniq          = ""
        plotar2.plotType      = CP(self.selection.plotType)
        plotar2.sources       = ",".join(self.selection.sources) if self.selection.sources else "*"
        plotar2.project       = CP(self.mappings.project)
        plotar2.userathost    = S('USER', 'JohnDoe') + "@" + S('HOST','<???>')
        plotar2.chansel       = None
        if self.selection.chanSel:
            plotar2.chansel = hvutil.range_repr(hvutil.find_consecutive_ranges(self.selection.chanSel))
        if self.selection.ddSelection:
            plotar2.freqsel       = str(len(self.selection.ddSelection))
        else:
            plotar2.freqsel       = "*"
        plotar2.polarizations = set()
        plotar2.time          = datetime.datetime.isoformat(datetime.datetime.now())[:-7]
        plotar2.comment       = ""
        plotar2.weightThres   = CP(self.selection.weightThreshold)

        # Annotate with averaging setting, if any 
        if sel_.averageTime != AVG.NoAveraging:
            if sel_.timeRange:
                # Need to to better formatting
                # One full timestamp, rest offsets?
                # all with [day/]time ?
                timefmt      = lambda x: x.strftime("%Hh%Mm%S")+"{0:.2f}s".format(x.microsecond/1.0e6)[1:]
                nspd         = 24 * 60 * 60
                (day0, frac) = divmod(self.mappings.timeRange.start, nspd)
                day0         = day0 * nspd
                dayoff       = []
                def fmtRange(xxx_todo_changeme7):
                    (s, e) = xxx_todo_changeme7
                    (soff, sday) = divmod(s-day0, nspd)
                    soff         = int(soff)
                    dispday      = not dayoff
                    if dispday:
                        dayoff.append( soff )
                    stm          = hvutil.secondsInDayToTime(sday)
                    (eoff, sday) = divmod(e-day0, nspd)
                    eoff         = int(eoff)
                    etm          = hvutil.secondsInDayToTime(sday)
                    return "{0}{1}->{2}{3}".format(str(soff)+"/" if dispday or soff!=dayoff[0] else "", timefmt(stm), 
                                                     "" if eoff==soff or (eoff==soff+1 and etm<stm) else str(eoff)+"/", timefmt(etm))
                timerngs = " ".join(map(fmtRange, sel_.timeRange))    
            else:
                se = self.mappings.timeRange
                timerngs = "{0}->{1}".format(ms2util.as_time(se.start), ms2util.as_time(se.end))
            plotar2.comment = plotar2.comment + "[" + ("" if sel_.solint is None else str(sel_.solint)+"s") +" " + str(sel_.averageTime) + " avg'ed " + timerngs + "]"

        if sel_.averageChannel != AVG.NoAveraging:
            plotar2.comment = plotar2.comment + "[" + str(sel_.averageChannel) + "averaged channels " + \
                    (hvutil.range_repr(hvutil.find_consecutive_ranges(sel_.chanSel)) if sel_.chanSel else "*") + ("" if sel_.solchan is None else ":{0}".format(sel_.solchan)) + "]"

        # transform into plots.Dict() structure
        nUseless = 0
        for (label, dataset) in iteritems(pl):
            tmp  = plots.plt_dataset(dataset.x, dataset.y, dataset.m)
            if tmp.useless:
                nUseless += 1
                continue
            plotar2[label] = tmp
        if nUseless>0:
            print("WARNING: {0} out of {1} data sets contained only NaN values ({2:.2f}%)!".format( nUseless, len(pl), 100.0*(float(nUseless)/len(pl)) ))
        return plotar2

    def organizeAsPlots(self, plts, np):
        # process a 'Dict[label] => dataset' into
        # 'Dict[plotlabel] =>  Dict[datasetlabel] => dataset'
        # based on the current newPlot settings

        # from the current newPlot setting, find the plot and dataset axes
        # 1. find the axes that make up the plot label (data set label will be the rest)
        plotaxes = map_(lambda ax_nw: ax_nw[0], filter(lambda ax_nw: ax_nw[1]==True, iteritems(np)))
        splitter = plots.label_splitter(plotaxes)
        # 2. Go through all of the plots and reorganize
        def proc_ds(acc, l_dataset):
            (l, dataset) = l_dataset
            (plot_l, dataset_l) = splitter(l)
            ds = dataset.prepare_for_display( self._showSetting )
            if ds is None:
                return acc
            acc.setdefault(plot_l, plots.Dict())[ dataset_l ] = ds
            return acc
        rv = parsers.copy_attributes(plots.Dict(), plts)
        return reduce(proc_ds, iteritems(plts), rv)

    def processLabel(self, plts):
        # Version 2.0: Based on all the plot- and data set labels, come up
        # with the set of non-unique attributes in each of them. This
        # minimizes the amount of information displayed on the screen.
        # (information that is the same in all labels of a certain type don't
        # add to the information content and actually lower the information
        # density ...)
        # Keep track of #of occurrences of atrribute value per attribute
        plotattrs = collections.defaultdict(set)
        dsattrs   = collections.defaultdict(set)

        # reset some of the variables
        plts.polarizations = set()
        plts.uniq          = ""

        for plotlab in plts.keys():
            # analyze the label
            for (k,v) in plotlab.key():
                plotattrs[k].add( v )
            # process all data sets in the current plot
            for dslab in plts[plotlab].keys():
                # analyze the label
                for (k,v) in dslab.key():
                    dsattrs[k].add( v )

                # keep track of polarizations
                plts.polarizations.add( plotlab.P if plotlab.P else dslab.P )

        # If there was no P information, make sure 'None' is not in the set;
        # it won't play nice with the ",".join( ... )
        plts.polarizations.discard( None )
        plts.polarizations = ",".join(plts.polarizations) 

        # analyze the plot + data set attributes for unique values
        # those get appended to the project
        (plotuniq, plotrest) = hvutil.partition(lambda k_v: len(k_v[1])==1, iteritems(plotattrs))
        (dsuniq  ,   dsrest) = hvutil.partition(lambda k_v: len(k_v[1])==1,   iteritems(dsattrs))

        # For the unique values, we extract the VALUES and append them
        # for the non-unique values, we produce an ATTRIBUTE list such that
        # we can ask the label to format only those
        # Note that the 'uniq's are dict with set as values and we cannot use
        #   "[0]"-style indexing to get the first (the only) member out of the
        #   set ... [but we know there is only one member in the set; that's
        #   what we've filtered on! So convert to list and take the 0'th element
        expand = lambda k_v: (k_v[0], k_v[1].pop())
        uniq   = plots.label.format(map_(expand, plotuniq)+map_(expand, dsuniq))
        if uniq:
            plts.uniq = uniq
        # ok, that's one, now extract the 'keys' (the attribute names)
        plts.plotlabel = map_(operator.itemgetter(0), plotrest)
        plts.dslabel   = map_(operator.itemgetter(0),   dsrest)

        return plts


    def doMinMax(self, plts, **opts):
        defaults = {'verbose': True}
        defaults.update( **opts )
        ## Per plot we must compile the min/max of both X, Y axes per type
        #   meta[ <plot> ][ <type> ]{'xmin': ... , 'xmax': ..., 'ymin':..., 'ymax':...}
        #       contains the x/y range per plot per type, based on all datasets in the plot
        #   limits[ <type> ]{'xmin':..., 'xmax':..., 'ymin':..., 'ymax':...}
        #       contains the global x/y range per type across all plots
        DD            = collections.defaultdict
        plts.meta     = DD(lambda: DD(plots.Dict))
        plts.limits   = DD(plots.Dict)

        what_to_show = self._showSetting
        def reductor(acc, lds):
            tref = acc[lds[0].TYPE]
            lds1 = lds[1]
            # We *know* that xlims/ylims exist because datasets that do not have
            # any points in them are filtered out in organizeAsPlots()
            xl         = lds1.xlims
            yl         = lds1.ylims
            tref[ 0 ].append( xl[0] ) # xmin
            tref[ 1 ].append( xl[1] ) # xmax
            tref[ 2 ].append( yl[0] ) # ymin
            tref[ 3 ].append( yl[1] ) # ymax
            return acc

        s = NOW()
        glimits = DD(lambda: (list(), list(), list(), list()))
        for plotlab in plts.keys():
            for (tp, mdata) in iteritems(reduce(reductor, iteritems(plts[plotlab]), DD(lambda: (list(), list(), list(), list())))):
                mref      = plts.meta[plotlab][tp]
                mref.xlim = (min(mdata[0]), max(mdata[1]))
                mref.ylim = (min(mdata[2]), max(mdata[3]))
                # append to global limits
                glimits[tp][ 0 ].append( mref.xlim[0] )
                glimits[tp][ 1 ].append( mref.xlim[1] )
                glimits[tp][ 2 ].append( mref.ylim[0] )
                glimits[tp][ 3 ].append( mref.ylim[1] )
        # and set them
        for (tp, mdata) in iteritems(glimits):
            mref      = plts.limits[tp]
            mref.xlim = (min(mdata[0]), max(mdata[1]))
            mref.ylim = (min(mdata[2]), max(mdata[3]))
        e = NOW()

        if defaults['verbose']:
            print("min/max processing took\t{0:.3f}s                ".format( e-s ))

        if False:
            for k in plts.meta.keys():
                print("META[",k,"]")
                for d in plts.meta[k].keys():
                    print("   DS[",d,"]/ x:",plts.meta[k][d].xlim," y:",plts.meta[k][d].ylim)
            for k in plts.limits.keys():
                print("LIMITS[",k,"]/ x:",plts.limits[k].xlim," y:",plts.limits[k].ylim)
        return plts

    def drawFunc(self, plotar, dev, fst, onePage=None, **opts):
        plotter = plots.Plotters[plotar.plotType]
        s = NOW()
        plotter.drawfunc(dev, plotar, fst, onePage, **opts)
        e = NOW();
        if opts.get('verbose', True):
            print("drawing took\t{0:.3f}s                ".format( e-s ))

    def reset(self):
        self.msname              = None
        self.plots               = None
        self.dirty               = False
        self.npmodify            = 0
        self.scanlist            = []
        self.averaging           = AVG.NoAveraging
        self.selection           = selection.selection()
        self.mappings            = ms2mappings.mappings()
        self._showSetting        = FLAG.Unflagged
        self.readFlags           = True


    def markedDirty(self, *args):
        if args:
            self.dirty = args[0]
        return self.dirty



def mk_output(pfx, items, maxlen):
    lines = [ copy.copy(pfx) ]
    def reducer(acc, item):
        if len(acc[-1])+len(item)>maxlen:
            acc.append( copy.copy(pfx) )
        acc[-1] += " {0}".format(item)
        return acc
    for l in reduce(reducer, items, lines):
        print(l.strip())




class environment(object):
    def __init__(self, uniq, devNm):
        self.j         = jplotter(unique=uniq)
        self.device    = None
        self.devNColor = 0
        # plots    = datasets organized per plot, ready for plotting
        #            (i.e. the "new plot" settings have been applied)
        # rawplots = the raw one-dimensional list of plots
        self.plots     = None
        self.rawplots  = None
        # keep track of which kinds of post-processing need to be done
        self.newRaw    = False # True if we have new rawplots
        self.newPlots  = False # True if someone re-subdivided rawplots => plots
        self.first     = 0
        self.devName   = devNm
        self.wd        = os.getcwd()
        self.post_processing_mod = None
        self.post_processing_fn  = None

    def select(self):
        if self.device is None:
            try:
                self.device = ppgplot.pgopen(self.devName)
                # set up color index table
                (loci, hici) = ppgplot.pgqcir()
                # for now, define at most 32 extra colours
                # so we have 32 + 16 (predfined in PGPLOT) - 2 (black/white) = 46 colours
                nExtra = min(hici - loci, 32)
                # Generate extra colours and write them in the color index table
                # was: map(lambda (ci, (r,g,b)): ppgplot.pgscr(...), ...)
                # the map() was only used for its side-effects so drain()+map() 
                # is a good replacement
                def set_pgscr(ci_rgb):
                    (ci, (r, g, b)) = ci_rgb
                    ppgplot.pgscr(ci, r, g, b)
                drap(set_pgscr, zip(itertools.count(loci), gencolors.getncol_rgb(nExtra)))
                # set background to white and text to black
                ppgplot.pgscr(0, 1.0, 1.0, 1.0)
                ppgplot.pgscr(1, 0.0, 0.0, 0.0)
                self.devNColor = loci + nExtra
            except:
                raise RuntimeError("Sorry, failed to open device '{0}'".format(self.devName))
        ppgplot.pgslct(self.device)
        ppgplot.pgask( False )
        ocwd = os.getcwd()
        os.chdir(self.wd)
        ncwd = os.getcwd()
        if ocwd!=ncwd:
            print("[wd now is: {0}]".format(ncwd))

    # we expect to be called as ".cwd("path/to/cd/to", ...)" or ".cwd()"
    # (if '...' is non-empty, that's an error though
    # note: could have done with "cwd(self, args=None)" but then, if the user
    #       input is:
    #            > cd aap noot
    #       (erroneous user input, but then again, they're users ;-))
    #       the error message is:
    #           "TypeError: cwd() takes at most 2 arguments (3 given)"
    #       which is confusing since there were only 2 arguments typed
    #       (the user does not (need to) know about the (hidden) 'self') ...
    def cwd(self, *args):
        if len(args)>1:
            raise RuntimeError("This command only supports 0 or 1 arguments")
        # support '~' expansion at the start of the path
        os.chdir( re.sub(r"^~", os.environ['HOME'], ("~" if not args else args[0])) )
        self.wd = os.getcwd()

    def close(self):
        if self.device:
            ppgplot.pgslct(self.device)
            ppgplot.pgclos()
        self.device = None

    def changeDev(self, newDevName):
        self.close()
        self.devName = newDevName
        self.select()

    def navigable(self):
        # only windows are navigable, they have numbers
        rxWindow = re.compile(r"^[0-9]+/xw", re.I)
        return rxWindow.match(self.devName)

    def postProcess(self, *args):
        if args:
            if len(args)>1:
                raise RuntimeError("postProcess accepts only zero or one arguments")
            if args[0]=="none":
                self.post_processing_fn  = None
                self.post_processing_mod = None
            else:
                # args[0] = [/path/to/]MODULE.FUNCTION
                # extract the path from the module - in case someone
                # gave one we have to look for the module in that path
                (m_path, m_file) = os.path.split(args[0])
                # split the MODULE.FUNCTION into MODULE and ".FUNCTION"
                (m_name, m_fn)   = os.path.splitext(m_file)
                if not m_fn:
                    raise RuntimeError("Invalid MODULE.FUNCTION specification")
                # strip the "." from ".FUNCTION"
                m_fn  = m_fn[1:]
                f     = None
                opath = CP(sys.path)
                try:
                    # We must support loading from ourselves so we add copies
                    # of directories in sys.path that have a jiveplot
                    # subdirectory to sys.path
                    drap(partial(sys.path.insert, 0),
                                 filter(os.path.isdir,
                                        map_(lambda p: os.path.join(p, 'jiveplot'), sys.path)))
                    # Now we add another level - go over all the paths _again_
                    # but now adding m_path and see if that path exists, but only if m_path seems
                    # to be a relative path
                    if not m_path:
                        # no path at all, just MODULE.FUNCTION - add os.getcwd() to search
                        # where we are executing from too
                        # No need to go over sys.path because there is no m_path to add ;-)
                        sys.path.insert(0, os.getcwd())
                    elif m_path[0]!='/':
                        # relative path
                        drap(partial(sys.path.insert, 0),
                                     filter(os.path.isdir,
                                            map_(lambda p: os.path.join(p, m_path), sys.path)))
                        # since it's a relative path, should look in current working dir as well
                        p = os.path.join(os.getcwd(), m_path)
                        if os.path.isdir(p):
                            sys.path.insert(0, p)
                    else:
                        # m_path is an absolute path, add it
                        if os.path.isdir(m_path):
                            sys.path.insert(0, m_path)

                    (f, p, d) = imp.find_module(m_name)
                    mod = imp.load_module(m_name, f, p, d)
                    self.post_processing_fn  = mod.__dict__[m_fn]
                    self.post_processing_mod = args[0]
                except ImportError:
                    print("Failed to locate module {0}".format(m_name))
                except KeyError:
                    print("Function {0} not found in module {1}".format(m_fn, m_name))
                finally:
                    if f:
                        f.close()
                    sys.path = opath

        print("postProcessing: {0}".format( self.post_processing_mod ))


# Some commands need to have the whole expression as a single argument
# but we'd like non-empty strings (i.e. nothing remains after stripping 
# the command and whitespace) to not count as an argument.
def all_or_nothing(s):
    return [s] if s else []

#####
#####      The jcli "jive command line interface" thingamabob
#####
def run_plotter(cmdsrc, **kwargs):
    """Start a command line interface to select/plot data from a MS.

    Supported keyword args:

        unique=True|False   a MS can carry much more meta data (e.g.
                            antenna's, spectral windows etc) than are
                            actually referenced in the MAIN table.
                            unique=True => only show meta data (selectables)
                                           that ARE actually referenced
                            unique=Fale => show ALL meta data
                            default: True
    """
    ## Set default parameters
    defaults = { 'unique': True}
    defaults.update(kwargs)
    # borrow the "mkcmd" function and only make it visible
    # in this scope
    from jiveplot import command
    from jiveplot.command import mkcmd
    from jiveplot.helpfile import Help

    # These objectes we certainly need
    app = "jcli"
    c = command.CommandLineInterface(debug=defaults['debug'], app=app)

    o         = type('', (), {})()
    o.curdev  = 42
    foo       = {o.curdev: environment(defaults['unique'], "{0}/xw".format(o.curdev))}
    j         = lambda : foo[o.curdev].j
  
    # Start building the commandset

    # the "ms" command
    c.addCommand( \
        mkcmd(rx=re.compile(r"^ms\b.*$"), hlp=Help["ms"], \
              args=lambda x : re.sub(r"^ms\s*", "", x).split(), \
              cb=lambda *args : j().ms(*args), id="ms") )

    # uniq meta data on next ms open?
    c.addCommand( \
        mkcmd(rx=re.compile(r"^uniq(\s+[tfTF10])?$"), id="uniq", \
              args=lambda x: re.sub(r"^uniq\s*", "", x).split(), \
              cb=lambda *args: j().uniqueMetaData(*args), \
              hlp=Help["uniq"]) )

    # run indexr on the MS
    c.addCommand(
        mkcmd(rx=re.compile(r"^inde?xr$"), id="indexr",
            cb=lambda : j().indexr(), hlp=Help["indexr"]) )

    # 'listr' is the 'task' to display the scanlist
    c.addCommand(
        mkcmd(rx=re.compile(r"^listr\b.*$"), id="listr",
              args=lambda x: re.sub("^listr\s*", "", x).split(),
              cb=lambda *args: j().listScans(*args),
              hlp="listr\n\tDisplay list of scans found by 'indexr'") )

    # select scans
    c.addCommand(
        mkcmd(rx=re.compile(r"^scan(\s+.*)?$"), id="scan",
              args=lambda x: all_or_nothing(re.sub(r"^scan\s*", "", x)),
              cb=lambda *expr: j().scans(*expr), hlp=Help["scan"]))

    # the "range" command
    def unknown_range(x):
        def print_it():
            print("#### UNKNOWN range key '{0}'".format(x))
        return print_it

    def range_f(*args):
        if not j().haveMS():
            print("No MS opened yet")
        else:
            defaults = ['fq', 'ant', 'src', 'time']
            disps = { 
                    'fq'  : j().listFreqs,
                    'ch'  : j().listFreqs,
                    'p'   : j().listFreqs,
                    'sb'  : j().listFreqs,
                    'bl'  : j().listBaselines,
                    'ant' : j().listAntennas,
                    'src' : j().listSources,
                    'time': j().listTimeRange,
                    'scan': j().listScans
                    }
            # If no arguments show default ranges. Yield error if unknown range key is requested
            # this map(...) was only called for its side-effects; thus drain()+map()
            # (==drap(...)) should be a sufficient replacement for Py2/Py3
            drap(lambda x: x(), \
                map(lambda k: disps.get(k, unknown_range(k)), \
                    defaults if not args else map(str.lower, args)))
        
    c.addCommand( \
        mkcmd(rx=re.compile(r"^r\b.*$"), hlp=Help["r"], \
              args=lambda x: re.sub(r"^r\s*", "", x).split(), cb=range_f, id="r") )

    # set the plot layout
    def layout_f(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        plotter = plots.Plotters[pt]
        if args:
            plotter.layout( *args )
        print("{0} {1} [{2}]".format(pplt("layout[{0}]:".format(pt)), plotter.layout(), plotter.layoutStyle() ))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^nxy(\s+[0-9]+\s+[0-9]+)?(\s+(fixed|flexible|rows|columns))*$", re.I), \
              # don't convert to integers just yet - leave that up to the actual layout function
              args=lambda x: re.sub("^nxy\s*", "", x).split(), \
              cb=layout_f, id="nxy", \
              hlp="nxy [nx ny] [fixed|flexible] [rows|columns]\n\tprint or set the current plot layout\n\nThe layout can be marked as fixed or flexible. In the latter case jplotter might re-arrange the layout to fit all plots on one page when this seems feasible. By default plot layouts are 'flexible' and 'rows' are filled first") )


    # list known plot-types "lp"
    def list_pt():
        print("Known plot-types:")
        # compute longest plot type name
        longest = max( map(compose(len, str), plots.Types) )
        # this map() was used only for its side-effects
        # so drain() + map() should be sufficient Py2/Py3 replacement
        drap(lambda x : print("{0:{1}} => {2}".format(x, longest, plots.Plotters[x].description())), sorted(plots.Types))
    c.addCommand( \
        mkcmd(rx=re.compile(r"^lp$"), hlp=Help["lp"], \
              cb=list_pt, id="lp") )

    # multisubband?
    rxYesNo = re.compile(r"^(?P<yes>t(rue)?)|f(alse)?$", re.I)
    def multisb(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        if args:
            if len(args)>1:
                raise RuntimeError("This command supports only one argument")
            mo = rxYesNo.match( args[0] )
            if not mo:
                raise RuntimeError("Invalid argument, it's not t(rue) or f(alse)")
            # it's either t(rue) or f(alse)
            plots.Plotters[pt].multisb( mo.group('yes') is not None )
        print("multisubband[{0}]: {1}".format( pt, plots.Plotters[pt].multisb() ))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^multi(\s+(t(rue)?|f(alse)?))?$", re.I), \
              args=lambda x : re.sub(r"^multi\s*", "", x).split(), \
              cb=multisb, id="multi", \
              hlp="multi [t(rue)|f(alse)]\n\tplot multiple subbands next to each other i.s.o over each other") )

    # Set x,y scaling of the plots
    #  * auto global [all plots same scale, automatically derived from maximum
    #       across all plots]
    #  * auto local [each plot gets autoscaled]
    #  * fixed limits [user defined limits]
    rScale  = re.compile(r"^(?P<ax>[xy])((?<=y)(?P<idx>[01]))?(\s+((?P<scaling>local|global)|(?P<lo>\S+)\s+(?P<hi>\S+)))?$")
    rxScale = re.compile(r"^x(\s+((?P<scaling>local|global)|(?P<lo>\S+)\s+(?P<hi>\S+)))?$")
    ryScale = re.compile(r"^y(?P<idx>[01])?(\s+((?P<scaling>local|global)|(?P<lo>\S+)\s+(?P<hi>\S+)))?$")
    def scale_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return

        # function 'pointers' to the scaling functions
        fns = {
            'x': lambda dummy, *scale : plots.Plotters[pt].xscale(*scale),
            'y': lambda idx, *args    : plots.Plotters[pt].yscale(idx, *args)
        }
        mo  = rScale.match(args[0])
        fn  = fns[ mo.group('ax') ]
        req = mo.group('scaling')
        lim = mo.group('lo') and mo.group('hi')
        idx = int(mo.group('idx')) if mo.group('idx') else 0 
        if req:
            if req=='local':
                s = plots.Scaling.auto_local
            elif req=='global':
                s = plots.Scaling.auto_global
            fn( idx, s )
        if lim:
            fn( idx, [float(mo.group('lo')), float(mo.group('hi'))] )
        print("{0}-scale [{1}]: {2}".format( mo.group('ax'), pt, fn(idx) ))

    # HV: 24/Apr/2017 - There is no documentation for 'y' or 'x'
    #                   because they're all under 'xyscale' which
    #                   is a bit unintuitive
    c.addCommand( \
        mkcmd(rx=rxScale, id="x", args=lambda x: x,
              cb=scale_fn, hlp=Help["xyscale"]) )
    c.addCommand( \
        mkcmd(rx=ryScale, id="y", args=lambda x: x,
              cb=scale_fn, hlp=Help["xyscale"]) )

    # Allow labels to be set on axes!
    rxLabel = re.compile(r'label(\s+\S+.*)?$')
    def label_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        labels = plots.Plotters[pt].setLabel(*args)
        for (which, tp, txt) in labels:
            print("{0}: {1}[{2}] '{3}'".format(pt, which, tp, txt))

    c.addCommand( \
            mkcmd(rx=rxLabel, id="label", args=lambda x : re.sub(r"^label\s*", "", x), \
                  cb=label_fn, hlp=Help["label"])) #"""label [<axis1>:'<label1 text>' [<axisN>:'<labelN text>' ...]]\n\tShow or set axis label(s)""") )

    # allow sorting by arbitrary keys
    #  sort [p ch fq etc]
    def sort_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        print("sort order [{0}]: {1}".format(pt, plots.Plotters[pt].sortby(*args)))
    c.addCommand( \
        mkcmd(rx=re.compile(r"^sort\b.*$"), \
              args=lambda x: re.sub("^sort\s*", "", x).split(), \
              cb=sort_fn, id="sort", \
              hlp=Help["sort"]) )

    # we can now set line width, point size and marker size
    def set_lw(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        lw = plots.Plotters[pt].setLineWidth(*args)
        print("linewidth[{0}]: {1}".format(pt, lw))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^linew(idth)?(\s+[0-9]+)?$"), \
              args=lambda x: map_(int, re.sub("^linew(idth)?\s*", "", x).split()),
              cb=set_lw, id="linew", \
              hlp="linew(idth) [<number>]\n\tset/display line width when drawing line plots"))

    def set_ps(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        ps = plots.Plotters[pt].setPointSize(*args)
        print("pointsize[{0}]: {1}".format(pt, ps))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^ptsz(\s+[0-9]+)?$"), \
              args=lambda x: map_(int, re.sub("^ptsz\s*", "", x).split()),
              cb=set_ps, id="ptsz", \
              hlp="ptsz [<number>]\n\tset/display point size when drawing point plots"))

    def set_ms(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        ms = plots.Plotters[pt].setMarkerSize(*args)
        print("markersize[{0}]: {1}".format(pt, ms))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^marksz(\s+[0-9]+)?$"), \
              args=lambda x: map_(int, re.sub("^marksz\s*", "", x).split()),
              cb=set_ms, id="marksz", \
              hlp="marksz [<number>]\n\tset/display marker size for marking marked points"))


    # Allow drawing lines, points or both
    def draw_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        if args:
            plots.Plotters[pt].setDrawer(*args)
        print("drawers[{0}]: {1}".format(pt, plots.Plotters[pt].setDrawer()))

    c.addCommand(
        mkcmd(rx=re.compile(r"^draw\b.*$"), id="draw",
              args=lambda x: re.sub("^draw\s*", "", x).split(), cb=draw_fn, hlp=Help["draw"]) )

    # colour-key related functions
    def ckey_f(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        ckf = plots.Plotters[pt].colkey_fn( *args )
        print("ckey[{0}]: {1}".format(pt, ckf))

    c.addCommand( \
        mkcmd(rx=re.compile(r"^ckey\b.*$"), id="ckey",
              hlp=Help["ckey"],
              args=lambda x: all_or_nothing(re.sub("^ckey\s*", "", x)), cb=ckey_f) )

    # post-reading pre-plotting filtering
    # this allows the user to filter data sets before plotting
    # but after they've been yanked from disk
    # [User request: "can we not display cross-pol phase in the phase panel of Amplitude+Phase plots?"]
    #    (User == JayB)
    #   commands:
    #       > filter none
    #       > filter p in [ll,rr]
    #       > filter phase: bl ~ /wb*/ and sb in [1,2]
    #       > filter 0: sb != 2
    #   queries
    #       > filter
    #       > filter phase:
    #       > filter 0:
    rxFilter = re.compile(r"^filter(\s+(?P<idx>\d+|[a-zA-Z]+)\s*:)?(\s+(?P<filter>\S.*))?$")
    def filter_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        pref = plots.Plotters[pt]
        mo   = rxFilter.match(args[0])
        idx  = mo.group('idx')
        filt = mo.group('filter')

        # Verify that, if an index is specified, it is a sensible one
        if idx:
            try:
                idx = int(idx)
            except ValueError:
                # no, definitely not an int, so better be a valid y-Axis specifier
                try:
                    idx = pref.yAxis.index(idx)
                except ValueError:
                    idx = len(pref.yAxis)
            if idx<0 or idx>=len(pref.yAxis):
                raise RuntimeError("Invalid index {0}".format(mo.group('idx')))
            idx = [idx]

        # if idx still None, it's all y-Axes
        if idx is None:
            idx = list(range_(len(pref.yAxis)))

        # command or query is defined by wether or not there is a filter string
        if filt:
            for i in idx:
                pref.filter_f(i, filt)
        prefix = lambda i: pplt("filter[{0}/{1}]:".format(pt, pref.yAxis[i]))
        for i in idx:
            print("{0} {1}".format(prefix(i), pref.filter_f(i)))

    c.addCommand( 
            mkcmd(rx=rxFilter, id="filter", args = lambda x: x, cb=filter_fn,
                  hlp=Help["filter"])
        )

    # animation!
    # animate the plots by some third axis (=combination of attribute values)
    rxAnimate = re.compile(r"^animate\s+(?P<expr>\S.*)$")
    def animate_fn(*args):
        if not foo[o.curdev].navigable():
            raise RuntimeError("Animation only available on screen devices")
        # the parser might be fed with keywords - we may have to check if they are available
        tr = j().mappings.timeRange if j().mappings is not None else None
        # parse the expression and loop over the plots
        if tr:
            cruft = parsers.parse_animate_expr(args[0], start=tr.start, end=tr.end, length=tr.end-tr.start, \
                                                        mid=(tr.start+tr.end)/2.0, t_int=tr.inttm[0])
        else:
            cruft = parsers.parse_animate_expr(args[0])
        # check what the parser gave us
        # ( (dataset, filter), (groupby_f, sortfns) )
        # if the dataset is not available, not much to do is there?
        (ds_filter, grp_sort, settings) = cruft
        (dataset_id, filter_f)          = ds_filter
        (groupby_f,  sort_f)            = grp_sort
        e = env() 
        if dataset_id is None:
            # no dataset from memory so we'll have to refresh
            refresh(e)
            if not e.rawplots:
                raise RuntimeError("No plots created(yet)?")
            the_plots = e.rawplots
        else:
            if dataset_id not in datasets:
                raise RuntimeError("The data set {0} does not exist".format(dataset_id))
            the_plots = datasets[dataset_id]
            if not parsers.isDataset(the_plots):
                raise RuntimeError("The variable {0} does not seem to refer to a set of plots")
        keys = sort_f(filter(filter_f, the_plots.keys()))
        if not keys:
            raise RuntimeError("After filtering there were no plots left to animate")
        # Now we group_by and organize each set as plots
        print("Preparing ", len(keys), " datasets for animation")
        sequence = []
        s_time = NOW()
        for group_key, datakeyiter in itertools.groupby(keys, groupby_f): #groups:
            tmp = j().processLabel( j().organizeAsPlots(parsers.ds_key_filter(the_plots, datakeyiter), j().getNewPlot()) )
            if e.post_processing_fn:
                e.post_processing_fn( tmp, j().mappings )
            # rerun this because things may have changed
            j().doMinMax(tmp, verbose=False)
            if tmp:
                sequence.append( tmp )
        e_time = NOW()
        print("Preparing animation took\t{0:.3f}s                ".format( e_time - s_time ))
        if not sequence:
            print("No plots to animate, unfortunately ...")
            return None
        # loop indefinitely
        fps = settings.fps if hasattr(settings, 'fps') else 0.7
        try:
            env().select()
            dT = 1.0/fps
            print("Press ^C to stop the animation [{0}fps]".format( fps ))
            while True:
                for (page, page_plots) in enumerate(sequence):
                    # we really would like to have all plots on one page
                    # TODO: expand nx/ny to accomodate this?
                    s = NOW()
                    with plots.pgenv(ppgplot) as p:
                        j().drawFunc(page_plots, ppgplot, 0, plots.AllInOne, ncol=env().devNColor, verbose=False)
                    while True:
                        nsec = (s + dT) - NOW()
                        if nsec<=0:
                            break
                        time.sleep( nsec )
                    # wait for next interval
        except KeyboardInterrupt:
            pass
        return None

    c.addCommand( 
            mkcmd(rx=rxAnimate, id="animate", args = lambda x: x, cb=animate_fn,
                  hlp=Help["animate"])
        )

    # Marking points?
    #   commands:
    #       > mark none
    #       > mark y > 0
    #       > mark phase: y > 0
    #       > mark 0: y > 0
    #   queries
    #       > mark
    #       > mark phase:
    #       > mark 0:
    rxMark = re.compile(r"^mark(\s+(?P<idx>\d+|[a-zA-Z]+)\s*:)?(\s+(?P<mark>\S.*))?$")
    def mark_fn(*args):
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to operate on")
            return
        pref = plots.Plotters[pt]
        mo   = rxMark.match(args[0])
        idx  = mo.group('idx')
        mark = mo.group('mark')

        # Verify that, if an index is specified, it is a sensible one
        if idx:
            try:
                idx = int(idx)
            except ValueError:
                # no, definitely not an int, so better be a valid y-Axis specifier
                try:
                    idx = pref.yAxis.index(idx)
                except ValueError:
                    idx = len(pref.yAxis)
            if idx<0 or idx>=len(pref.yAxis):
                raise RuntimeError("Invalid index {0}".format(mo.group('idx')))
            idx = [idx]

        # if idx still None, it's all y-Axes
        idx = list(range_(len(pref.yAxis))) if idx is None else idx

        # command or query is defined by wether or not there is a mark string
        if mark:
            for i in idx:
                pref.mark(i, mark)
        prefix = lambda i: pplt("mark[{0}/{1}]:".format(pt, pref.yAxis[i]))
        for i in idx:
            print("{0} {1}".format(prefix(i), pref.mark(i)))

    c.addCommand( 
            mkcmd(rx=rxMark, id="mark", args = lambda x: x, cb=mark_fn,
                  hlp=Help["mark"])
        )

    # reset default plot settings for current selected plot type
    def reset():
        pt = j().getPlotType()
        if not pt:
            print("No plot type selected to reset")
            return
        plots.Plotters[pt].reset()
        
    c.addCommand( \
        mkcmd(rx=re.compile(r"^reset$"), \
              cb=reset, id="reset", \
              hlp="reset\n\treset current plot type to default layout, scaling etc") )

    # the plot properties command "pp"
    def plot_props():
        j().plotType()
        pt = j().getPlotType()
        if pt is not None:
            layout_f()
            mark_fn("mark")
        j().averageTime()
        j().averageChannel()
        j().solint()
        j().nchav()
        j().weightThreshold()
        j().newPlot()
        j().showSetting()

    c.addCommand( \
        mkcmd(rx=re.compile(r"^pp$"), hlp=Help["pp"], \
              cb=plot_props, id="pp") )


    # env() is a function returning the current environment, if any
    def env():
        if o.curdev is None:
            raise RuntimeError("Current plot device closed. Create a new one{0}.".format(
                    " or select one from {0}".format(list(foo.keys())) if foo else "" ))
        try:
            return foo[o.curdev]
        except KeyError:
            raise RuntimeError("Internal error: device '{0}' is not a key in the dict?!".format(o.curdev))

    open_win  = lambda x: ppgplot.pgopen(repr(x)+"/xw")
    def open_file(x):
        x  = str(x)
        mo = rxExt.match(x)
        if not mo:
            x = x + ".ps"
        if mo and not mo.group('ft'):
            x = x + "/cps"
        return ppgplot.pgopen(x)

    # actually get data sets from disk
    # store in 'rawplots' and organized
    # as plots in the 'plots' variable
    def refresh(e):
        e.newRaw = False
        if not e.rawplots or j().markedDirty():
            try:
                e.rawplots = j().makePlots()
            except KeyboardInterrupt:
                e.rawplots = None
                print(">>> plotting cancelled by user")
                return False

            if e.rawplots is None or len(e.rawplots)==0:
                print("No plots produced? Rethink your selection")
                e.rawplots = None
                return False
            # succesfully refreshed plots!
            e.newRaw   = True
            j().markedDirty( False )
        return e.newRaw

    def do_plot(e):
        # maybe we need to refresh
        refresh(e)
        # now go to standard plot post processing - something else may have changed
        redraw_after_new(e)

    def redraw_after_new(e):
        # do we have new raw plots? or did the newPlot setting changed?
        # either would warrant making a new plot/dataset array
        if (e.newRaw or j().newPlotChanged()) and e.rawplots:
            e.newRaw   = False # pre-reset this flag to prevent it rmaining set after e.g. an exeption below
            tmp = j().processLabel( j().organizeAsPlots(e.rawplots, j().getNewPlot()) )
            if not tmp:
                raise RuntimeError("WARNING: no plots generated")
            e.plots    = tmp
            e.first    = 0
            e.newPlots = True
        # if we have a new set of plots created ... we must run the postprocessing, if any
        if e.newPlots and e.plots:
            # okiedokie, that's done (pre-reset the flag such that it doesnt remain
            # set in case of error
            e.newPlots = False
            if e.post_processing_fn:
                e.post_processing_fn( e.plots, j().mappings )
            # rerun this because things may have changed
            j().doMinMax(e.plots)
        # now it's time to redraw on da skreen
        do_redraw(e)

    def do_redraw(e):
        if not e.plots:
            return
        # redraw w/o refreshing or postprocessing
        e.select()
        # un-navigatable plotdevices get all plots in one go
        with plots.pgenv(ppgplot) as p:
            j().drawFunc(e.plots, ppgplot, e.first, foo[o.curdev].navigable(), ncol=e.devNColor)

    c.addCommand( \
            mkcmd(rx=re.compile(r"^pl$"), hlp="pl:\n\tplot current selection with current plot properties", \
            cb=lambda : do_plot(env()), id="pl") )

    # dry-run the current TaQL
    c.addCommand( \
            mkcmd(rx=re.compile(r"^run_taql$"), hlp="run_query:\n\tAttempt to selectthe current selection without plotting",\
                  cb=lambda: j().runTaQL(), id="run_taql")
            )

    # control what to show: flagged, unflagged, both
    c.addCommand( \
            mkcmd(rx=re.compile(r"^show(\s\S+)*$"), hlp=Help["show"], \
            args=lambda x: re.sub("^show\s*", "", x).split(), \
            cb=lambda *args: j().showSetting(*args), id="show") )

    # produce a hardcopy postscript file
    rxExt    = re.compile(r"(\.[^./]+)$")
    rxType   = re.compile(r'(/[a-z]+)$', re.I)
    type2ext = functools.partial(re.compile(r'^[/vc]*', re.I).sub, '.')
    ext2type = {".ps":"/CPS", ".pdf":"/PDF", ".png":"/PNG"}
   
    # convert user input device specification into
    # a PGPLOT compatible device string
    def user2pgplot(filenm):
        ext, tp = None, None
        pgfn    = copy.deepcopy(filenm)
        # check if trailing type was given
        mo = rxType.search(pgfn)
        if mo:
            # yes, remember it + strip it
            tp   = mo.group(1)
            pgfn = rxType.sub("", pgfn)
        # id. for extension
        mo = rxExt.search(pgfn)
        if mo:
            # yarrrs. remember + strip
            ext  = mo.group(1)
            pgfn = rxExt.sub("", pgfn)
        # if there was a type but not extension, use that for extension
        # if there was an extension but no type don't change that
        # if there were none, use defaults
        tbl = { (True, True)  : lambda e, t: (".ps", "/cps"),
                (True, False) : lambda e, t: (type2ext(t), t),
                (False, True) : lambda e, t: (e, ext2type.get(e.lower(),"Unknown")),
                (False, False): lambda e, t: (e, t)}
        (pext, ptp) = tbl[(ext is None, tp is None)](ext, tp)
        return (pgfn+pext, ptp)

    def mk_postscript(e, filenm):
        if not filenm:
            ppgplot.pgldev()
            return
        refresh(e)
        if not e.plots:
            print("No plots to save, sorry")
            return
        # returns device file name and type as separate items
        # so we can display the file name w/o the (inferred) type
        fn, ptp = user2pgplot(filenm)
        try:
            f = ppgplot.pgopen(fn+ptp)
        except:
            raise RuntimeError("Sorry, failed to open file '{0}'".format(e.wd +"/"+fn if not fn[0] == "/" else fn))
        ppgplot.pgslct(f)
        ppgplot.pgask( False )
        j().drawFunc(e.plots, ppgplot, 0, ncol=e.devNColor)
        ppgplot.pgclos()
        e.select()
        print("Plots saved to [{0}]".format(fn))

    c.addCommand( \
        mkcmd(rx=re.compile("^save(\s+\S+)?$"), id="save",
              args=lambda x: re.sub("^save\s*", "", x),
              cb=lambda x: mk_postscript(env(), x),
              hlp="save <filename>\n\tsave current plots to file <filename> in PostScript format.\nThe extension .ps and default lands cape '/cps' orientation will be added automatically if not given") )

    # arithmancy support - allows store/retrieve of (the result of applying an expression to) plots 
    #  # ... create plots
    #  $> store as phatime_sfxc
    #  # ... get other data, create different plots
    #  $> store as phatime_unb
    #  # now we can do Fun Stuff(tm)!
    #  $> load phatime_sfxc-phatime_sfxc
    #  $> pl
    datasets = {}   # our collection of variables

    # this fn is only called after a load/store has changed the current
    # set of plots. So we know we have a new set of raw plots
    def refresh_after_reload(e):
        # set current plot type!
        j().plotType(e.rawplots.plotType)
        # we stored plots from memory into current; prevent reloading from disk!
        j().markedDirty(False)
        # trigger full postprocessing
        e.newRaw = True
        # and let'r rip
        redraw_after_new(e)

    # Feature request Lorant S: be able to show currently defined variables
    rxShowVars = re.compile(r"^(store|load)\s*$")

    def show_vars_fn():
        print("Currently defined variables:")
        namelen  = max(max(map(len, datasets.keys())), 10) if datasets else 0
        for (k,v) in iteritems(datasets):
            print("{0:<{1}} = {2}".format(k, namelen, "'{0}' from {1} [{2} plots]".format(v.plotType, v.msname, len(v)) if parsers.isDataset(v) else str(v)))
            

    def store_fn(expr):
        if rxShowVars.match(expr):
            return show_vars_fn()
        # set current plots as default
        e = env()
        datasets['_'] = prevplt = e.rawplots
        parsers.parse_dataset_expr(expr, datasets)
        e.rawplots    = datasets['_']
        # plots replaced? replot
        if isinstance(e.rawplots, type({})) and e.rawplots is not prevplt:
            print("stored: {0} datasets in current".format( len(e.rawplots) ))
            refresh_after_reload(e)

    def load_fn(expr):
        if rxShowVars.match(expr):
            return show_vars_fn()
        e = env()
        prevplt    = e.rawplots
        e.rawplots = parsers.parse_dataset_expr(expr, datasets)
        if isinstance(e.rawplots, type({})):
            print("loaded: {0} datasets into current".format( len(e.rawplots) ))
            # and redraw if necessary
            if prevplt is not e.rawplots:
                refresh_after_reload(e)

    c.addCommand( \
            mkcmd(rx=re.compile(r"^store\b.*"), id="store", args=lambda x:x,
                  cb=store_fn, hlp=Help["store"]) )
    c.addCommand( \
            mkcmd(rx=re.compile(r"^load\b.*"), id="load", args=lambda x:x,
                  cb=load_fn, hlp=Help["load"]) )


    # multi window support
    def win_fn(*args):
        if args:
            if len(args)>1:
                raise RuntimeError("This command only supports one argument")
            # select [possibly create first] window
            dev_id = args[0]
            if dev_id not in foo:
                foo[dev_id] = environment(defaults['unique'], "{0}/xw".format(dev_id))
            o.curdev = dev_id
            foo[o.curdev].select()
        print("Current plot window: ",o.curdev)

    c.addCommand( \
            mkcmd(rx=re.compile(r"win(\s+[0-9]+)?$"), id="win", \
                  args=lambda x: map_(int, re.sub(r"win\s*", "", x).split()), \
                  cb=win_fn, \
                  hlp="win [<num>]\n\topen/select plot window <num> for subsequent plots or display current plot window") )

    # and mult file support
    def file_fn(*args):
        if args:
            if len(args)>1:
                raise RuntimeError("This command only supports one argument")
            # select [possibly create] file
            rxRefile = re.compile(r"^refile\s+(?P<file>.+)$")
            refile   = rxRefile.match(args[0])
            fn       = refile.group('file') if refile else args[0]
            fn, ptp  = user2pgplot(fn)
            fn       = fn+ptp
            if refile:
                foo[o.curdev].changeDev(fn)
            else:
                if fn not in foo:
                    foo[fn] = environment(defaults['unique'], fn)
                o.curdev = fn
                foo[fn].select()
        print("Current plot file: ",foo[o.curdev].devName)

    c.addCommand( \
            mkcmd(rx=re.compile(r"^(re)?file(\s+\S+)?$"), id="file", \
                  args=lambda x: all_or_nothing(re.sub(r"^file\s*", "", x)), \
                  cb=file_fn, \
                  hlp="file [<filename>]\n\topen/select plot file <filename> for subsequent plots") )

    # close current device
    def close_fn():
        foo[o.curdev].close()
        del foo[o.curdev]
        o.curdev = None

    c.addCommand( mkcmd(rx=re.compile(r"^close$"), id="close", cb=close_fn,
        hlp="close\n\tclose current plot device [file or window]\nNote: this action takes any opened MS, selection and plot settings  in the current plot down with it!") )

    # advance/go back a number of pages of plots/go to a specific page
    #  support:
    #   ([0-9]+[np]?)|[npfl]
    # * only number? go to that page
    # * number + [np] go forward/back the number of pages
    # * shortcuts [fl] for "first" and "last"
    # * just [np] go foward/back one page
    rxPage = re.compile(r"^(?P<num>[0-9]+)(?P<direction>[np])?|(?P<shorthand>[npfl])$")
    def page_fn(e, cmdstr):
        # sanity check
        if not e.plots or len(e.plots)==0:
            print("No plots at all. Make some first before browsing through them")
            return
        # if o.curdev is not integer, we cannot iterate through it
        if not foo[o.curdev].navigable():
            print("current device is not window so cannot navigate")
            return
        # get the current plot type and find the layout (for the number of plots/page)
        pt = j().getPlotType()

        nppage   = plots.Plotters[pt].layout().nplots()
        nplot    = len(e.plots)
        last     = nplot - (nplot % nppage)
        # the only time "last" can be equal to "nplot" is when
        # (nplot % nppage) == 0 i.e. when the number of plots to
        # plot fits exactly onto an integer number of pages
        if last==nplot:
            last = last - nppage

        # given current first plot, compute the new first plot
        switches = {
             'f' : lambda x: 0
            ,'l' : lambda x: max(last, 0)
            ,'n' : lambda x: min(e.first + x * nppage, last)
            ,'p' : lambda x: max(e.first - x * nppage, 0)
        }

        # we know 'cmdstr' matches rxPage
        mo = rxPage.match(cmdstr)

        if mo.group('shorthand'):
            # move one page; "first" and "last" ignore the argument
            n         = 1
            direction = mo.group('shorthand')
        else:
            n         = int(mo.group('num'))
            direction = mo.group('direction')
            if direction is None:
                # this means absolute page number.
                # User requested consisten use of 0-based or 1-based numbering.
                # Plot pages are labelled in 1-based numbering; this code
                # used 0-based page addressing.
                # The difference only becomes obvious when going to an
                # absolute page number ("2 pages next" is unambiguous).
                n = n - 1
        oldfirst = e.first
        e.first  = switches.get(direction, lambda x: min(x*nppage, last))( n )
        if oldfirst!=e.first:
            #print "go to page {0}".format( e.first / nppage )
            do_redraw( e )

    c.addCommand( \
        mkcmd(rx=rxPage, id="page", args=lambda x: x, \
              cb=lambda x: page_fn(env(), x), \
              hlp="<num>[pn]?|[fl]\n\tbrowse through plot pages.\n"+
"""
* just digits? go to that page of plots
* digits + [pn]? go to n-th previous or next page
* [fl]   go to first/last page of plots
* [np]   go forward/back one page of plots
   """) )

    def navigate_fn(e):
        mouse = {'A': 'n', 'X':'p'}
        if not e.plots or len(e.plots)==0:
            print("No plots to browse")
            return
        if e.device is None:
            print("No device to navigate")
            return
        # if o.curdev is not integer, we cannot iterate through it
        if not foo[o.curdev].navigable():
            print("current device is not window so cannot navigate")
            return
        # Enter into main loop
        print("entering interactive mode. 'q' to leave, 'help page' for navigation commands.")
        buf = ""
        while True:
            (x,y,ch) = ppgplot.pgcurs()
            ch = mouse.get(ch, ch)
            if ch in "q":
                break
            elif ch in "fl":
                page_fn(e, ch)
            elif ch in "0123456789":
                buf += ch
            elif ch in "np":
                buf += ch
                page_fn(e, buf)
                buf = ""
            elif ord(ch)==27:
                # 'ESC'
                buf = ""
            elif ord(ch)==13:
                # enter - if buf is just
                # a number, we go to that page
                if buf:
                    page_fn(e, buf)
                buf = ""
            #else:
            #    print "character ",ch,ord(ch)," unhandled"

    rxInteractive = re.compile(r"^i$")
    c.addCommand( \
        mkcmd(rx=rxInteractive, id="interactive", \
              cb=lambda : navigate_fn(env()), \
              hlp="i\n\tenter interactive mode to browse through plot pages. Same commands as for 'page' command, 'q' to quit."))


    # show the current selection, "sl"
    def curselection():
        if not j().haveMS():
            print("No MS opened yet")
        else:
            j().freqsel()
            j().channels()
            j().baselines()
            j().sources()
            j().times()
    c.addCommand( \
        mkcmd(rx=re.compile(r"^sl$"), hlp=Help["sl"], \
              cb=curselection, id="sl") )

    # set/inquire plot type, "pt"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^pt(\s+\S+)?$"), hlp=Help["pt"], \
              args=lambda x : re.sub(r"^pt\s*", "", x).split(), \
              cb=lambda *args : j().plotType(*args), id="pt") )

    # set/display the timerange selection
    c.addCommand( \
        mkcmd(rx=re.compile(r"^time\b.*$"), hlp=Help["time"], \
              args=lambda x : all_or_nothing(re.sub(r"^time\s*", "", x)), \
              cb=lambda *args: j().times(*args), id="time") )

    # The channel selection "ch"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^ch\b.*$"), hlp=Help["ch"], \
              args=lambda x : re.sub(r"^ch\s*", "", x).split(), \
              cb=lambda *args: j().channels(*args), id="ch") )

    # The frequency selection "fq"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^fq(\s+(none|[0-9/:PpXxLlRr*,]+))*$"), hlp=Help["fq"], \
              args=lambda x : re.sub(r"^fq\s*", "", x).split(), \
              cb=lambda *args: j().freqsel(*args), id="fq") )

    # The baseline selection "bl"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^bl(\s+[0-9a-zA-Z|()*+\-!]+)*$"), hlp=Help["bl"], \
              args=lambda x : re.sub(r"^bl\s*", "", x).split(), \
              cb=lambda *args: j().baselines(*args), id="bl") )

    # The source selection "src"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^src(\s+\S+)*$"), hlp=Help["src"], \
              args=lambda x: re.sub("^src\s*", "", x).split(), \
              cb=lambda *args: j().sources(*args), id="src") )

    # Average in time or frequency, set solution interval (time averaging buckets)
    c.addCommand( \
        mkcmd(rx=re.compile(r"^avt\b.*$"), hlp=Help["avt"], \
              args=lambda x: re.sub("^avt\s*", "", x).split(), \
              cb=lambda *args: j().averageTime(*args), id="avt") )
    c.addCommand( \
        mkcmd(rx=re.compile(r"^avc\b.*$"), hlp=Help["avc"], \
              args=lambda x: re.sub("^avc\s*", "", x).split(), \
              cb=lambda *args: j().averageChannel(*args), id="avc") )

    c.addCommand( \
        mkcmd(rx=re.compile(r"^solint\b.*$"), hlp=Help["solint"], \
              args=lambda x: re.sub("^solint\s*", "", x).split(), \
              cb=lambda *args: j().solint(*args), id="solint") )
    c.addCommand( \
        mkcmd(rx=re.compile(r"^nchav\b.*$"), hlp=Help["nchav"], \
              args=lambda x: re.sub("^nchav\s*", "", x).split(), \
              cb=lambda *args: j().nchav(*args), id="nchav") )

    # Weigth threshold
    c.addCommand( \
        mkcmd(rx=re.compile(r"^wt\b.*$"), hlp=Help["wt"], \
              args=lambda x: re.sub("^wt\s*", "", x).split(), \
              cb=lambda *args: j().weightThreshold(*args), id="wt") )

    # the 'new plot' command
    c.addCommand( \
        mkcmd(rx=re.compile(r"^new\b.*$"), \
              hlp=Help["new"], args=lambda x: re.sub("^new\s*", "", x).split(), \
              cb=lambda *args: j().newPlot(*args), id="new") )

    # the TaQL command - insert raw taql command
    c.addCommand( \
        mkcmd(rx=re.compile(r"taql\b.*$"), hlp=Help["taql"], \
              args=lambda x: all_or_nothing(re.sub("^taql\s*", "", x)), \
              cb=lambda *args: j().taqlStr(*args), id="taql") )

    # O/S interface "cd" , "pwd" and "ls"
    c.addCommand( \
        mkcmd(rx=re.compile(r"^cd\b.*$"), id="cd",
              args=lambda x: command.quote_split(x, ' ')[1:],
              # *args = list of arguments passed to lambda. First argument is list thus two dereferences
              #         (we only use the first argument to "cd" - "cd foo bar" will do "cd foo"
              cb=lambda *args: env().cwd(*args),
              hlp="cd [dir]\n\tchange current working directory (default: $HOME)") )

    c.addCommand( \
        mkcmd(rx=re.compile("^ls(\s+\S.+)?$"), id="ls", args=lambda x: x, cb=lambda x: os.system(x),
              hlp="ls [dir]\n\tlist directory contents (default: current working directory)") )

    def pwd():
        print(os.getcwd())
    c.addCommand( \
        mkcmd(rx=re.compile(r"^pwd$"), id="pwd", cb=pwd,
              hlp="pwd\n\tprint current working directory") )

    # Write selection to new MS (reference copy)
    def write_tab(tabname):
        if not j().haveMS():
            raise RuntimeError("No MS opened yet")
        with ms2util.opentable(j().haveMS()) as orgtab:
            t2     = orgtab.query(j().selection.selectionTaQL())
            t2.copy(tabname)
            t2.close()

    c.addCommand( \
        mkcmd(rx=re.compile(r"^write\s+\S+$"), id="write", \
              hlp="write <tablename>\n\tWrite current selection out as new (reference) table", \
              args=lambda x: re.sub("^write\s*", "", x), cb=write_tab) )

    # configure/inspect post processing
    c.addCommand( \
        mkcmd(rx=re.compile(r"^postprocess\b.*"), id="postprocess",
              hlp="postprocess [MODULE.FUNCTION]\n\tSet/display function to call on processed data",
              args=lambda x: re.sub("^postprocess\s*", "", x).split(), 
              cb=lambda *args: env().postProcess(*args)) )

    #def test_f(*args):
    #    print "test_f/n_arg=",len(args)," args=",args

    c.addCommand(
        mkcmd(rx=re.compile(r"^test_f(\s+\S+)*$"), id="test_f",
              args=lambda x: re.sub("^test_f\s*", "", x).split(), cb=lambda *args: test_f(*args)) )

    c.run(cmdsrc)

    # Clean up all plot windows
    for (k,v) in iteritems(foo):
        v.close()
    ppgplot.pgend()

#if __name__=='__main__':
#print '***********************************************'
#print
#print ' You can now type jplotter.jcli() to start the '
#print '        command line interface....'
#print
#jcli()

