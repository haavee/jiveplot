#
#  A collection of utilities for ms2 (lookup tables etc..)
#  Rewrite in Python, January 2013, Harro Verkouter
#
#
#  $Id: ms2util.py,v 1.20 2017-01-27 13:50:28 jive_cc Exp $
#
#  $Log: ms2util.py,v $
#  Revision 1.20  2017-01-27 13:50:28  jive_cc
#  HV: * jplotter.py: small edits
#          - "not refresh(e)" => "refresh(e); if not e.plots ..."
#          - "e.rawplots.XXX" i.s.o. "e.plots.XXX"
#      * relatively big overhaul: in order to force (old) pyrap to
#        re-read tables from disk all table objects must call ".close()"
#        when they're done.
#        Implemented by patching the pyrap.tables.table object on the fly
#        with '__enter__' and '__exit__' methods (see "ms2util.opentable(...)")
#        such that all table access can be done in a "with ..." block:
#           with ms2util.opentable(...) as tbl:
#              tbl.getcol('DATA') # ...
#        and then when the with-block is left, tbl gets automagically closed
#
#  Revision 1.19  2015-09-23 12:28:36  jive_cc
#  HV: * Lorant S. requested sensible requests (ones that were already in the
#        back of my mind too):
#          - option to specify the data column
#          - option to not reorder the spectral windows
#        Both options are now supported by the code and are triggered by
#        passing options to the "ms" command
#
#  Revision 1.18  2015-04-29 14:34:14  jive_cc
#  HV: * add support for retrieving the actual frequencies of the channels
#
#  Revision 1.17  2015-02-16 12:51:07  jive_cc
#  HV: * "getcolslice()" in combination with j2ms2 inefficient tiling scheme
#        results in very poor performance. So now we do the "slicing"
#        ourselves in Python [might change again when j2ms2 does things more
#        efficiently or when the casa
#
#  Revision 1.16  2014-04-25 12:22:30  jive_cc
#  HV: * deal with lag data correctly (number of channels/number of lags)
#      * there was a big problem in the polarization labelling which is now fixed
#
#  Revision 1.15  2014-04-24 20:09:19  jive_cc
#  HV: * indexr now uses 'SCAN_NUMBER' column for scan determination
#
#  Revision 1.14  2014-04-14 22:08:01  jive_cc
#  HV: * add support for accessing scan properties in time selection
#
#  Revision 1.13  2014-04-14 14:46:05  jive_cc
#  HV: * Uses pycasa.so for table data access waiting for pyrap to be fixed
#      * added "indexr" + scan-based selection option
#
#  Revision 1.12  2014-04-10 21:14:40  jive_cc
#  HV: * I fell for the age-old Python trick where a default argument is
#        initialized statically - all data sets were integrating into the
#        the same arrays! Nice!
#      * Fixed other efficiency measures: with time averaging data already
#        IS in numarray so no conversion needs to be done
#      * more improvements
#
#  Revision 1.11  2014-04-08 23:34:12  jive_cc
#  HV: * Minor fixes - should be better now
#
#  Revision 1.10  2014-04-08 22:41:11  jive_cc
#  HV: Finally! This might be release 0.1!
#      * python based plot iteration now has tolerable speed
#        (need to test on 8M row MS though)
#      * added quite a few plot types, simplified plotters
#        (plotiterators need a round of moving common functionality
#         into base class)
#      * added generic X/Y plotter
#
#  Revision 1.9  2013-12-12 14:10:16  jive_cc
#  HV: * another savegame. Now going with pythonic based plotiterator,
#        built around ms2util.reducems
#
#  Revision 1.8  2013-08-20 18:23:50  jive_cc
#  HV: * Another savegame
#        Got plotting to work! We have stuff on Screen (tm)!
#        Including the fance standardplot labels.
#        Only three plot types supported yet but the bulk of the work should
#        have been done I think. Then again, there's still a ton of work
#        to do. But good progress!
#
#  Revision 1.7  2013-06-19 12:28:44  jive_cc
#  HV: * making another savegame
#
#  Revision 1.6  2013-03-31 17:17:56  jive_cc
#  HV: * another savegame
#
#  Revision 1.5  2013-03-09 16:59:07  jive_cc
#  HV: * another savegame
#
#  Revision 1.4  2013-02-19 16:53:29  jive_cc
#  HV: * About time to commit - make sure all edits are safeguarded.
#        Making good progress. baselineselection, sourceselection and
#        timeselection working
#
#  Revision 1.3  2013-02-11 09:40:33  jive_cc
#  HV: * saving work done so far
#        - almost all mapping-functionality is in place
#        - some of the API functions are starting to appear
#
#  Revision 1.2  2013-01-29 12:23:45  jive_cc
#  HV: * time to commit - added some more basic stuff
#
#  Revision 1.1.1.1  2013-01-25 09:53:40  jive_cc
#  HV: Initial import of some of the python code for the
#      rewrite of standardplots into python
#
#  Revision 1.3  2003-10-29 12:35:38  verkout
#  HV: Changed the way of how to open sub-tables of a MS; now using the right way rather than the JIVE way..:)
#
#  Revision 1.2  2001/04/19 14:59:32  verkout
#  HV: Added project detection+added it to the plots properties
#
#  Revision 1.1.1.1  2001/04/06 13:34:34  verkout
#  Files, new + from jivegui/MS1
from   __future__ import print_function
from   functional import map_, filter_, zip_, enumerate_
from   six        import iteritems
import itertools, operator, math, re, jenums, hvutil, numpy, pyrap.tables, pyrap.quanta, sys
import datetime, copy, collections
from functools import reduce

# Return a nicely human readable string representation
# of a MS::TIME column value
as_time = lambda x : pyrap.quanta.quantity(x, "s").formatted("dmy", 9)

# string -> time conversion with a knack
#  The system supports dayoffsets. In order for this to work correctly 
#  it is advisory to fill in the refdata parameter with the zero-date.
#  It does not matter what the actual time of day of that refdate will be
#  since the date will be automatically truncated to 0h00m on the refdate
def str2time(s, refdate=0.0):
    totime = lambda x : pyrap.quanta.quantity(x).totime()

    #  compute 0h00 on refdate
    secday = 60.0*60.0*24.0
    time0  = (math.floor(refdate/secday))*secday

    # remove all whitespace
    s = re.sub("\s+", "", s)

    # If it was all numeric, assume it's MJD seconds - the internal
    # measurement set format
    if re.match(r"^\d+(\.\d*)?$", s):
        return pyrap.quanta.quantity(float(s), "s")

    # rxrelday = RE to see if it was a relative day + timestamp
    #   no more slashes may occur after the first one
    relday   = re.match(r"^([+-]?\d+)/([^/]+)$", s)

    # IF it was a relative day - take out the relative day nr
    #  and the actual timestamp
    if relday:
        time0  += (secday * float(relday.group(1)))
        s       = relday.group(2)

    # now we should be left with a timestamp which casa(core) can
    # parse as a timestring
    # There is one form of timestring that casacore does not recognize:
    #   MMmSS.SSSs (just minutes and seconds)
    # Let's add our own format for that
    minsec = re.match(r"^(\d+)m(\d+(\.\d*)?s)?$", s)
    tmq    = None
    if minsec:
        # set the timequantity
        tmq = totime(minsec.group(1)+"min") 
        if minsec.group(2):
            tmq += totime(minsec.group(2))
    else:
        # attempt the conversion
        tmq = totime(s)

    # if we have < 1 day's worth of seconds, add the refdate
    if tmq.get_value("s")<secday:
        tmq += pyrap.quanta.quantity(time0, "s").totime()

    return tmq


# we like our table opens to be silent
# so if the user did not pass a value for
# 'ack' (the 5th argument for the table
# function) we'll add a default of 'False'
def opentable(*args, **kwargs):
    d = {}
    if len(args)<5:
        d['ack'] = False
    d.update(kwargs)
    # HV: Before we used to return the pyrap.tables.table() return value unadorned.
    #     This, however, does not have success written all over it. Specifically,
    #     either the pyrap, boost.python or C++ Table::... code holds a reference
    #     from '<name>' to internal object. 
    #     This means that if '<name>' changes on disk, pyrap.tables.table('<name>')
    #     will keep on returning the _old_ (!!!!) table information AS LONG AS THERE
    #     ARE pyrap.tables.table() OBJECTS ALIVE THAT HAVE NOT __EXPLICITLY__ (!!!!)
    #     CALLED ".close()" ON THE TABLE.
    #
    #     This seems to happen based on *name* of the table because the table objects
    #     you get returned from pyrap.tables.table() are /different/; you really
    #     get different instances of table object:
    #        >>> t1 = table('foo.ms')
    #        >>> t2 = table('foo.ms')
    #        >>> id(t1) == id(t2)
    #        False
    #
    #     We fix this by adding '__enter__' and '__exit__' "methods" to the pyrap.tables.table
    #     object such that it can be used as a context manager; i.e.:
    #
    #         with opentable('<name>') as tbl:
    #            # blah
    #            print len(tbl)
    #
    #      When 'tbl' goes out of scope (leaves the with-scope), the table is automagically closed.
    realtable = pyrap.tables.table(*args, **d)
    # casacore-python (or python-casacore) - the successor of pyrap,
    # does seem to have '__enter__'/'__exit__' methods
    if not hasattr(realtable, '__enter__'):
        setattr(realtable, '__enter__', lambda *args: realtable)
    if not hasattr(realtable, '__exit__'):
        setattr(realtable, '__exit__' , lambda *args: realtable.close())
    return realtable

# Exceptions that can be thrown
class NotAMeasurementSet(Exception):
    def __init__(self, ms):
        self.ms = ms

    def __str__(self):
        return "{0} is not a minimal MeasurementSet!".format(self.ms)

class InvalidFreqGroup(Exception):
    def __init__(self,fgrp):
        self.freqGroup = fgrp

    def __str__(self):
        return "FREQ_GROUP {0} not found!".format(self.freqGroup)

class InvalidSubband(Exception):
    def __init__(self,sb):
        self.subbandIdx = sb

    def __str__(self):
        return "Subband #{0} is out of range".format(self.subbandIdx)

class InvalidSpectralWindow(Exception):
    def __init__(self, spw):
        self.spWinId = spw

    def __str__(self):
        return "SPECTRAL_WINDOW_ID {0} is unknown".format(self.spWinId)

class InvalidDataDescId(Exception):
    def __init__(self, ddid):
        self.dataDescId = ddid

    def __str__(self):
        return "DATA_DESC_ID {0} is unknown".format(self.dataDescId)

class InvalidPolIdForSubband(Exception):
    def __init__(self,pid,*args):
        self.polarizationId = pid
        self.moreArgs       = args

    def __str__(self):
        return "The polarizationId {0} is not valid for {1}".format(self.polarizationId, self.moreArgs)

class InvalidAntenna(Exception):
    def __init__(self, id):
        self.antennaId = id

    def __str__(self):
        return "The antenna ID {0} is unknown".format(self.antennaId)

class InvalidBaselineId(Exception):
    def  __init__(self, blid):
        self.baselineIdx = blid

    def __str__(self):
        (x,y) = decodeblidx(self.baselineIdx)
        return "At least one of the antennas in baseline {0} is unknown ({1} or {2})".format(self.baselineIdx, x, y)

class InvalidPolarizationCode(Exception):
    def __init__(self,polstr):
        self.polarizationCode = polstr

    def __str__(self):
        return "The polarization code {0} is not a recognized polarization".format(self.polarizationCode)

class InvalidFieldId(Exception):
    def __init__(self,fld):
        self.fieldId = fld

    def __str__(self):
        return "The field code {0} is not a recognized field".format(self.fieldId)

##############################################################################################
##                    data structures
##############################################################################################

##
## The polarization mapping
##
class polarizationmap():
    ## there exists a static mapping between AIPS++ polarization code
    ##  numerical value (as found in the MS) to the human readable equivalent
    ##  so we start with a bit of class-level code to deal with that
    class polcode():
        def __init__(self, **kwargs):
            self.code = kwargs['id']
            self.name = kwargs['name']

    CorrelationStrings = [
          polcode(id=1 ,name='I' ),
          polcode(id=2 ,name='Q' ),
          polcode(id=3 ,name='U' ),
          polcode(id=4 ,name='V' ),
          polcode(id=5 ,name='RR'),
          polcode(id=6 ,name='RL'),
          polcode(id=7 ,name='LR'),
          polcode(id=8 ,name='LL'),
          polcode(id=9 ,name='XX'),
          polcode(id=10,name='XY'),
          polcode(id=11,name='YX'),
          polcode(id=12,name='YY')#,
          #polcode(id=13,name='RX'),
          #polcode(id=14,name='RY'),
          #polcode(id=15,name='LX'),
          #polcode(id=16,name='LY'),
          #polcode(id=17,name='XR'),
          #polcode(id=18,name='XL'),
          #polcode(id=19,name='YR'),
          #polcode(id=20,name='YL') 
          ]

    @staticmethod
    def polcodeById(corrtype):
        try:
            [pcode] = filter_(lambda x: x.code==corrtype, polarizationmap.CorrelationStrings)
            return copy.deepcopy(pcode)
        except ValueError:
            raise InvalidPolarizationCode(corrtype)

    # Return a list of correlationIDs for which the name matches the regex...
    @staticmethod
    def matchingName2ID(rx):
        return [x.code for x in polarizationmap.CorrelationStrings if rx.match(x.name)]

    @staticmethod
    def correlationId2String(id):
        return ",".join( polarizationmap.correlationId2Strings(id) )
    @staticmethod
    def correlationId2Strings(id):
        # Assume 'id' is a list-of-correlation-types
        return map_(lambda x: polarizationmap.polcodeById(x).name, id)

    #  Turn an array of polarizationcombination strings into
    #  an array of correlation types
    #  FIXME XXX FIXME
    @staticmethod
    def string2CorrelationId(s):
        lows = s.lower()
        cs   = map(lambda x: polarizationmap.polcode(id=x.code, name=x.name.lower()), polarizationmap.CorrelationStrings)
        def findid(nm):
            try:
                [x] = filter_(lambda y: y.name.lower()==nm.lower(), cs)
                return x.code
            except ValueError:
                raise InvalidPolarizationCode(nm)
        (result,names) = ([],s.lstrip().rstrip().split(','))
        map(result.append, map(findid, names))
        return result

    # This is where the instance methods go - these are
    # for mapping what was actually contained in the MS
    #  polmap = [(polid, poltypes)]
    def __init__(self, polmap):
        self.polarizationMap = polmap

    # Note: technically speaking the polarization id is
    #       a direct index into the list of polarizations
    #       so *technically* we could do self.polarizationMap[id]
    #       but then we - because of pythonicity - also can
    #       address polarization id "-1", which would be the
    #       last one in the list. Which we don't want because
    #       then the "id" isn't actually the id anymore.

    # return the polarizations as a string with comma-separated
    # polarization strings
    def getPolarizationsString(self, id):
        return ",".join(self.getPolarizations(id))

    # return the polarization strings as list of strings
    def getPolarizations(self, id):
        try:
            # enforce unique search result otherwise bomb out
            [x] = filter_(lambda y: y[0]==id, self.polarizationMap)
            return polarizationmap.correlationId2Strings(x[1])
        except ValueError:
            raise InvalidPolarizationCode(id)

    def getCorrelationTypes(self, id):
        try:
            # enforce unique search result otherwise bomb out
            [x] = filter_(lambda y: y[0]==id, self.polarizationMap)
            return x[1]
        except ValueError:
            raise InvalidPolarizationCode(id)

    def polarizationIDs(self):
        return [x[0] for x in self.polarizationMap]

    def nrPolarizationIDs(self):
        return len(self.polarizationMap)

def makePolarizationMap(nm, **args):
    errf   = hvutil.mkerrf("makePolarizationMap({0})".format(nm))
    with opentable(nm) as tbl:
        with opentable(tbl.getkeyword('POLARIZATION')) as poltab:
            if len(poltab)==0:
                return errf("No rows in POLARIZATION table?!")

            # read all polarization types
            get_type = lambda x: x['CORR_TYPE']
            return polarizationmap(zip_(itertools.count(), map(get_type, poltab)))



ct2str = polarizationmap.correlationId2String

class subband:
    # ddmap = Map POLARIZATION_ID => DATA_DESCRIPTION_ID,
    #    all polarization IDs this subband was correlated with
    #
    # TODO FIXME
    #   it is possible that >1 DATA_DESCRIPTION_ID maps to
    #   the same (SPECTRAL_WINDOW, POLID) tuple. So basically
    #   the ddmap should be
    #         Map POLARIZATION_ID => [DATA_DESCRIPTION_ID, ...]
    def __init__(self, spwid, f0, nchan, bw, ddmap, **kwargs):
        self.spWinId       = spwid
        self.frequency     = f0
        self.numChannels   = nchan
        self.bandWidth     = bw
        self.datadescMap   = ddmap
        self.formatStr     = "{frequency[0]:5.4f}MHz/{bw:<.1f}MHz nch={nch} {polarizations}"
        self.domain        = kwargs.get('domain', jenums.Type.Unknown)
        # in lag domain the frequency axis becomes delay in units of s
        if self.domain==jenums.Type.Lag:
            freqs            = self.frequency
            self.frequency   = numpy.arange(-self.numChannels, self.numChannels) * 1./(2*self.bandWidth)
            self.numChannels = len(self.frequency)
            self.formatStr   = "{0:5.4f}MHz {1:.2}us - {2:.2}us".format(freqs[0]/1.e6, self.frequency[0]/1e-6, self.frequency[-1]/1e-6) + "  nlag={nch} {polarizations}"

    def __str__(self, **kwargs):
        pmap = kwargs.get('polarizationMap', None)
        if pmap is None:
            polarizations = " ".join(map("P{0}".format, self.datadescMap.keys()))
        else:
            polarizations = " ".join(map(lambda x: "P{0}={1}".format(x, ct2str(pmap.getCorrelationTypes(x))), self.datadescMap.keys()))
        return self.formatStr.format(frequency=(self.frequency/1.0e6), bw=self.bandWidth/1.e6, spw=self.spWinId, ddMap=self.datadescMap, nch=self.numChannels, polarizations=polarizations)

    # (attempt to) map the indicated polarization id to a DATA_DESCRIPTION_ID
    #  None if we don't have this polid
    #  FIXME TODO
    #     would become a list-of-data_desc_id's
    def mapPolId(self, polid):
        try:
            return self.datadescMap[polid]
        except KeyError:
            raise InvalidPolIdForSubband(polid, str(self))

    # unmap a DATA_DESC_ID to a POLARIZATION_ID, if we have the DATA_DESC_ID,
    # [] otherwise
    def unmapDDId(self, ddid):
        def foldfn(x, acc):
            # Note: 'x' is always a (int, int) tuple but by analyzing it like below
            #        we can perform the extending of the accumulator only if the 
            #        datadescription id we're looking for is in this subband's 
            #        data description map in a one liner
            ###
            ###  FIXME TODO
            ###     the data desc map would be a list-of-data_desc_id's so we must
            ###     look for 'ddid' in the list of datadescids!
            ###       (use "if ddid in v")
            acc.extend([k for (k,v) in [x] if v==ddid])
            return acc
        return hvutil.dictfold(foldfn, [], self.datadescMap)

    def getDataDescIDs(self):
        return list(self.datadescMap.values())

## the spectral map class
##   holds a mapping and defines lookup functions on those mappings
class spectralmap:
    # spm         = Map FQGROUP -> [subband()]
    #  but we want each subband tagged with a numberical index;
    #  our spectralmap should look like:
    # spectralMap = Map FQGROUP -> [(sbidx, subband())]
    #  thus we tag the incoming list of subbands with a
    #  logical subband number, starting from 0
    #
    #  Update: assume FQGROUP == (FREQ_GROUP :: int, FREQ_GROUP_NAME :: string)
    def __init__(self, spm):
        self.spectralMap  = hvutil.dictmap(lambda k_v: enumerate_(k_v[1]), spm)

    # Simple API functions
    def nFreqId(self):
        return len(self.spectralMap)

    def freqIds(self):
        return map_(operator.itemgetter(0), self.spectralMap.keys())

    def freqGroupName(self, fq):
        try:
            [key] = filter_(lambda num_name: fq==num_name[0] or fq==num_name[1], self.spectralMap.keys())
            return key[1]
        except ValueError:
            raise InvalidFreqGroup(fq)

    # our lookup key may now be either int or string.
    # may raise KeyError if not found
    def _findFQ(self, fqkey):
        try:
            [key] = filter_(lambda num_name: fqkey==num_name[0] or fqkey==num_name[1], self.spectralMap.keys())
            return self.spectralMap[key]
        except ValueError:
            raise KeyError

    def subbandsOfFREQ(self, fq):
        try:
            return self._findFQ(fq)
        except KeyError:
            raise InvalidFreqGroup(fq)

    # FREQ0 indexed by spectral-window id and via "FREQGRP/SUBBAND"
    #   typically we first unmap the SPECTRAL_WINDOW_ID to a
    #    FREQGROUP/SUBBAND and then do the (easy) lookup
    def frequencyOfSPW(self, spwid):
        try:
            rv = self.unmap(spwid)
            return self.frequenciesOfFREQ_SB(rv.FREQID, rv.SUBBAND)[0]
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def frequenciesOfSPW(self, spwid):
        try:
            rv = self.unmap(spwid)
            return self.frequenciesOfFREQ_SB(rv.FREQID, rv.SUBBAND)
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def frequencyOfFREQ_SB(self, fq, sb):
        try:
            fqref = self._findFQ(fq)
            return fqref[sb][1].frequency[0]
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    def frequenciesOfFREQ_SB(self, fq, sb):
        try:
            fqref = self._findFQ(fq)
            return fqref[sb][1].frequency
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    # Id. for NUMCHAN
    def numchanOfSPW(self, spwid):
        try:
            rv = self.unmap(spwid)
            return self.numchanOfFREQ_SB(rv.FREQID, rv.SUBBAND)
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def numchanOfFREQ_SB(self, fq, sb):
        try:
            fqref = self._findFQ(fq)
            return fqref[sb][1].numChannels
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    # Id. for the polarization IDs - NOTE! These are lists!
    def polarizationIdsOfSPW(self, spwid):
        try:
            rv = self.unmap(spwid)
            return self.polarizationIdsOfFREQ_SB(rv.FREQID, rv.SUBBAND)
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def polarizationIdsOfFREQ_SB(self, fq, sb):
        try:
            # the polarization Ids are the keys in the datadescription map,
            # the values are the associated data description id's
            fqref = self._findFQ(fq)
            return list(fqref[sb][1].datadescMap.keys())
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    # Id. for the bandwidth
    def bandwidtOfSPW(self, spwid):
        try:
            rv = self.unmap(spwid)
            return self.bandwidthOfFREQ_SB(rv.FREQID, rv.SUBBAND)
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def bandwidthOfFREQ_SB(self, fq, sb):
        try:
            fqref = self._findFQ(fq)
            return fqref[sb][1].bandWidth
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    def typeOfFREQ_SB(self, fq, sb):
        try:
            fqref = self._findFQ(fq)
            return fqref[sb][1].numStr
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    # Id. for the DATA_DESCRIPTION_IDs - these can be used in TaQL directly
    #   we go from spwid => (FQGROUP, SUBBAND), then look inside that SUBBAND
    #   for POLID
    def datadescriptionIdOfSPW(self, spwid, polid):
        try:
            rv = self.unmap(spwid)
            return self.datadescriptionIdOfFREQ_SB_POL(rv.FREQID, rv.SUBBAND, polid)
        except AttributeError:
            raise InvalidSpectralWindow(spwid)

    def datadescriptionIdOfFREQ_SB_POL(self, fq, sb, polid):
        try:
            # the polarization Ids are the keys in the datadescription map,
            # the values are the associated data description id's
            fqref = self._findFQ(fq)
            return fqref[sb][1].mapPolId(polid)
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    # return all the spectral window Id's for the given FREQGROUP
    def spectralWindows(self, fq):
        try:
            fqref = self._findFQ(fq)
            return [x[1].spWinId for x in fqref]
        except KeyError:
            raise InvalidFreqGroup(fq)

    # How many subbands has the indicated FREQGROUP?
    def nSubbandOfFREQ(self, fq):
        try:
            return len(self._findFQ(fq))
        except KeyError:
            raise InvalidFreqGroup(fq)

    # Return all DATA_DESCRIPTION_IDs (mostly used to be able
    # to see if all datadescription ids are selected: if that's true,
    # the TaQL can be dropped...)
    def datadescriptionIDs(self):
        # All the datadescription IDs are contained in the subband objects,
        #  which are contained in lists-per-freqgroup
        # so we 'fold' over all our freq-groups and collect all data-desc-ids
        def foldfn(x, acc):
            (fgrp, sblist) = x
            for (idx,sb) in sblist:
                acc.extend(sb.getDataDescIDs())
            return acc
        return sorted(list(hvutil.dictfold(foldfn, [], self.spectralMap)))
        #return sorted(list(set(dictfold(foldfn, [], self.spectralMap))))

    #  Map FREQID (integer) + zero-based subbandnr into
    #  a spectral window id
    #
    #  return None on error, SEPCTRAL_WINDOW_ID (int) otherwise
    def map(self, freqid, sbid):
        try:
            if sbid<0:
                raise InvalidSubband(sbid)
            fqref = self._findFQ(fq)
            return fqref[sbid][1].spWinId
        except KeyError:
            raise InvalidFreqGroup(fq)
        except IndexError:
            raise InvalidSubband(sb)

    #  Unmap a given spectral window id (note: this is the *zero*based rownumber!!)
    #
    #  retval = None on error, record with FREQID=xxx and SUBBAND=yyyy where
    #  FREQID is the freqid (you'd never guessed that eh?) and SUBBAND is the *zero*
    #  based subband nr in this freqid that the requested spectral window represents..
    def unmap(self, spwid):
        for (fq, sbs) in iteritems(self.spectralMap):
            for (idx, sb) in sbs:
                if sb.spWinId==spwid:
                    o = type('',(),{})()
                    o.FREQID  = fq[0]
                    o.SUBBAND = idx
                    return o
        raise InvalidSpectralWindow(spwid)

    #  Unmap a DATA_DESC_ID into FREQID/SUBBAND/POLID
    def unmapDDId(self, ddid):
        # look in all FREQGROUPS, in all SUBBANDS for the given DATA_DESC_ID
        for (k,v) in iteritems(hvutil.dictmap(lambda k_v: [sb for sb in k_v[1] if sb[1].unmapDDId(ddid)], self.spectralMap)):
            if len(v)==1:
                class sbres:
                    def __init__(self,fid,sb,pid):
                        self.FREQID  = fid
                        self.SUBBAND = sb
                        self.POLID   = pid
                    def fsbpol(self):
                        return (self.FREQID, self.SUBBAND, self.POLID)
                [pol] = v[0][1].unmapDDId(ddid)
                return sbres(k[0], v[0][0], pol)
            elif len(v)==0:
                raise InvalidDataDescId(ddid)
            else:
                raise RuntimeError("Non-unique search result for DATA_DESC_ID={0}".format(ddid))
        return None

    # Print ourselves in readable format
    def __str__(self):
        r =  "*** SPWIN <-> FREQID/SUBBAND MAP ***\n"
        for (fgrp,subbands) in iteritems(self.spectralMap):
            r = r+"FQ={0} ({1})\n".format(fgrp[0], fgrp[1])
            for (idx,sb) in subbands:
                r = r + "  {0:2d}: {1}".format(idx,sb)
                if len(sb.datadescMap)>0:
                    r = r + " // {0}\n".format( \
                                      ",".join(["POLID #{0} (DDI={1})".format(pol,ddi) \
                                                   for (pol,ddi) in iteritems(sb.datadescMap)]) \
                    )
        return r



def assertMinimalMS(nm):
    ti = pyrap.tables.tableinfo(nm)
    if ti['type']!='Measurement Set':
        raise NotAMeasurementSet(nm)


def makeSpectralMap(nm, **kwargs):
    errf    = hvutil.mkerrf("makeSpectralMap({0})".format(nm))
    with opentable(nm) as tbl:
        datadom = getDataDomain(nm, **kwargs)

        # verify that we have a non-empty spectral window table
        with opentable( tbl.getkeyword('SPECTRAL_WINDOW') ) as spwint:
            if len(spwint)==0:
                return errf("No rows in spectral window table")
            if 'FREQ_GROUP' not in spwint.colnames():
                return errf("No 'FREQ_GROUP' column found?")

            # id. for the datadescription table
            with opentable(tbl.getkeyword('DATA_DESCRIPTION')) as ddt:
                if len(ddt)==0:
                    return errf("No rows in data description table")

                #  spmap:
                #      Map [FREQ_GROUP] -> { Map [SubbandIdx] = > {FREQ0, NCHAN, DATADESCMAP} }
                spmap     = collections.defaultdict(list)

                # columns in the spectral window table
                freqgroup = lambda x : x['FREQ_GROUP']
                fgrpname  = lambda x : x['FREQ_GROUP_NAME']
                chanfreq  = lambda x : x['CHAN_FREQ']
                numchan   = lambda x : x['NUM_CHAN']
                totbw     = lambda x : x['TOTAL_BANDWIDTH']

                # columns in the datadescription table
                spwid     = lambda x : x['SPECTRAL_WINDOW_ID']
                polid     = lambda x : x['POLARIZATION_ID']

                # Do different things depending on wether to load the
                # full spectral window table or only the ones that
                # are actually used in the MS

                # function to generate the spectral window's "key" function
                # from a row in the SPECTRAL_WINDOW table
                key_f = lambda row: (freqgroup(row), fgrpname(row))

                # function to create a fully fledged subband() object
                # from a row in the SPECTRAL_WINDOW table + some extra
                # info (the row# in the SPECTRAL_WINDOW table and
                # the 'polarization-id' => 'data_description_id' mapping)
                sb_f  = lambda n, row, pmap: subband(n, chanfreq(row), numchan(row), totbw(row), pmap, domain=datadom.domain)

                if not kwargs.get('unique', False):
                    ## Read the spectral window table and find all possible
                    ## correlations with it
                    for (idx,row) in enumerate(spwint):
                        # each spectral window may appear multiple times in the datadescription
                        # table - with different polarization IDs. So we build a mapping
                        # of polid -> datadescid for all the datadescriptions that match the
                        # current spectral window id
                        ddmap = dict((polid(row), ddid) for (ddid,row) in enumerate(ddt) if spwid(row)==idx)
                        # build an entry for this subband and add to current mapping
                        spmap[ key_f(row) ].append( sb_f(idx, row, ddmap) )
                else:
                    ## Input: list-of-DATA_DESC_IDs
                    ##   unmap to (key, subband) pairs
                    ##   group by (key)
                    ##   join subbands [could be same subband but different polarization id]

                    ## Extract the unique DATA_DESC_IDs from the main table, then we
                    ## simply unmap those
                    def reductor(acc, ddid):
                        # get the spectral-window id and polid for this DDID
                        (spw, pol) = (spwid(ddt[ddid]), polid(ddt[ddid]))
                        # create the subband
                        row   = spwint[spw]
                        sb    = sb_f(spw, row, {pol:ddid})
                        key   = key_f(row)
                        try:
                            # check if this sb is already present in the current
                            # freqgroup, we may have hit another POLID
                            processed = False
                            for k in spmap[key]:
                                if k.spWinId==sb.spWinId:
                                    # irrespective of the outcome of this operation we can
                                    # flag this particular data desc id as processed, such that
                                    # it don't get added
                                    processed = True
                                    # now update the subband object's data desc map with this one's
                                    # let's verify that existing (POLID -> DATA_DESC_ID) are not violated!
                                    for (p,d) in iteritems(sb.datadescMap):
                                        if p in k.datadescMap:
                                            if k.datadescMap[p]!=d:
                                                #### FIXME TODO
                                                ####    if we update the subband() object to support >1 DATA_DESC_ID
                                                ####    mapping to the same (SPECTRAL_WINDOW, POLID) tuple this can go
                                                raise RuntimeError("Inconsistent DATA_DESCRIPTION/SPECTRAL_WINDOW table. \
                                                                     Spectral window {0} appears with POLID={1} -> DATA_DESC_ID={2} \
                                                                     but also as POLID{3} -> DATA_DESC_ID{4}".format( \
                                                                        sb.spWinId, p, d, p, k.datadescMap[p] ))
                                        else:
                                            # ok, no entry for this polarization yet
                                            k.datadescMap[p] = d
                            # if we didn't find the subband yet, add it to the current key!
                            if not processed:
                                spmap[key] = spmap[key]+[sb]
                        except KeyError:
                            spmap[key] = [sb]
                        return spmap
                    # On an 8.7Mrow (that's 8.7e6 rows) MS this statement takes 54.96(!) seconds
                    # Un-effing-believable!
                    # pyrap.tables.taql("select unique DATA_DESC_ID from $tbl", locals={"tbl":tbl}).getcol("DATA_DESC_ID"), \
                    #
                    # Let's see if we can do better!
                    # Thus:
                    #
                    # pyrap.tables.taql("select unique DATA_DESC_ID ...")     takes 54.96s  (8.7Mrow)
                    # set(sorted(tbl.getcol('DATA_DESC_ID')))                 takes  4.38s  (   ..  )
                    # set(tbl.getcol('DATA_DESC_ID'))                         takes  1.35s  (   ..  )
                    # tbl.getcol('DATA_DESC_ID')                              takes  0.36s  (   ..  )
                    # numpy.unique( tbl.getcol('DATA_DESC_ID') )              takes  0.64s  (   ..  )
                    #
                    # Looks like we have a winner!
                    spmap = reduce(reductor, numpy.unique( tbl.getcol('DATA_DESC_ID') ), {})
                # do not forget to sort all subbands by frequency
                sort_order = kwargs.get('spw_order', 'by_frequency').lower()

                if sort_order=='by_frequency':
                    sortfn = lambda x: x.frequency[0] #lambda x,y: cmp(x.frequency[0], y.frequency[0])
                elif sort_order=='by_id':
                    sortfn = lambda x: x.spWinId #lambda x,y: cmp(x.spWinId, y.spWinId)
                else:
                    raise RuntimeError("The spectral window ordering function {0} is unknown".format( kwargs.get('spw_order') ))
                return spectralmap( hvutil.dictmap( lambda kvpair : sorted(kvpair[1], key=sortfn), spmap) )



##
## The baseline mapping
##
class baselinemap:
    # antennalist is [(antenna, id), ...]
    # baselineleList = [(baselineId, baselineName),...]
    def __init__(self, antennalist, **kwargs):
        # keep a sorted list ourselves
        self.antennaList  = sorted(antennalist, key=operator.itemgetter(1))

        # Check if the 'baselines' were explicitly given. If not, we form
        # the baselines ourselves out of all antenna pairs
        # The baselines can be passed in as keyword-arg:
        #  ..., baselines=[(x,y), ...], ...
        # a list of antenna pairs
        bls = kwargs['baselines'] if 'baselines' in kwargs else None
        if not bls:
            # the entries in 'antennaList' are ('<name>', AntennaId) tuples!
            bls = [(x[1],y[1]) for (idx, x) in enumerate(self.antennaList) for y in self.antennaList[idx:]]
        # Now transform the list of indices into a list of names + codes
        self.baselineList = map_(lambda x_y: (x_y, "{0}{1}".format(self.antennaName(x_y[0]), self.antennaName(x_y[1]))), bls)

    def baselineNames(self):
        return map_(operator.itemgetter(1), self.baselineList)

    def baselineIndices(self):
        return map_(operator.itemgetter(0), self.baselineList)

    def baselineName(self, blidx):
        try:
            [(x,y)] = filter_(lambda idx_nm: idx_nm[0]==blidx, self.baselineList)
            return y
        except ValueError:
            raise InvalidBaselineId(blidx)

    def baselineIndex(self, blname):
        try:
            [(x,y)] = filter_(lambda idx_nm: idx_nm[1]==blname, self.baselineList)
            return x
        except ValueError:
            raise InvalidBaselineId(blname)

    def antennaNames(self):
        return map_(operator.itemgetter(0), self.antennaList)

    # return the list of (antenna, id) tuples, sorted by id
    def antennas(self):
        return sorted(self.antennaList, key=operator.itemgetter(1))

    def antennaName(self, id):
        try:
            [(nm, i)] = filter_(lambda ant_antid: ant_antid[1]==id, self.antennaList)
            return nm
        except ValueError:
            raise InvalidAntenna(id)

    def antennaId(self, name):
        try:
            namelower = name.lower()
            [(nm, id)] = filter_(lambda ant_antid: ant_antid[0].lower()==namelower, self.antennaList)
            return id
        except ValueError:
            raise InvalidAntenna(name)

def makeBaselineMap(nm, **kwargs):
    errf   = hvutil.mkerrf("makeBaselineMap({0})".format(nm))
    with opentable(nm) as tbl:
        with opentable( tbl.getkeyword('ANTENNA') ) as antab:
            if len(antab)==0:
                return errf("No rows in the ANTENNA table")

            # If we want to know only the antenna's that are 
            # actually *used* ....
            filter_f  = None
            baselines = None
            if 'unique' not in kwargs or not kwargs['unique']:
                filter_f = lambda x : True
            else:
                # compile the list of unique antenna-id's that are
                # actually used in the MS
                #
                # Again - we hit the ***** 'performance' of
                #         pyrap.tables.taql.
                #
                #  uniqry = lambda col:
                #             set(
                #               pyrap.tables.taql("SELECT {0} AS FOO from $tbl".format(col), locals={"tbl":tbl}).getcol("FOO")
                #             )
                #
                # All measurements taken on a 8.7Mrow (8.7e6 rows) MS
                #
                # uniqry( "ANTENNA1" )                takes 1.51s
                # tbl.getcol('ANTENNA1')              takes 0.36s
                # x = tbl.getcol('ANTENNA1')
                # numpy.unique( x )                   takes 0.23s
                # set(numpy.unique(x))                takes 0.24s
                #
                # Thus for getting the unique antennas it's fastests to
                # get the columns for ANTENNA1, ANTENNA2, create sets and union them
                #
                #
                #uniqry    = lambda col: set(pyrap.tables.taql("SELECT {0} AS FOO from $tbl".format(col), locals={"tbl":tbl}).getcol("FOO"))
                #ants      = uniqry("ANTENNA1") | uniqry("ANTENNA2")
                a1        = tbl.getcol('ANTENNA1')
                a2        = tbl.getcol('ANTENNA2')
                ants      = set(numpy.unique(a1)) | set(numpy.unique(a2))
                filter_f  = lambda x : x in ants
                # retrieve the uniqe baselines. This also takes a LOOOONG time
                # so we gonna do it mighty different.
                # since we already have ANTENNA1, ANTENNA2 columns we're going
                # to play a neat trick.
                # Using numpy we multiply antenna1 by 1000 and add antenna2
                # so we have an array of baseline codes (integers).
                # Then we uniquefy those and translate them back to
                # tuples with antenna indices
                maxant    = max(ants)+1
                make_tup  = lambda blcode: (blcode // maxant, blcode%maxant)
                baselines = map_(make_tup, numpy.unique(numpy.add(numpy.multiply(a1, maxant), a2)))

            names = antab.getcol('NAME')

            # MSv1 does not have 'ANTENNA_ID' so we'll make a
            # straight 0 -> n numbering scheme
            if 'ANTENNA_ID' in antab.colnames():
                ids = antab.getcol('ANTENNA_ID')
            else:
                ids = itertools.count()

            return baselinemap(filter(lambda n_i: filter_f(n_i[1]), zip(names, ids)), baselines=baselines)



##
## The fieldmap
##
class fieldmap:
    # the fieldmap = [(id, name),...]
    def __init__(self,fm):
        self.fieldMap = fm

    def fldextractor(self, n):
        return map_(operator.itemgetter(n), self.fieldMap)

    def getFields(self):
        return self.fldextractor(1)

    def getFieldIDs(self):
        return self.fldextractor(0)

    def nrFields(self):
        return len(self.fieldMap)

    # See comment above in the
    # polarizationmap.getPolarizations(self,id) method
    def field(self, id):
        try:
            # enforce unique search result otherwise bomb out
            [x] = filter_(lambda y: y[0]==id, self.fieldMap)
            return x[1]
        except ValueError:
            raise InvalidFieldId(id)
    def unfield(self, name):
        try:
            # enforce unique search result otherwise bomb out
            [x] = filter_(lambda y: y[1]==name, self.fieldMap)
            return x[0]
        except ValueError:
            raise InvalidFieldId(name)

def makeFieldMap(nm, **kwargs):
    errf   = hvutil.mkerrf("makeFieldMap({0})".format(nm))
    with opentable(nm) as tbl:
        with opentable(tbl.getkeyword('FIELD')) as fldtab:
            if len(fldtab)==0:
                return errf("No rows in FIELD table?!")
            if 'unique' not in kwargs or not kwargs['unique']:
                filter_f = lambda x: True
            else:
                # Build up a list of FIELD_IDs that are actually
                # *used* in the MS
                # slow method
                #    field_ids = set(pyrap.tables.taql("select unique FIELD_ID from $tbl", locals={"tbl":tbl}).getcol("FIELD_ID"))
                # fast method (not really, we make it even faster ... see other functions with timings)
                #field_ids = set(tbl.getcol("FIELD_ID"))
                field_ids = numpy.unique(tbl.getcol('FIELD_ID'))
                filter_f  = lambda fld_nm: fld_nm[0] in field_ids

            get_name = lambda x: x['NAME']
            return fieldmap(filter_(filter_f, zip(itertools.count(), map(get_name, fldtab))))


##
##  Time'server' - get timerange from a measurementset
##
def getTimeRange(nm, **kwargs):
    errf   = hvutil.mkerrf("getTimeRange({0})".format(nm))
    class startend:
        def __init__(self, st, en, ti):
            self.start = st
            self.end   = en
            self.inttm = ti
    with opentable(nm) as tbl:
        if len(tbl)==0:
            return errf("No rows in table!")

        # slow version ...
        #(start, end) = hvutil.minmax( opentable(nm).getcol('TIME') )
        # fast version! (well, apparently not so much!)
        #tms = pyrap.tables.taql("select unique TIME from $tbl ORDER BY TIME ASC", locals={"tbl":tbl}).getcol("TIME")

        # Timing showed that
        #    pyrap.tables.taql("select unique TIME ...") took 2.37s on a 8.7Mrow MS (that's 8.7e6 rows)
        #    pyrap.tables.taql("select TIME ...")        took 1.97s on that MS
        #    sorted( tbl.getcol('TIME') )                took 1.37s on that MS
        #    tbl.getcol('TIME')                          took 0.38s on that MS
        #    
        #    using numpy:
        #      tm = tbl.getcol('TIME')                   takes 0.38s
        #      numpy.amin( tm )                          takes 0.02s
        #      numpy.amax( tm )                          takes 0.02s
        #
        # Thus the fastest way to get the full time range of an MS is to read the
        # whole TIME column and go through the array *twice* [using numpy ...]
        # to find the minimum and the maximum separately.
        utms  = numpy.unique(tbl.getcol('TIME'))
        return startend(numpy.amin(utms), numpy.amax(utms), list(numpy.unique(tbl.getcol('EXPOSURE'))))


## Given a measurement set, analyze the ARRAY_ID, TIME,FIELD_ID and EXPOSURE columns
## to re-create a scan index of start/end time, field and array [and integration time
##  but that's not too important]
##
## 'indexr()' will return a list of scan objects

class scan(object):
    sortOrder = operator.attrgetter('start', 'array_id', 'field_id')
    def __init__(self, array, sn, field, stime, etime, tint):
        self.array_id    = array
        self.field_id    = field
        self.field       = ""
        self.start       = stime
        self.end         = etime
        self.t_int       = tint
        self.scan_number = sn

    @property
    def length(self):
        return self.end - self.start

    @property
    def mid(self):
        return self.start + self.length/2.0

    def __str__(self):
        (m, s) = divmod(self.length, 60.0)
        return "{0:3d}: {1} {2: 3d}m{3:.2f}s dT:{4:5.2f}s {5:<15s} ({6}) (ARRAY_ID {7})".format(
            self.scan_number, as_time(self.start), int(m), s, self.t_int, self.field, self.field_id, self.array_id)

## fudge is the gap time in number of integration times after which we decide
## there is a new scan [that is, if there is no change in field]. So a discontinuity
## in the TIME axis longer than "fudge * EXPOSURE" w/o change in field and/or array
## will start a new scan
def indexr(msname, **kwargs):
    rv = []
    ag = operator.itemgetter('SCAN_NUMBER', 'ARRAY_ID', 'FIELD_ID', 'EXPOSURE')
    with opentable(msname) as ms:
        ti = ms.iter(["ARRAY_ID", "SCAN_NUMBER", "FIELD_ID"])
        for tab in ti:
            (sn, aid, fld, exp) = ag(tab[0])
            tms                 = tab.getcol('TIME')
            rv.append( scan(aid, sn, fld, numpy.min(tms), numpy.max(tms), exp) )
        return rv


def indexr_heur(msname, fudge=2.1):
    ac = []
    ag = operator.itemgetter('ARRAY_ID', 'TIME', 'FIELD_ID', 'EXPOSURE')
    with opentable(msname) as ms:
        arr, tm, fld, exp = ag(ms[0])
        ac.append( scan(arr, fld, tm, exp) )
        ti = ms.iter(["ARRAY_ID", "TIME", "FIELD_ID", "EXPOSURE"])
        for tab in ti:
            arr, tm, fld, exp = ag(tab[0])
            delta = abs(fudge*exp)
            last  = ac[-1]
            if last.field_id!=fld or ((last.start-delta)>tm or (last.end+delta)<tm):
                # new scan
                ac.append( scan(arr, fld, tm, exp) )
            ac[-1].end = tm
        return sorted(ac, key=scan.sortOrder)

##  DataDomain stuff.
##  Provide for routine that attempts to
##  determine the domain of a given MS
knownColumns = { 'DATA':  jenums.Type.Spectral, 
        'LAG_DATA':       jenums.Type.Lag,
        'MODEL_DATA':     jenums.Type.Spectral,
        'CORRECTED_DATA': jenums.Type.Spectral }

def getDataDomain(ms, **kwargs):
    with opentable(ms) as tbl:
        colnames   = tbl.colnames()

        # Check which data-column is requested, if any
        column     = kwargs.get('column', None)

        # Depending on the method, select the data column candidate(s)
        candidates = ['DATA', 'LAG_DATA'] if column is None else [column.upper()]
        def reductor(acc, col):
            return col if col in colnames else acc
        thecolumn  = reduce(reductor, candidates, None)

        if thecolumn is None:
            if len(candidates)>1:
                raise RuntimeError("None of the columns {0} available in the MS".format( candidates ))
            else:
                raise RuntimeError("The column {0} is not available in the MS".format( candidates[0] ))

        # return an object with attributes .domain and .column
        return type('',(),{'domain': knownColumns.get(thecolumn, jenums.Type.Unknown), 'column':thecolumn})()

## Return the project code from the MS
## Currently only the first one is returned
## Let's make it an assertion that there's
## only one [we'll find out soon enough
## if it isn't and then figure out how to
## deal with it]
def getProject(ms, **kwargs):
    errf   = hvutil.mkerrf("getProject({0})".format(ms))
    with opentable(ms) as tbl:
        with opentable(tbl.getkeyword('OBSERVATION')) as obstab:
            # assert that all project codes lists are equal ...
            try:
                [proj] = list(set(obstab.getcol('PROJECT')))
                return proj
            except ValueError:
                return errf("Multiple project codes found!")

## A function returning a nice progress update including a bar + percentage
##    's', 'e' are the lower/upper limit of the range, 'cur' where you currently are
def progress(cur, s, e, sz, pfx="Progress"):
    d = float(e)-float(s)
    frac = (float(cur - s)/d) if d>0 else 0
    pos  = int( frac * sz )
    if pos==0:
        pos = 1
    return pfx+" |" + "="*(pos-1) + ">" + " "*(sz-pos)+"| {0:6.2f}%".format(frac*100)

## A function which returns a column slicer function and possibly integrates a function
## to be called on the data [e.g. "numpy.abs", "numpy.real" ... or something of your own]
def mk_slicer(blc, trc, fn=None):
    # we can already create the slice
    # 14 Feb 2019: got warning:
    #
    # /home/verkout/jiveplot/ms2util.py:1238: FutureWarning: Using a non-tuple
    # sequence for multidimensional indexing is deprecated; use
    # `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be
    # interpreted as an array index, `arr[np.array(seq)]`, which will result
    # either in an error or a different result.
    # return lambda tab, col, s, n: tab.getcol(col, startrow=s, nrow=n)[indarr]
    indarr = tuple([Ellipsis]+list(map(lambda st_en: slice(st_en[0], st_en[1]), zip(blc, trc))))

    if fn:
        return lambda tab, col, s, n: fn(tab.getcol(col, startrow=s, nrow=n))[indarr]
    else:
        return lambda tab, col, s, n: tab.getcol(col, startrow=s, nrow=n)[indarr]

def mk_processor(fn=None):
    if fn:
        return lambda tab, col, s, n: fn(tab.getcol(col, startrow=s, nrow=n))
    else:
        return lambda tab, col, s, n: tab.getcol(col, startrow=s, nrow=n)

## Reduce a MeasurementSet. For each row in the table the function 'function' is
## called with the accumulator and a tuple, made out of the cell values of your selected columns.
## The columns and the order in which they are passed are dictated by the "columns"
## argument - a list of MS columns you'd like to retrieve.
##
## The iteration is split into chunks of 'chunksize' rows, so as not to have to read
## the whole table in one go but it should be transparent to 'function'.
##
## The default 'columngetter' is "ms.getcol()".
## But sometimes, when dealing with an array-valued
## column, you don't want to read the whole column in but just
## a channel or a few channels. Then
## It's possible to use the 'mk_slicer' (see above).
## Pass it in via the 'slicers=' keyword arg to 'reducems'.
##   'slicers=' is taken to be a dict of
##      slicers [ <columnname> ] = fun( table, colnm, startrow, nrow )
## You can provide your own column slicer if you wish, as long as it adheres 
## to the prototype "fun(table, columnname, startrow, nrow)"
now = datetime.datetime.now
def reducems(function, ms, init, columns, **kwargs):
    defaults  = { 'verbose': len(ms)>500000, 'slicers': {}, 'chunksize':5000 }
    defaults.update(kwargs)
    slicers   = defaults['slicers']
    chunksize = defaults['chunksize']
    # default column getter
    # allow user to override for specific column (pass function/4 in via kwargs)
    fn    = mk_processor()
    fns   = map(lambda col: (col, slicers.setdefault(col, fn)), columns)
    def log(msg):
        sys.stdout.write(msg+" "*10+"\r")
        sys.stdout.flush()
    logfn = log if defaults['verbose'] else lambda x: None
    i     = 0
    mslen = len(ms)
    s = now()
    while i<mslen:
        cs   = min(chunksize, mslen-i)  # chunksize
        logfn(progress(i, 0, mslen, 50))
        init = reduce(function, itertools.izip( *map_(lambda c_f: f(ms, c_f[0], i, cs), fns)), init)
        i    = i + cs
    e = now()
    logfn(" "*60)
    if defaults['verbose']:
        print("reducems took ", (e-s))
    return init

def chunkert(f, l, cs, verbose=True):
    def log(msg):
        sys.stdout.write(msg+" "*10+"\r")
        sys.stdout.flush()
    logfn = log if verbose else lambda x: None
    while f<l:
        n = min(cs, l-f)
        logfn(progress(f, 0, l, 50))
        yield (f, n)
        f = f + n
    logfn(progress(f, 0, l, 50))
    logfn(" "*80)
    raise StopIteration

## Bare-bones reduce ms. No fancy progress display
def reducems_raw(function, ms, init, columns, **kwargs):
    # allow user to override for specific column (pass function/4 in via 'slicer' kwarg)
    slicers = kwargs.get('slicers', {})
    chunksz = kwargs.get('chunksize', 5000)
    fns     = map(lambda col: (col, slicers.get(col, lambda tab, c, s, n: tab.getcol(c, startrow=s, nrow=n))), columns)
    return reduce(lambda acc, i_cs:\
                    reduce(function, itertools.izip( *map_(lambda c_f: f(ms, c_f[0], i_cs[0], i_cs[1]), fns)), acc),
                  chunkert(0, len(ms), chunksz), init)

## reducems calls function as:
##     function(acc, col1, col2, col3, ..., coln)
## With 'colN' having the column data for a block of columns
## so 'function' must iterate over the elements itself
def reducems2(function, ms, init, columns, **kwargs):
    # allow user to override for specific column (pass function/4 in via 'slicer' kwarg)
    slicers = kwargs.get('slicers', {})
    chunksz = kwargs.get('chunksize', 5000)
    fns     = map(lambda col: (col, slicers.get(col, lambda tab, c, s, n: tab.getcol(c, startrow=s, nrow=n))), columns)
    return reduce(lambda acc, i_cs: function(acc, *map_(lambda c_f: c_f[1](ms, c_f[0], i_cs[0], i_cs[1]), fns)),
                  chunkert(0, len(ms), chunksz, verbose=kwargs.get('verbose', False)), init)
