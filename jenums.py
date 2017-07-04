# $Id: jenums.py,v 1.5 2017-02-21 09:10:05 jive_cc Exp $
#
# $Log: jenums.py,v $
# Revision 1.5  2017-02-21 09:10:05  jive_cc
# HV: * DesS requests normalized vector averaging - complex numbers are first
#       normalized before being averaged. See "help avt" or "help avc".
#
# Revision 1.4  2013-03-31 17:17:56  jive_cc
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
#
# Personally, I like enums (or atoms) for readability
# (atoms are a concept from Erlang who got them from Prolog)
# they're like enums only infinitely better

# http://stackoverflow.com/questions/36932/whats-the-best-way-to-implement-an-enum-in-python
#def enum(*sequential, **named):
      #enums = dict(zip(sequential, range(len(sequential))), **named)
      #return type('Enum', (), enums)
class enum(object):
    def __init__(self, *seq):
        self.enums = seq
        for e in self.enums:
            setattr(self, e, e)

    # you can iterate over the enum to find out all defined values
    def __iter__(self):
        class enumiter(object):
            def __init__(self,enuminst):
                self.iterable = enuminst
                self.iter     = iter(enuminst.enums)
            def next(self):
                return getattr(self.iterable, self.iter.next())
        return enumiter(self)

    def __getitem__(self, idx):
        if idx in self.enums:
            return idx
        raise IndexError,"{0} does not exist".format(idx)

Type      = enum('Unknown', 'Lag', 'Spectral')
Axes      = enum('P', 'CH', 'SB', 'FQ', 'BL', 'SRC', 'TIME', 'TYPE')
Averaging = enum('None', 'Scalar', 'Vector', 'Vectornorm')
