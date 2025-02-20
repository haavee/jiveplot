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
from . import enumerations

Type      = enumerations.Enum('Unknown', 'Lag', 'Spectral')
Axes      = enumerations.Enum('P', 'CH', 'SB', 'FQ', 'BL', 'SRC', 'TIME', 'TYPE')
Averaging = enumerations.Enum('NoAveraging', 'Scalar', 'Vector', 'Vectornorm', 'Sum', 'Vectorsum')
Flagstuff = enumerations.Enum('Unflagged', 'Flagged', 'Both')
Symbol    = enumerations.Enum("Unflagged", "Flagged", "Marked", "Markedflagged")
