# name really sais it all: a grab-bag of potentially useful stuff
# $Id: hvutil.py,v 1.16 2015-09-23 12:25:48 jive_cc Exp $
# $Log: hvutil.py,v $
# Revision 1.16  2015-09-23 12:25:48  jive_cc
# HV: * add method 'split_optarg' for simple-minded "command line parsing".
#       Splits its "*args" (all strings) in two parts: arguments and options
#       where options are defined to be strings matching 'key=value'.
#       It provides the possibility to specify a check/convert utility to
#       transform <value> to a desired type.
#
# Revision 1.15  2015-09-21 11:36:22  jive_cc
# HV: * The big push. The 'command.py' interface long left something
#       to be desired if a command specified an 'args()' method to transform
#       the line of input into multiple arguments and call the callback function
#       with those arguments [hint: it wouldn't].
#       Now the command.py object calls the callback as:
#         "cmd.callback( *cmd.args(input) )"
#     * all commands have been visited and modified to be specified as
#          "def xyz( .., *args)"
#       and do the right thing if called with wrong number of arguments.
#       Many command's regex's got simpler.
#     * fixed some error messages to be more readable
#     * extended the documentation of x, y scaling of plots ("xyscale")
#     * clear error message if adressing an invalid panel in a plot
#     * the 'new plot' specification was widened: can now use comma-separated
#       list of axes to set the new plot value in one go:  "new p,sb true"
#     * the "cd" command now does '~' expansion at the beginning of the path
#     * fixed bug in pgplot device name generation in case details were left out
#
# Revision 1.14  2014-04-23 14:16:39  jive_cc
# HV: * added generalized version of "quote_split" which allows splitting
#       based on a separator char unless it is within a quoted section where
#       the open/close markers are arguments to the function
#
# Revision 1.13  2014-04-15 12:21:56  jive_cc
# HV: * pagelabel is now centralized computed in the base class
#       we now have maximum consistency between all plots
#
# Revision 1.12  2014-04-08 22:41:10  jive_cc
# HV: Finally! This might be release 0.1!
#     * python based plot iteration now has tolerable speed
#       (need to test on 8M row MS though)
#     * added quite a few plot types, simplified plotters
#       (plotiterators need a round of moving common functionality
#        into base class)
#     * added generic X/Y plotter
#
# Revision 1.11  2014-04-02 17:55:29  jive_cc
# HV: * another savegame, this time with basic plotiteration done in Python
#
# Revision 1.10  2013-12-12 14:10:15  jive_cc
# HV: * another savegame. Now going with pythonic based plotiterator,
#       built around ms2util.reducems
#
# Revision 1.9  2013-09-03 17:34:30  jive_cc
# HV: * Amazing! All plot types do work now. Can save plots to file.
#       Still some details left to do obviously ...
#
# Revision 1.8  2013-06-19 12:28:43  jive_cc
# HV: * making another savegame
#
# Revision 1.7  2013-04-02 16:35:21  jive_cc
# HV: * quote_split() moved to command.py (presumably mostly useful there)
#     * added cycle_detection algorithm to find cycles inna graph
#
# Revision 1.6  2013-03-31 17:17:56  jive_cc
# HV: * another savegame
#
# Revision 1.5  2013-03-09 16:59:07  jive_cc
# HV: * another savegame
#
# Revision 1.4  2013-02-19 16:53:29  jive_cc
# HV: * About time to commit - make sure all edits are safeguarded.
#       Making good progress. baselineselection, sourceselection and
#       timeselection working
#
# Revision 1.3  2013-02-11 09:40:33  jive_cc
# HV: * saving work done so far
#       - almost all mapping-functionality is in place
#       - some of the API functions are starting to appear
#
# Revision 1.2  2013-01-29 12:23:44  jive_cc
# HV: * time to commit - added some more basic stuff

# sys imports
from   __future__ import print_function

import re
import copy
import math
import string
import datetime
import operator
import itertools

from   six        import iteritems
from   functools  import (reduce, partial)

# own stuff
from .functional  import (map_, filter_, zip_, enumerate_, range_, is_not_none, drap)

## Partition a list into two lists - one with the elements satisfying the predicate
## and one with the elements who don't
def partition(pred, lst):
    y, n = (list(), list())
    for elem in lst:
        y.append(elem) if pred(elem) else n.append(elem)
    return (y, n)

# map a function over all the values in the dict
# {k:v} => {k:f(v)}
def dictmap(f, d):
    return dict((k, f((k,v))) for (k,v) in iteritems(d))

# reduce a dict to one value:
#  acc = f( (k1, v1), f( (k0,v0), acc0 ) )  etc
# "f" is called as: f(elem, acc)
# and should return the new accumulator
def dictfold(f, a, d):
    for x in iteritems(d):
        a = f(x, a)
    return a

def mkerrf(pfx):
    def actualerrf(msg):
        raise RuntimeError("{0} {1}".format(pfx, msg))
    return actualerrf

## Return the minimum + maximum of a sequence in one go.
## Note: the sequence must have at least 1 element
def minmax(seq):
    mi,ma = (None, None)
    for item in seq:
        mi = min(mi, item)
        ma = max(ma, item)
    return (mi, ma)
    #return reduce(lambda (mi,ma) ,y : (min(mi,y), max(ma,y)), seq[1:], (seq[0], seq[0])) if len(seq) else (None, None)

## Round floating point number to n decimal places
def round(n):
    factor = 10**n
    return lambda x: math.floor( x * factor )/factor

## Determines if a is a subset of b
def contains(a, b):
    return set(a).issubset(set(b))

## enumerate the items in iterable and
## yield the slice of [first, last]
def enumerateslice(iterable, first, last):
    return itertools.takewhile(lambda i_v: i_v[0]<last, itertools.dropwhile(lambda i_v: i_v[0]<first, enumerate(iterable)))

## Convert fractional day into datetime.time
def fractionalDayToTime(frac):
    ofrac = frac
    (frac, h) = math.modf(frac*24)
    (frac, m) = math.modf(frac*60)
    (frac, s) = math.modf(frac*60)
    return datetime.time(h, m, s, int(frac*1.0e6))

## Convert time in seconds-since-start-of-day into datetime.time
def secondsInDayToTime(sssod):
    # 3600 seconds per hour
    (h, s)  = divmod(sssod, 3600)
    (m, s)  = divmod(s,       60)
    (fs, s) = math.modf( s )
    return datetime.time(int(h), int(m), int(s), int(fs*1.0e6))

## Generalized version of 'quote_split' as found in command.py
## This one defaults to "'" for opening and "'" for closing quotes
## but these characters can be overridden
##
## quote_split  do not extract _words_ but split the
##              string at the indicated character,
##              provided the character is not inside
##              quotes
##
##      "aap  'noot 1; mies 2'; foo bar"
##  results in:
##      ["aap 'noot 1;mies 2'", "foo bar"]
def quote_split(s, splitchar=';', quotes="'"):
    q = list(quotes)
    if len(q)==1:
        q.append( q[0] )
    if len(q)!=2:
        raise RuntimeError("Quotes must be an array of two length")
    rv = [""]

    # switch quotiness if we see quotes[switch]
    switch  = 0
    inquote = False
    for i in range_(len(s)):
        if not inquote and s[i]==splitchar:
            rv.append( "" )
            continue
        rv[-1] = rv[-1] + s[i]
        if s[i]==q[switch]:
            inquote = not inquote
            switch  = 1 - switch
    return rv

## itemgetter: differs wrt operator.itemgetter (standard Python version) in that
## this one _always_ yields a tuple, no matter how many arguments were given.
## The built-in itemgetter does, depending on the amount of arguments, the following:
##
##   #of args to "operator.itemgetter()"   action
##     0                                    Crash! ('list indices must must be integers, not list')
##     1                                    return an object
##     2 or more                            return a tuple with the addressed objects
##
## This version:
##
##     n                                    return a tuple with the n addressed objects
##                                            irrespective of n==0, n==1,... n==42, ...
itemgetter = lambda *args: lambda thing: tuple([thing[x] for x in args])
attrgetter = lambda *args: lambda thing: tuple([getattr(thing, x) for x in args])


## Finds consecutive ranges in the sequence
## returns a list of (start, end) values
## of ranges start ... end (the end is inclusive)
## of which the sequence was made of
## Based on
## http://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
def find_consecutive_ranges(seq):
    # groupby() returns [(key, group)]
    #   where group = iterable sequence
    #   of [(idx, item)] (because that's
    #   what we gave to the groupby() -
    #   we enumerated our original items!
    # so after the groupby phase we
    #  first extract all the groups (we don't care
    #  about the actual grouping key)
    #  by mapping the itemgetter over the list of groups
    # then we're left with an iterable which contains iterable
    # sequences of which we need to find the first and last item
    # Apparently there can be instances where the groupby returns
    # an empty list so we need to take care of that and
    # filter those non-results out
    def maker( key_seq ):
        # undo the enumeration we added to allow
        # identification of consecutive elements
        l = map_(operator.itemgetter(1), key_seq[1])
        return (l[0], l[-1]) if l else None
    return filter_(is_not_none,
                   map(maker,
                       itertools.groupby(enumerate(sorted(seq)), lambda i_x:i_x[0]-i_x[1])))

## After having found consecutive ranges, you might want a string representation
##  [(start, end), ...] =>
##     "<start>"  (if start==end)
##     "<start>-<end>" otherwise
def range_repr(l, rchar=":"):
    return ",".join(map(lambda s_e: "{0}".format(s_e[0]) if s_e[0]==s_e[1] else "{0}{2}{1}".format(s_e[0],s_e[1],rchar) if abs(s_e[0]-s_e[1])>1 else "{0},{1}".format(s_e[0],s_e[1]), l))


## Does the inverse of the previous one:
## expands string "1:10,13,20:22" into [1,2,3,4,5,6,7,8,9,10,13,20,21,22]
##   and "5:2" into [5,4,3,2]
##
## Supports arbitrary increments
##    1:10:2 => [1,3,5,7,9]
## Support arithmetic:
##  "2*10:3*10,12-2:12+2"
##  All expressions will be converted to int after evaluating
def expand_string_range(s, rchar=":"):
    rxNum = re.compile(r"^\d+$")
    rxRng = re.compile(r"^(?P<s>[-\d\.+*/%()]+)"+rchar+r"(?P<e>[-\d\.+*/%()]+)(:(?P<step>[-+]?\d+))?$")
    def count_from_to(s,e,step):
        while abs(s-e)>=abs(step):
            #print "s:{0} e:{1} diff:{2}".format(s, e, abs(s-e))
            yield s
            s = s + step
        if abs(s-e)<=abs(step):
            yield s
        return
    def mkcounter(item):
        mo = rxRng.match(item)
        if mo:
            # start, end may be expressions
            (s,e) = (int(eval(mo.group('s'))), int(eval(mo.group('e'))))
            defstep = 1 if (s<e) else -1
            step    = mo.group('step')
            step    = int(step) if step else defstep
            # Test if we actually *can* count from start -> end using step:
            # e.g.:    1 -> 10, step -1 isn't going to end very well is it?
            #         -1 -> -10, step 1         ,,           ,,
            # Step size "0" is ONLY allowed if start==end!
            # Also assure ourselves that the step direction and counting
            # direction are identical
            if (not step and (s-e)) or (((e-s) * step )<0):
                raise RuntimeError("cannot count from {0} to {1} with step {2}".format(s, e, step))
            return count_from_to(s, e, step)
        else:
            mo = rxNum.match(str(eval(item)))
            if not mo:
                raise ValueError("{0} is not a number! (It's a free man!)".format(item))
            # 'item' may be an expression!
            item = int(eval(item))
            # Note: seems superfluous to return a counter for 1 number but now
            # we have a list of iterables which can easily be transformed into
            # one list via itertools.chain() (see below)
            return count_from_to(item, item, 1)
    return list(itertools.chain(*[mkcounter(x) for x in s.split(",")]))


##
## String processing utilities
##

## take a list of (pattern, replacement) tuples and run them
## over the string to produce the final edited string
subber   = lambda acc, pat_repl: re.sub(pat_repl[0], pat_repl[1], acc)
sub      = lambda txt, lst: reduce(subber, lst, txt)

# Return a predicate function which will keep returning True
# until n occurrences of the character 'c' have been seen.
# Can be used e.g. conjunction with itertools.takewhile()
def untilnth(n, char):
    # we need to creat an in-place modifyable 'thing'
    # orelse a "n=n-1" in the match() inner fn
    # will give an UnboundLocalError exception:
    # http://eli.thegreenplace.net/2011/05/15/understanding-unboundlocalerror-in-python/
    o = type('',(),{"n":n})
    def match(c):
        if c==char:
            o.n = o.n - 1
        if o.n<=0:
            return False
        return True
    return match

## return the leading part of the string up until the n'th
## occurrence of 'char'
## This is 'better' than s.split(sep=char)[0:n]
## because you will have lost the actual sep characters!
## (could do "char.join(s.split(sep=char)[0:n])" but then
## you're introducing elements in the string yourself ...
## (you don't know now many "split()" has taken out!
def strcut(s, n, char):
    return ''.join(itertools.takewhile(untilnth(n, char), s))


## escape split - split a string at whitespace,
## honouring the escape character (single quote)
##   aap 'noot mies'    is OK
##   aap  'no\'ot mies' is OK
##
## Note: the quoted string MUST be surrounded by whitespace:
##   aap  no'oot mies'  is NOT OK
##   aap  'noot''mies'  is NOT OK
#rxSplit          = re.compile(r"('.*?(?<!\\)'|\S+)")
rxSplit          = re.compile(r"('.*?(?<!\\)'(?=\s|$)|\S+)")
rxUnescapedQuote = re.compile(r"(?<!\\)'")
stripEscapeChars = partial(re.sub, r"\\", r"")
def escape_split(s):
    # three stage process:
    #   (1) find all quoted string + whitespace separated words
    #   (2) detect errors - if there are non-escaped quote(s) in
    #       a word dat's an error
    #   (3) strip all escape sequences

    # (1) find a list of strings and strip the leading+trailing quote
    #     if any
    words  = map_(lambda x: x[1:-1] if x[0]=="'" else x, rxSplit.findall(s))

    # (2) detect syntax errors like "n'oot" "aap 'noot''mies'"
    if any(map(rxUnescapedQuote.search, words)):
        raise SyntaxError("quoted string error - either no whitespace between " \
                          "quoted elements or an unescaped quote in the middle " \
                          "of a word")

    # (3) strip the escape character(s)
    return map_(stripEscapeChars, words)


## Flatten a list of lists
## http://grokbase.com/t/python/tutor/051c5rgj5s/flattening-a-list
def flatten(a):
    """Flatten a list."""
    return bounce(flatten_k(a, lambda x: x))


def bounce(thing):
    """Bounce the 'thing' until it stops being a callable."""
    while callable(thing):
        thing = thing()
    return thing


def flatten_k(a, k):
    """CPS/trampolined version of the flatten function.  The original
    function, before the CPS transform, looked like this:

    def flatten(a):
        if not isinstance(a,(tuple,list)): return [a]
        if len(a)==0: return []
        return flatten(a[0])+flatten(a[1:])

    The following code is not meant for human consumption.
    """
    if not isinstance(a,(tuple,list)):
        return lambda: k([a])
    if len(a)==0:
        return lambda: k([])
    def k1(v1):
        def k2(v2):
            return lambda: k(v1 + v2)
        return lambda: flatten_k(a[1:], k2)
    return lambda: flatten_k(a[0], k1)


## cycle_detect: Return a list of all cycles in the given graph
##    Uses depth-first search and a stack of 'seen' nodes to detect
##    if we cross a node we've seen before.
##    graph = { "n":["n1","n2"], "m":["n3","n4"], ....}

# helper function: the caching depth-first-search
def dfs_seen(node, graph, seen, cycles):
    if node in seen:
        cycles.add(frozenset(seen[seen.index(node):]))
    else:
        seen.append(node)
        if node in graph:
            # drain + map() - we not interested in the return value of the map()
            drap(lambda x: dfs_seen(x, graph, copy.deepcopy(seen), cycles), graph[node])
    return cycles

def cycle_detect(graph):
    # for each starting point, start with a fresh 'seen' and collect all cycles
    # IF there is a cycle with 'n' nodes in it, we would end up with
    # 'n' entries of the same "seen" set, therefore our accumulation is a set:
    # it will automatically weed out the duplicates
    return list(reduce(lambda acc, k: dfs_seen(k, graph, list(), acc), graph, set()))


##
## Simple-minded 'command (line)' parsing
## by separating list of arguments into those matching "<key>=<value>" from
## those that don't. For those that do match this pattern, the match-object
## containing two groups - 'key' and 'value' - is returned.
##
## "split_optarg(*args)" => (arguments, options)
##
##  arguments = list of strings, elements in *args that did not match 'k=v'
##  options   = dict of {key:value} pairs parsed from *args
##
## Note that one or both lists could be empty, depending on the contents of *args
rxKeyValue = re.compile(r"^(?P<key>[^=]+)=(?P<value>.+)$")
identity   = lambda x: x
def mo_or_org(x, **convdict):
    mo = rxKeyValue.match(x)
    return (mo.group('key'), convdict.get(mo.group('key'), identity)(mo.group('value'))) if mo else x

def split_optarg(*args, **convdict):
    (a, o) = partition(lambda x: isinstance(x, str), map(lambda y: mo_or_org(y, **convdict), args))
    return (a, dict(o))


# Simple+slow uniq/unique function that retains order
# cf. https://stackoverflow.com/a/37163210
def uniq(l):
    used = set()
    return [x for x in l if x not in used and (used.add(x) or True)]
