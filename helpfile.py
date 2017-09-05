## help texts for the commands
Help = {
    ##################################################################
    # ms
    ##################################################################
	"ms":
"""ms [msname [options]]
    open MS/display current MS
    
If a MeaurementSet (MS) is opened, the code initializes all mappings and
analyzes the frequency setup. After having opened an MS you can proceed with
selecting data and operating on it.

The command supports the following options:

    column=<data column name>
                        use column '<data column name>' to load data from. By
                        default the code auto-selects between 'DATA' and 'LAG_DATA'

    spw_order=<reordering method>
                        using this option it is possible to set the spectral window
                        (re)ordering method. See the 'r'(ange) command for background.

                        Recognized values for <reordering method> are:

                            by_frequency    re-order spectral windows by
                                            frequency. this is the default
                            by_id           do not reorder the spectral windows,
                                            present them in the order they
                                            appear in the spectral_window table

    unique=t(rue)|f(alse)
                        override the global setting of 'unique'; influences
                        which meta data to read (see the help of the 'uniq'
                        command)

Example:

    > ms X3c1.ms unique=f spw_order=by_id column=alma_phase_corr

""",
    
    ##################################################################
    # indexr
    ##################################################################
    "indexr":
"""indexr
    run indexr to create scan list of current MS

Sometimes it is handy to be able to select data from your MS by 'scan' rather
than by time range or bluntly by source. After having run 'indexr' on the
current opened MS you can use the scan indices to select a time-range and
source.

Inspect the list of scans using 'r scan' (see the 'r' command ('r'='range')).
It was decided to not include the scan list in the default output because it
can be quite a long list.

A scan, as defined by this software, is a time-continuous range where the same
(sub)array of telescopes (ARRAY_ID) is observing the same source (FIELD_ID).

Note that no attempt is being made yet to deal with observing using different
modes (FREQ_GROUP, via DATA_DESCRIPTION_ID and SPECTRAL_WINDOW) at the same
time. We assume that the same ARRAY_ID, at the same TIME cannot be observing at
more than one FREQ_GROUP (setup).

Should this be a problem, contact verkouter@jive.nl and request it. You might
get lucky.
""",

    ##################################################################
    # scan
    ##################################################################
    "scan":
"""scan [simple-syntax|complicated-syntax]
    select or display scan selection

After having run 'indexr' it is possible to select entire scan(s) (using
'simple-syntax') or a specific part of a set of scan(s), based on their
properties using 'complex-syntax'.


==================== simple-syntax ===========================

Selecting whole scan(s) or scan ranges is simple; use the following syntax:

    > scan [id|id-id|id-id:step]* 
    
'id' is the scan number as printed by 'listr' or 'r scan'. Ranges are inclusive
of end points.

    Example:

    > scan 10-12 20 30-40:2

==================== complex-syntax ===========================

Sometimes it is useful to select only a part of a scan - say if you want to
time average some data but do not want to have to integrate the whole scan. Or
sometimes you want to select scans based on other properties than just their
number. For those applications the complex-syntax offers an SQL like selection
mechanism. Mostly useful in conjunction with 'solint' equal to none (see
'solint') command.

The general idea of the complex-syntax is to select a time range from a set of
scans, optionally satisfying some condition(s):

    time-expression 'to' time-expression ['where' cond ['order by' ...] [limit ...]]

The "time 'to' time" expression will be evaluated for all scans satisfying the
conditions, to end up with a list of time ranges selected from the MS, if
succesful.

The first time-expression represents the start time of the desired data range
out of a matching scan, the second one, obviously, the end time.

If no 'where' conditions are specified ALL scans match.

time-expression
---------------

A 'time-expression' can be an arithmetic expression involving symbolic values
(explained later), absolute time stamps (less useful in this context but still
supported) and durations (very useful in this context).

Within a 'time-expression' the symbols 'start', 'mid' and 'end' are available
and they represent the respective values of the matching scans.
'time-expressions' are simple arithmetic expression, e.g.:

    "start + 10" or "end - (2 * (end-start)/4)"

When dealing with time, numbers are not necessarily the best choice. For your
convenience, durations are supported. They allow the specification of offsets
in a more natural way. Durations are integer numbers followed by time units of
'd'(ays) 'h'(ours) 'm'(inutes) 's'(econds). Seconds have an optional sub-second
part.

    "3d2h22m" for a duration of 3 days, 2 hours and 22 minutes
    "0.2s" 

The fields in a duration are optional but the order is important - a "higher"
unit of time MUST come before a lower one. So "3d10.2s" is valid but "10s2m"
isn't.

Combining the symbolic values and the durations it become very intuitive to
write e.g. the following:

    start + 22.5s to end - 1m22s

to select this time range out of each matching scan. A more useful example
would be to e.g. select the 10 seconds in the middle of the desired scans:

    mid - 5s to mid + 5s

Or to skip the first and last 10 seconds of a scan:

    start + 10s to end - 10s


The 'where' clause
------------------

Each scan has a number of attributes, most of them self-explaining:

    start           the scan start time
    end             its end time
    mid             the mid-point ((end - start) / 2)
    length          the scan's length in seconds
    t_int           the integration time as listed in the MS (float)
    scan_number     the value of the SCAN_NUMBER column for this scan (int)
    field_id        the value of the FIELD_ID column for this scan (int)
    field           the name of the source (unmapped FIELD_ID value) (string)
    array_id        the value of the ARRAY_ID column for this scan (int)

In the 'where' clause you can select scans based on expressions involving these
attributes. Logic operators 'and' and/or 'or' and parenthesized expressions are
also supported:

    > .... where t_int < 10 and (scan_number = 20 or scan_number = 30)

Besides the standard relational operators ("<", "<=", "=", ">" and ">=") the
system implements the 'in' and '~' operators.

The 'in' operator allows selecting scan where an attribute's value is contained
in a list of values:

    <attribute> 'in' [ value {,value} ]

Example:

    > .... where field_id in [0, 13, 20-30:2]

The '~' operator is specifically meant for string-valued fields, like the
source name. The '~' means "string compare" rather than arithmetically compare.
The '~' operator supports two types of operands, <string> or <regex>. The
<regex> allows you full regular expression support, e.g. case insensitive
mathing and partial pattern matching:

    <attribute> '~' <string> | <regex>

    <string> may be a (single)quoted string like 'string' or just characters.
      The string match will be case sensitive. 'string' may contain the shell
      wildcard characters '*' or '?' and their interpretation and effect is
      identical to shell wild card matching/expansion.
      When using wildcards be sure to use the quoted string "'pattern'" version 
      or else the expression parser will interpret the '*' as arithmetic
      multiplication.

    <regex> is specfied as sed(2) regex: "/pattern/{flags}". Currently the only
      supported flag is "i" to make it a case-insensitive pattern.

Examples:

The following two conditions are _almost_ identical; the second version
will match all source names starting with 'm15' as well as those starting
with 'M15'. The first one only those starting with 'M15'.

    > .... where field ~ 'M15*' 
    > .... where field ~ /m15.*/i

Full regular expression support:
    > .... where field ~ /(m15|3c)[0-9]*/i


In all of the condition expressions a wider format of time formats is supported
than just duration (see previous section). When selecting based on time the
same time formats as under the 'time' command are feasible: full-fledged time
stamps and time stamps on days relative to the reference day of the experiment
(day 0 of the experiment).

    > .... where start>14-jun-2013/12h43m0s 
    > .... where end <= 0/23h30m30s

""",

    ##################################################################
    # uniq
    ##################################################################
    "uniq":
"""uniq [01tf]
    Analyze MS for unique meta data or just load the subtables.

A MeasurementSet carries a lot of meta data in subtables, like the ANTENNA
table for antenna information or the SPECTRAL_WINDOW table for frequency
information.

The presence of data in these subtables has no relation to having any rows in
the MAIN table referring to it; the ANTENNA table could contain the full list
of known stations whilst in the MAIN table only data from three stations is
present. This means that you could get the idea that there's a hundred antennae
worth of data in your MS, just looking at all the available antenna ID's. This
in turn could lead to unexpected effects when selecting data only to find out
that "your selection is empty".

jiveplot offers you a choice of which meta data to display/allow you to select
from (see the "r" command - the 'range' of selectables). 

If 'uniq' is True, upon opening of the MS the MAIN table will be analyzed and
only the meta data for stations, frequencies, sources that are actually
referred will be loaded and allowed for selection.

If you would rather just load the full meta data tables, set 'uniq' to False
and all *possible* meta data is loaded and available for selection. Again -
wether there's actual data in (some of) the selection is pot luck.

On (very) large MSs (>10Million rows) the analysis will take a measurable
(though hopefully still short) time. This may be an argument to set 'uniq' to
False for quick opening of MSs.

Changing the setting will not take effect until a new MS is opened.

""",

    ##################################################################
    # uniq
    ##################################################################
	"taql":
"""taql [TaQL Query|none]
    display/edit raw TaQL query
    
Without arguments shows the TaQL query representing the current selection. 

Use the 'command' version with extreme caution - it will overwrite the actual
current TaQL query without altering the actual selection. It may be useful
though to be able just to run your own queries, as long as you know what to
expect (or what not to expect).

""",

    ##################################################################
    # r
    ##################################################################
	"r":
"""r (bl|src|time|fq|ch|p|sb|ant)*
   display range of selectables in current MS
   
This command displays the meta-data found in the MS. Without arguments an
overview of all meta data is given, with the exception of baselines. With one
or more arguments the command displays the range of the indicated selectable
meta data.

Baselines can only be displayed if specifically asked for ("r bl"). Most
entries speak for themselves; think of the antenna list, the source list or the
time range of the experiment.

One entry is significantly less directly related to how information is stored
and labelled in the MS: the spectral/frequency information.

This is because the labelling the MS uses is very different from how scientists
think about their frequency setups. This code re-organizes the frequency
meta-data and presents it in a fashion that is more familiar (and more sensible
IMHO - author's note).

In the MS visibility data are labelled with (amongst others) a single integer,
called the DATA_DESCRIPTION_ID. The DATA_DESCRIPTION_ID in turn maps to a
*pair* of integer indices: a SPECTRAL_WINDOW_ID (a row number in the
SPECTRAL_WINDOW table) and a POLARIZATION_ID (a row number in the POLARIZATION
table).

A single row in the SPECTRAL_WINDOW table describes a channel/subband; an entity
having frequency, bandwidth and number-of-spectral channels. A row in the
POLARIZATION table contains a set of physical polarization products - effectively
describing the polarization combinations present in the visibility data matrix.

From this it can be seen that it is possible to have the same SPECTRAL_WINDOW
correlated with different polarization setups; there is no *unique* way of
mapping a DATA_DESCRIPTION_ID to a frequency subband; to do this correctly one
must _always_ also tell which POLARIZATION_ID is desired. Of course if there's
only one POLARIZATION_ID present in the MS (i.e. everything correlated with the
same settings) this simplifies things a bit but still, the system must know
about it.

The SPECTRAL_WINDOW table is just a collection of descriptions of subbands.
There is no sort order - neither implicit nor explicit. Therefore it is
impossible to assume that, say, SPECTRAL_WINDOW_ID #x maps to subband #x in the
experiment's frequency setup. Furthermore, if the experiment observed using
multiple frequency setups, there is no guarantee that the SPECTRAL_WINDOWS are
written in the order the setups were used or stored elsewhere.

The author thinks that to place this burden on the end-user is insane - any
decent front-end should read this meta-data and present it in a sane way. This
is what the system does to sanitize:

    * read the complete spectral window table.

    * for each subband, gather all the polarization ids the subband was
    correlated with, by reading the DATA_DESCRIPTION table

    * group spectral windows by their FREQ_GROUP - a number to group
    SPECTRAL_WINDOWS together as one setup, configuration, mode, identified by
    FREQ_GROUP

    * sort the subbands for each FREQ_GROUP in increasing frequency, based on
    the frequency of the first spectral channel in the subband and re-label them
    as subband 0 - (n-1), with "n" being the number of subbands found for the
    FREQ_GROUP

All the information is displayed in the range command like below. This output
was from a slightly instrumented MS - such that it has multiple frequency
groups (setups) and the subbands in one of the setups were correlated using two
polarization combinations.

Per frequency group it lists the frequency group id and name, followed by the
subbands. Per subband the essential properties are listed like the first
frequency in the subband, bandwidth and how many spectral channels. Following
that all polarization ids (Pxxx) that specific subband was correlated with are listed,
including what the physical products in polarization setup xxx were.

listFreqs: FREQID=0 [sess311.L512]
listFreqs:   SB 0: 1626.4978MHz/8.0MHz 1024ch P0=RR,LL
listFreqs:   SB 1: 1634.4900MHz/8.0MHz 1024ch P0=RR,LL
listFreqs:   SB 2: 1642.4978MHz/8.0MHz 1024ch P0=RR,LL
listFreqs:   SB 3: 1650.4900MHz/8.0MHz 1024ch P0=RR,LL
listFreqs: FREQID=1 [FakeGrp]
listFreqs:   SB 0: 2626.4978MHz/8.0MHz 1024ch P0=RR,LL P1=RR,RL,LR,LL
listFreqs:   SB 1: 2634.4900MHz/8.0MHz 1024ch P0=RR,LL P1=RR,RL,LR,LL
listFreqs:   SB 2: 2642.4978MHz/8.0MHz 1024ch P0=RR,LL P1=RR,RL,LR,LL

These labels are used consistently throughout the plotting program to label or
designate "which subband do you mean". Plots are labelled using the FREQID,
SUBBAND and POLID indices, rather than those found in the MS. Where applicable
these indices are unmapped to their physical meanings (ie "polarization index 0"
will be written as "RR,LL".

""",

    ##################################################################
    # pl
    ##################################################################
	"pl":
"""pl
   plot current selection with current plot-properties""",

    ##################################################################
    # pp
    ##################################################################
	"pp":
"""pp
   display current plot-properties
   
Properties are e.g. which plot type, if averaging is to be applied, if so
_which_ averaging (scalar, vector) is used etc.

""",

    ##################################################################
    # lp
    ##################################################################
	"lp":
"""lp
   list all currently known plottypes""",

    ##################################################################
    # y  (not implemented?!)
    ##################################################################
	"y":
"""y [<type>]
   set/display Y-Axis type
   
<type> is one of:
[amp|pha|anp|re|im|reim|wt]""",

    ##################################################################
    # x  (not implemented?!)
    ##################################################################
	"x":
"""x [<type>]
   set/display X-Axis type 
   
<type> is one of [time|chan]""",
	"avt":
"""avt [<method>]
   set/display time averaging method 
   
<method> is one of: [none|scalar|vector|vectornorm]

vectornorm is like vector only the complex numbers are first normalized
before they are averaged. This is mostly useful for phase-versus-* plots.
""",

    ##################################################################
    # avc
    ##################################################################
	"avc":
"""avc [<method>]
   set/display channel averaging method
   
<method> is one of: [none|scalar|vector|vectornorm]

vectornorm is like vector only the complex numbers are first normalized
before they are averaged. This is mostly useful for phase-versus-* plots.
""",

    ##################################################################
    # solint
    ##################################################################
    "solint":
"""solint [none|<time duration>]
    set/display time averaging interval

When time averaging is requested (see 'avt') on 'quantity versus channel'
plots, the system must be told to integrate in which size 'time bin'. 

In 'quantity versus time' plots, the time averaging setting 'avt' is ignored
but 'solint' is honoured if set. In this case the 'by time' data will be scalar
averaged into bins of length 'solint' seconds, thus potentially reducing the
amount of points to be plotted.

The averaged data will be time stamped with the midpoint of the 'solint'
interval.

The 'solint' duration can be specified in d(ays), h(ours), m(inutes) and s(seconds) and
combinations thereof, honouring the hierarchy - "smaller" units must follow
"larger" units of time:

    > solint 1.2s     (set to 1.2 seconds)
    > solint 1.2m     (set to 1.2 minutes, i.e. 72.0s)
    > solint 1h3.2s   (set to 3603.2 seconds)
    > solint 1s1m     (invalid; m(inutes) is larger than s(econds))

Depending on the type of plots and value of 'solint', time averaging does
different things:

solint value = none / not set
-----------------------------

in 'quantity versus channel' plots:

    This is a special case of time averaging. There is no pre-defined size of
    time bin but all data of the selected time range(s) will be averaged
    separately. See the 'time' and 'scan' commands on how to select multiple
    time ranges.

    E.g. if you selected whole scans then each scan will represent a time range
    and thus all data of each individual scan will be averaged separately, to
    produce a result per scan.

    The same applies to time range(s) selected using the 'time' command -
    all data from an individual time range will be averaged.

in 'quantity versus time' plots:

    no effect; each data point found in the selected time range(s) (if any
    selected) is plotted as-is

solint value > 0.1
-------------------

In both types of plots, data will be accumulated in bins of size 'solint'
seconds and averaged before being plotted. In 'quantity by channel' plots, the
data are averaged according to the 'avt' averaging method (vector or scalar).
In 'by time' plots the data are (implicitly) scalar averaged over the 'solint'
time interval.

Any useful application of time averaging has the solint greater than the
integration time (see 'r'(ange) command). Values < 1.0s are perfectly
legitimate but may defeat the purpose, unless the integration time of the data
is << 1.0s of course.

Note that no attempt will be made to deal with data that partially spills into
the previous or next time bin. The time stamp of an integration will be
truncated such that it will be an integer multiple of the 'solint' value,
disregarding fractions of the integration time falling outside the solint
interval.

This is mostly an efficiency measure, to allow plotting very large data sets.
For detailed inspection it's always better to zoom in to the area of interest
w/o time averaging.

The time averaging code currently implements a hard lower limit for solint of
0.1 seconds.

Of course this parameter's name pays homage to Classic AIPS.
""",

    ##################################################################
    # wt
    ##################################################################
	"wt":
"""wt [double]
   set/display weight threshold""",

    ##################################################################
    # new
    ##################################################################
	"new":
"""new [<type> t(rue)|f(alse)]*
   set/display condition(s) on which new plots are created

<type> is one of [p|ch|bl|sb|src|time|all] or a comma-separated list of these,
for each of the seven interferometric axes:

    p(olarization), ch(annel=frequency bin),
    b(ase)l(ine), s(ub)b(and), s(ou)rc(e) and time.

The pseudo-type 'all' can be used to easily clear or set all values to the same
value.

Setting <type> to T(rue) means that for every unique value of <type> in the
current selection a new plot will be created. This allows exact grouping of the
selected data which is to be compared/investigated.

Example:

    You want to compare the polarization response for different baselines per
    subband:

        > new all false p,sb true

    means that a plot is started for each unique polarization, subband
    combination and all baselines, sources, times and channels  overplotted.

    Suppose you want to compare the polarization response from different
    baselines by subband. Then a 'new plot' setting of:

        > new all false sb t

    indicates that only for each unique subband a new plot is started. All
    baselines, channels &cet will be overplotted.

Note: the '*' behind the syntax means that this sequence may be repeated any
number of times!  i.e. with one single 'new' command you can set any number of
values.

Example: 
    > new p,bl true ch,sb,src,time false
would completely overwrite the current settings""",

    ##################################################################
    # pge  / nxt / prev / n / p
    ##################################################################
	"pge":
"""(nxt|prev) [nr]
   advance/go back [nr] pages of plots
   
The default is one page of plots""",

    ##################################################################
    # nxy
    ##################################################################
	"nxy":
"""n[x|y] [<int>]
   set/display number of plots in x/y direction

*** important note ***

IF the number of generated plots is less than nx * ny, the system will maximize
the size of the plots itself discarding nx/ny settings!

   """,

    ##################################################################
    # sl
    ##################################################################
	"sl":
"""sl
   display current data selection""",

    ##################################################################
    # ch
    ##################################################################
	"ch":
"""ch [none|<range>|<singlechannel> ...]*
   set/display current channel selection 

Without arguments displays the current channel selection and returns.

Using the special selection 'none' you can disable channel selection.

For not-'none' selections each "ch" command starts with an empty channel
selection. Each (white-space separated!) argument is parsed and the indicated
channel(s) are added to the selection. A <range> of channels may be specified
as ch1:ch2[:inc]; which select all channels from 'ch1' up-to-and-including
'ch2' with increment inc, which defaults to 1.

For your convenience the following symbols have been defined:

    [first|mid|last|all]

each of which will evaluate to the corresponding channel in the current MS.

arithmetic is supported, eg:

    > ch 0.1*last:0.9*last
would select the inner 80% of the channels

    > ch mid-2:mid+2
selects the five channels around the center
channel.

   """,

    ##################################################################
    # bl
    ##################################################################
	"bl":
"""bl (none|[[+-][!]<blcode>]*)
   set/display current baseline selection

Without arguments displays the current baseline selection and returns.

The special selection 'none' empties the baseline selection.

Each "bl" command first clears the selection. Then, the arguments ('selectors')
are processed from left-to-right, each selector modifying the current selection
as indicated by the selector:

    <blcode>      => +<blcode> (no "+-" specified defaults to "+")
    !<blcode>     => +!<blcode>
    +<blcode>     = add baselines matching <blcode> to selection
    -<blcode>     = subtract  ,,      ,,        ,, from   ,,
    [+-]!<blcode> = add or subtract baselines NOT matching <blcode>
     

   <blcode> is the name of the baseline; the combination of the two-letter
   antennacodes, e.g. "wbjb" for the baseline between Westerbork and Jodrell
   Bank.

But there's more! Wildcards (= '*') are allowed as well as regular-expression
type specifications:

examples:
     wb*            = match all baselines to westerbork ("wb")
                      including baselines like "efwb"
     jb(wb|ef)      = match baselines "jbwb" and "jbef"
     (jb|mc)(wb|nt) = matches four baselines

The system assumes two- or three-letter station codes. Should you need to select
stations with different number of character then use parenthesis to help the
system in finding the station name(s):

     (foobar)*
     (foobar|bazbar)ef

For your pleasure the following special names have been defined: 

    [auto|cross|all]

which will dynamically evaluate to the appropriate subset of baselines in the
current MS

""",

    ##################################################################
    # time
    ##################################################################
	"time":
"""time [none | [t1 'to' [+]t2 [,t3]*]]
   select/display time range(s) or single integration(s)

A lot can be done with this command - this help describes all three major
sections: the syntax, the time formats supported and the arithmetic/symbols that
can be used. The latter is extremely powerful, a definitive must-read!

The special selector 'none' can be used to clear the current time selection.

===== selection syntax =====

Multiple time ranges/instances in time can be selected; use a comma separated
list of time(range) selectors.

"t1 to [+]t2"

The syntax "t1 to [+]t2" selectes the time range from time stamp "t1" up to and
including time stamp "t2". If the first character of "t2" is a plus sign ("+")
the remainder of "t2" is interpreted as a duration, relative to t1.

"t3"

A single time stamp selects the integration with that time stamp. This may be
difficult as it requires the time in the MS to match very closely with the
parsed value.

===== time formats =====

A number of time-specifying formats is supported.
The general format of a time stamp is:

     [date|dayoffset]time stamp

meaning that a time stamp must be prefixed with an (absolute) date or a relative
day, where day 0 is the date on which the experiment starts.

Each of the parts in the syntax potentially has various recognized formats:

* time stamp
        hh:mm:ss[.sss]         (1)
        [*d[*h[*m[*[.*]s]]]]   (2)

    In (1) the hours, minutes and seconds are implicit, thus the colons are
    mandatory. In format (2), however, any of the fields may be left out; the
    "*" means a number, the character following the number indicates the unit.

    Examples:
        2:44:30.5     (i.e.  2h 44m 30.5s)
        1h2.25s       (i.e.  1h  0m 2.25s)

* date
        yyyy/mm/dd[/T]
        dd-(MON|mm)-yyyy[/T]

    In order to keep the date parsing predictable (the underlying CASA code can
    parse into unpredictable results!) the amount of supported date formats has
    been limited to the above formats.

    In both cases the year (yyyy) MUST be specified as a four digit quantity
    (to disambiguate). "mm", "dd" are numeric fields (1 or 2 digits), "MON" is
    the three-letter English month name.

    The date code is followed by a time stamp (see previous section): the
    date and time must be separated. Both the "/" as well as the ISO8601 "T"
    separator are recognized.

    Examples:
        31-oct-2011/0:1:1      (yields: 31-Oct-2011/00:00:01.000)
        2012/6/1T16h31m50.3s   (yields: 01-Jun-2012/16:31:50.300)

* dayoffset
        [-]nrdays/

    By prefixing a time stamp with "<number>/" or "-<number>/" the time stamp is
    placed on the date computed from adding the offset <number> to the start
    date of the experiment.

    Examples:
        Suppose the start-time of the experiment is
        31-Oct-2011/12:42:30.000

        then:
            -1/16:31:50.3 gives 30-Oct-2011/16:31:50.300
            1/12h         gives 01-Nov-2011/12:00:00.000

===== arithmetic =====

Ah, now it's getting interesting. In order to ease selecting time ranges in the
experiment some symbolic time stamps have been implemented as well the
possibility to do elementary arithmetic on time stamps.

The following pseudo time stamps have been defined:

    $start $mid $end $length $t_int

When used, they evaluate to the corresponding value (in absolute date/time,
except for the duration and integration time) based on the current MS. 

In order to help dealing with time, for arithmetic one can use 'durations' -
time spans. They look a lot like time stamps but they are not; they just
translate in "a number of seconds".

The support 'duration' format is:

        [*d[*h[*m[*[.*]s]]]]

    Which means, like with the time stamp before, any of the fields may be
    omitted but a 'larger' unit of time MUST precede a 'smaller' one.

Below are a number of examples of actual time stamp agnostic selections, that
will select the indicated time range of the experiment, irrespective of the
_actual_ time stamps in the experiment.

    # select time range from start -> end (the whole experiment)
    > time $start to $end

    # select time range from start -> start + length (id.)
    > time $start to + $length  

    # select the second half of the experiment
    > time $start + ($length/2) to $end

    # the third hour-and-a-half:
    > time $start + 2*1h30m to +1h30m

    # the last hour of the experiment:
    > time $end - 1h to $end

    # do some arithmetic using the integration time:
    # select 10 integrations after skipping the first 5: 
    > time $start + 5 * $t_int to + 10 * $t_int

    # Involving relative day time stamps - select
    # from 23h00m00s on day 0 of the experiment to 01h00m00s
    # on the next day of the experiment.
    > time 0/23h to 1/1h

    # of course you can mix + match
    > time $start to 0/23h30m
    > time 1/1h to $end - 1h

    # comma separated statements select multiple time ranges.
    # here we select two 1 minute time ranges, 1 minute apart.
    # (this time we show the output on the terminal)
    > time $start to +1m , $start + 2m to +1m
    time: 31-Oct-2011/12:42:31.250 -> 31-Oct-2011/12:43:31.250
    time: 31-Oct-2011/12:44:31.250 -> 31-Oct-2011/12:45:31.250 

    # note that overlapping time ranges will be automatically joined!
    > time $start to +4m , $start + 3m to +3m
    time: 31-Oct-2011/12:42:31.250 -> 31-Oct-2011/12:48:31.250

Anyway, you get the idea. You'll find lots of caveats, constructions that won't
be recognized or give very weird time values. Still, you should be able to make
good use of this feature, as can be seen from the examples.

Stuff that certainly don't work:
    > time [-]1/$start

because the relative day offset expects a time stamp to follow, not a full
fledged date-and-time.

However, this can trivially be rewritten to:
    > time $start [+-] 1d

which *does* work and gives the expected date+time stamp.""",

    ##################################################################
    # s
    ##################################################################
	"s":
"""sb [<range> <single sp.window>]
   set/display current spectralwindow (subband) selection.
   
<range> may be specified as spw1:spw2[:inc] selecting
spectral windows spw1->spw2 with increment inc (defaults
to 1). The word all would select all spectral windows""",

    ##################################################################
    # p
    ##################################################################
	"p":
"""p [<polcombi> <polcombi>... ]:
   set/display current polarization selection 
   
<polcombi> may be specified as two out of [xylr*] or the
string all.  eg: l* would select ll lr ** would select
all polarizations note: for your pleasure the following
symbolic selections have been added:
[all|parallel|cross]""",

    ##################################################################
    # src
    ##################################################################
	"src":
"""src [none|[[-+][!]<source>]*]:
   set/display current source selection 
 
Without arguments displays the current source selection and returns.

The special selector 'none' removes the current source selection.

Each "src" command first clears the selection. Then, the arguments ('selectors')
are processed from left-to-right, each selector modifying the current selection
as indicated by the selector:

    <source>      = +<source> (no "+-" specified defaults to "+")
    !<source>     = +!<source>
    +<source>     = add sources matching <source> to selection
    -<source>     = subtract  ,,      ,,     ,,   from   ,, 
    [+-]!<source> = add or subtract sources NOT matching <source>
     

<source> is the name of the source as it appears in the MS.  The range command
("r") lists all selectables, including the source names. The match is performed
case insensitive.  Also, the <source> pattern is autmatically anchored at begin
and end - it will not look for the <source> pattern IN the source names; in
stead the whole source name must match the pattern. This has notable effects for
e.g.  the "!" based selectors - it's very easy to write a pattern that doesn't
match *any* of your source names, hence the "!" will select ALL of the sources.

<source> may contain wildcards or limited regular expression style patterns:

    # select all sources whose name begins with "3c" or "3C"
    > src 3c*

    # selects sources that contain the string "nrao" anywhere
    > src *nrao*

    # select all sources whose name does NOT begin with 3c
    > src !3c*

    # select all sources whose name does NOT begin with 3c but
    # exclude the ones having "NRAO" in their name
    > src !3c* -*nrao*

    # select all J16* and M8* sources
    > src (j16|m8)*

    # select 3c84 + 3c312
    > src 3c(84|312)
""",

    ##################################################################
    # gui (not implemented)
    ##################################################################
	"gui":
"""gui [plot|select]
   open the gui the plotter or the selector

the jivegui consists of three gui-elements the main-gui
(for file-control, the selector-gui and the plotter-gui""",

    ##################################################################
    # f (not implement)
    ##################################################################
	"f":
"""f
   do simpleminded fringe-fit on dataselection

the current fringe-fit does a baseline based fringe-fit""",

    ##################################################################
    # si (not implemented)
    ##################################################################
	"si":
"""si [<double>]:
    set/display current solution interval for fringe-fit

unit is seconds!""",

    ##################################################################
    # hc (not implemented; see "save")
    ##################################################################
	"hc":
"""hc <filename>
   make hardcopy of all plots

<filename> is the PGPLOT destination, thus
it may be specified as:

    <label>[/<device>]

<device> is any PGPLOT acceptable device. The default <device> is "cps" (Color
PostScript), implying that <label> is interpreted as a filename.

Examples:
    # plot to X-Window with title <label>
    > hc <label>/xw
    # plot to PostScript file <label>
    > hc <label>/cps

""",

    ##################################################################
    # xyscale (implemented - now is the "x|y[n] ...." command
    ##################################################################
	"xyscale":
"""x|y[01] [local|global|<lo> <hi>]
    set or display x-axis or y-axis scaling.

Using this command the x- or y-scale of the plots (or panel-wihin-plot) can be
set or reviewed.


The plot software recognizes three different scaling options:
    'local'     each individual plot will be auto scaled based on the 
                minima and maxima of the data sets displayed in each plot
    'global'    all plots share the same scale, which will be derived from
                the global minima and maxima across all plots

    <lo> <hi>   explicitly set the scale of axis "x" or "y" (optionally
                for panel "0" or "1" if indicated) to the given <lo> and <hi>
                values (numerical). This can be regarded as an external global
                scale.
                Note: user-friendly setting of x-axis limits when the x-axis is
                time (such that it recognizes human-readable time formats for
                <lo> and <hi>) is planned but may likely have low priority.

The numerical suffix "0" or "1" means to set the scale of panel "0" or "1" in
the plot(s). The panel numbering counts from lowest panel to highest panel.
Typically this is most useful if multiple quantities are plotted at the same
time, e.g. in "anp*" plots where both amplitude and phase are plotted. 

Thus, in "anp" plots, the amplitude is plotted at the bottom and phase in the
top. To make the amplitude plots scale locally (e.g. if cross- and
autocorrelations are both present) this would work:

    > y0 local

Or change the phase panel away from the -180,180 degree default:

    > y1 0 360

The new scaling options will be visible after the next plot command.""",

    ##################################################################
    # pt
    ##################################################################
	"pt":
"""pt [plottype]
   set/display current plottype 

Use the lp command to list available plottypes.""",

    ##################################################################
    # fq
    ##################################################################
	"fq":
"""fq [none | [[freqids/]subbands[/[polid:]polarizations]]*]
   set/display current frequency/polarization selection

Without arguments displays the current frequency selection and returns.

The special selector 'none' clears the current frequency selection.

In order to understand this command you should familiarize yourself with how the
system reads the frequency setup/subband and polarization from the MS and
presents it to you. Read the documentation for the range command "r".  Part of
the complication comes from what the MS actually *allows* to be stored. 

An "fq" command starts with an empty frequency selection and processes the
selectors from left to right and adds the indicated selection to the current
selection. The syntax looks awkward but let's explain. In principle a full-blown
"fq"-selector looks like:

    <freqids>/<subbands>/<polarizations>

which selects a specific (set of) <subbands> and <polarizations> from a (set
of) <freqids> (frequency groups, mode, setup, frequency configuration, ...).

If there is only one freqid in the experiment it may be left out, that's the
'optionality' described by [freqids/] in the syntax.

The <freqids> and <subbands> fields are ranges or sets of numbers (or a
combination thereof):

    <freqids>, <subbands> = <set> ( , <set> )
    <set>                 = <number> | <number>:<number>

The <polarizations> field is optional. If unspecified it selects all
polarizations for the given <fqid>/<subband> combinations. 

If specified, the <polarizations> field is a comma separated list of strings,
identifying the polarizations to select. The set of polarizations may be,
optionally, prefixed with the polarization ID from which to select the
indicated polarizations: the MS allows the same spectral window to be
correlated with different polarization combinations. The system will typically
select, if you don't specify, all that apply. Usually there is only one
polarization ID so there won't be any difference (other than specifiying a
non-existant polarization ID). 

For your convenience the special "polarization" values "X" and "P" (case
sensitive) have been defined; they dynamically select "all cross polarizations"
and "all parallel polarizations", for the indicated <fqid>/<subband>
combinations, respectively.

    <polarizations> = <set> ( , <set> )
    <set>           = [<polid>:]<pols>
    <polid>         = numeric Polarization ID label from MS
    <pols>          = rr, rl, lr, ll, P, X

Note: wildcards are allowed in the polarizations, e.g. it's possible to select
"r*" to select "RR" and "RL", should they be available.

Examples:
    
0,3/0:4/l*
select pols LL+LR from subbands 0->4 from both FREQID 0 and 3

1/1:3,6:8/X
select available cross pols from subbands 1->3 and 6->8 from FREQID 1

0/1:RR
select polarization RR from polarization ID 1, from subband 0 from the default
FREQID (this will only work when there's only one FREQID in the MS)

**** USEFUL NOTES ****

If there is only one FREQID present in the MS the <freqidsel>
may be omitted.

If the <polsel> is omitted the system will automatically select all
polarizations in the first polarization ID found (usually there is only one
polid, which makes this useful)

Subbands are counted from 0.

There is no direct connection to SPECTRAL_WINDOW in the MS; internally the
subbands are re- ordered (within each FREQID) such that subband 0 has the
lowest frequency and subband (nsub-1) the highest.


You can use the r (range) command to check on available
FREQIDs and subbands/FREQID.

""",

    ##################################################################
    # sort
    ##################################################################
"sort":
"""sort (none|[AX*])
    print or set the current plot sort order.

The order in which the plots are displayed can be influenced with this command.
Each plot has labels, see 'new' command for a description of all the possible
labels. By specifying these labels as sort order one can sort the plots by one
or more fields.  Use 'none' to turn off sorting of the plots.

Examples:
  first, make sure that new plots are created for every FQ/SB/BL combination
  (set all to false, then set the 'new plot' property for the labels 'fq', 'sb'
   and 'bl' to true):

    > new all f fq t sb t bl t
  
  Now it's possible to sort the plots by subband:
    > sort sb
    
  or by baseline first, then by subband:
    > sort bl sb

Note that the order of the labels appearing in the sort command determines the
order of the sort keys.

""",

    ##################################################################
    # mark
    ##################################################################
"mark":
"""mark [index:] [expression]
    mark points satisfying [expression] in a visually distinctive manner

Sometimes you want to be able to quickly find points satisfying a special
condition, or just see IF there are data points satisfying a particular
criterion. For those needs there is this 'mark' command.

Without arguments or just an index it will print the current marking
expression for that y-axis type or for all of the y-axis types if no specific
index was given. The colon separates an index from an (optional) expression.

The expression is a boolean condition which is evaluated for every data point.
Points for which the evaluated expression yields 'True' are drawn in such a way
they stand out w.r.t. the rest of the plotted points.

For plots where two quantities are plotted (e.g. "amplitude and phase versus
...") it is possible to specify a different marking condition per y-axis type.

The index can be given as a numerical index 0..n (0 being the lowest panel
in the plot, n-1 the highest) OR as the actual name of the quantity to which
the expression should apply. See examples below.

If no index is given, "all y-axes" is assumed.

The 'expression' is, preferrably, a valid boolean Python expression involving
the variables 'x' and/or 'y'; the respective 'x' and 'y' values of the data
point being tested. The expression is interpreted as Python code - so
arithmetic and boolean operations like 'and' and/or 'or' are perfectly legal,
if you stick to Python syntax ...

The special expression "none" will remove the marking condition for the
indicated y-axis/axes.

For your convenience:

    * all functions from the python 'math' module are imported and can be used
      without module prefix (thus 'sin(x)' rather than 'math.sin(x)')

    * the following, per data set dynamically computed, symbols are available
      for your expression:
        xmin, xmax, ymin, ymax: the limits of the data set the current data
                                point came from
        avg, sd               : the mean and standard deviation of the data set
                                the current data point came from

Note: in plots of quantity versus channel, the 'x' axis is just the channel
number, not the frequency of the spectral channel under consideration.

Examples:

    Display marking conditions for all data sets:
        > mark

    or for a particular data set: (note the use of ":" to force interpretation
    as index)
        > mark 0:
        > mark phase:
    
    In a weight-versus-time plot (plottype 'wt'), mark all points that
    have a weight below 0.98:

        > mark y<0.98

    Mark all points that are more than 2.0 standard deviations away from the
    average:

        > mark abs(y-avg)>2.0*sd

    Or mark all points that have an x-axis value between 33 and 65 in the
    'phase' sub plot of *-and-phase-versus-*:

        > mark phase: x>33 and x<65

    Because the phase is always the upper plot in these plots, one could
    equivalently say:

        > mark 1: x>33 and x<65

    Mark the lowest 10% of amplitudes in each data set:

        > mark amplitude: y < ymin + 0.1*(ymax - ymin)

""",

    ##################################################################
    # store
    ##################################################################
"store":
"""store [[<expression>] ['as' <variable name>]]
    store the result of <expression> as <variable name> or display defined variables

The plot software supports storing the result of a data manipulating expression
as a variable. Stored variables can be referenced in future data manipulating
expressions. In this context, data can refer to either numbers or a collection
of plots. See also the "load" command.

Without arguments the store command displays the currently defined variables and
what their value is. Should a variable refer to a set of plots, it displays some
metadata about them for identifyability.

The biggest use case of this feature is comparison of data sets - e.g.
differencing them or computing their ratio.

In order to understand the functioning of these expressions it is necessary to
know a little of how the data sets are stored. Each data set is labelled with
TIME, FQ, SB, BL, P, CH, SRC fields and contains two arrays: x-abcissae and
y-ordinates; the actual data. The expressions in this command operate on the
y-ordinates of the data sets as a whole. To say "add 3 to the data sets" is
saying "add the value of 3 to each individual y-ordinate in all data sets".

Variable names are alphanumeric strings (extended with the '_' character).

        [a-zA-Z0-9_]+      e.g. phatime_unb0, POL_rr

        The special variable '_' means "the current set of plots" (*)

        It is possible to create plots that produce multiple y-values, e.g.
        "phase and amplitude versus channel". Should it be necessary to
        restrict processing to a particular subset of the data, the data set
        variable can be extended with a ".<type>" specifier which will filter
        only the data sets of type <type>:

            ms1_data.phase or foo.real

        (The "lp" (list plot-types) command should be indicative of which types
        of data sets the program can produce.)

Either of the <expression> or 'as <variable name>' can be left out but not
both. The unspecified part will default to '_'. Thus:

        > store as foo 
            is short for:
            > store _ as foo
        i.e. "save current set of plots by the name of foo"

        > store foo
            is short for:
            > store foo as _

        i.e. replace the current set of plots by whatever foo refers to. In
        fact, this is equivalent to the command:
            > load foo      (see "help load" for details)

The expression can be any expression involving numbers and variables
and supports all standard arithmetic operations including exponentiation ('^')
and unary minus. Note that unary minus has higher precedence than '^', thus
'-1^2' evaluates to +1.

    Examples:
        > store 1+2*3+4 as eleven     

        # ... open MS, select data, create plots
        > store as ms1_data           (store collection of plots)
        # ... open other MS, select data & create plots
        > store as ms2_data           (id.)

        # now it gets interesting ...
        > store ms1_data - ms2_data as diff
        > store ms1_data / ms2_data as ratio

        # or, more convoluted ....
        > store ms1_data / ((0.1*ms2_data)^eleven/12)

        # diff and ratio can now be loaded and visually inspected.
        > load diff

        # note: in combination with the "win xxxx" function, to open multiple
        # plot windows, this becomes quite powerful

It should be noted that an operation which combines data sets (e.g. "a + b")
will only produce results for data sets that have the same labels; thus having
the same TIME, SB, FQ, BL, P, CH, SRC values. The expression-evaluation engine
iterates over the data sets that both collections have in common and evaluates
the expression on all the y-values of that data set. 

Thus:

        > store foo - bar

        will compute, for all data sets that 'foo' and 'bar' have in common,
        the difference between all the y-ordinate arrays in each data set.
        Should the number of y-ordinates be different, then only the first 'n'
        values will be differenced and returned, with 'n' being the shorter
        length of the two.

(*) Of course, no software could be complete without a mild laugh about Perl
(;-)) where the '_' variable is the 'default' variable, whatever that means ...

""",

    ##################################################################
    # load
    ##################################################################
"load":
"""load <expression>
    load previously stored plots as current or display defined variables

It is possible to load a previously stored set of plots, potentially modifying
them, as current set of plots. See "help store" for more details on storing
plots and the expression syntax.

Without arguments the load command displays the currently defined variables and
what their value is. Should a variable refer to a set of plots, it displays some
metadata about them for identifyability.

Example:

    Having previously stored data from two different measurement sets (but with
    identical labels), it is possible to immediately load the difference as:

        > load foo - bar

    In fact, any expression can be entered (see "help store"). It is now also
    possible to subtract e.g. polarizations or subbands, even if that makes no
    sense. See the extended subscript section below.

    A nice side effect is that the plots will be restored based on the
    *current* 'new plot' setting, which may be different from the 'new plot'
    setting at the time of loading the original data. (See "help new" for
    details about this.)

    The practical upshot is that, to change the organization of the plots, it
    is no longer necessary to re-read (and potentially re-process) the data
    from disk, which can be a lengthy process. In the olden days, if the 'new
    plot' settings changed and a replot was requested, all data would be
    re-read from disk.

    It is possible to do this:

        #   open MS, select data, make plots
        > pl
        #   ... time passes (program grovels over hard disk) ...
        #   plots are displayed.
        #   Now store them:
        > store as foo

        #   Change new plot settings:
        > new bl f sb t

        $   load the plots back in and voila, they're reorganized
        > load foo

Subscripting:

    As explained under "help store", when combining variables that represent a
    series of data sets, only those with matching property values are combined.

    Under that scheme, subtracting polarizations from each other is impossible -
    the polarization property of those data sets have different values and thus
    would never be combined in an expression like this:
        > load foo - bar

    It is now possible to use variable subscripting to select a subset of data
    sets from that variable and "erase" the property value:
        > load foo[p=ll]
    
    The expression "foo[p=ll]" returns a temporary variable which contains the
    list of data sets addressed by the variable "foo" but only those for which
    the "P"(olarization) property has the value "LL". 

    Because the property's value is "erased" after the selection, it becomes
    possible to type in this:

        > load foo[p=ll] - foo[p=rr]

    Because now the two temporary values represent lists of plots with the same
    values bar the polarizations. But since this property's value has been
    erased they will not be compared and thus the expression will subtract the
    right-hand polarization data from the left hand's.

    The expression within '[' ... ']' can contain multiple, comma separated
    selectors for the attributes:
    
        <expression> = '[' <condition> { ',' <condition> } ']'
        <condition>  = <attribute> '=' <value> 
        <attribute>  = "P", "CH", "SB", "SRC"
        <value>      = <number> | <text>

    The system knows that certain properties are text-valued and other numerical
    valued. So "sb=aap" will fail, hopefully with a useful error message.

""",

    ##################################################################
    # ckey
    ##################################################################
"ckey":
"""ckey [<expression> | none]
    set or display data set colouring condition(s)

Without arguments it displays the current data set colouring condition, an
argument <expression> changes the data set colour assignment. This setting is
kept and changed per plot type. The <expression> "none" will reset the
colouring algorithm to the default ("ckey_builtin").

Background: the plot program has plots and data sets: one plot consists of one
or more data sets displayed in it. Each individual data point has seven
attributes: P, CH, SB, FQ, BL, SRC, TIME.

Normally each data set gets a unique, automatically assigned, colour, based on
its label. A data set's label is made up of the values of the subset of
attributes that are not the x-axis or in the plot label (see the "new" command)
or whose value is the same for all data sets. (E.g. if all data is for the same
source, the SRC attribute will have the same value in all data sets and thus
will be be removed from the data set label on the pretext of being
un-informative.)

The general idea of the ckey command is that it is possible to express
intentions like:

    "if a data set's attribute X has a value which is one of the set [a,b,c]
    then give it the colour Y", or

    "use the SB and P attribute values as unique key in stead of whatever the
    default behaviour is"

To express these intents an <expression> can be passed as argument(s) to the
ckey command. The <expression> syntax is as follows:

    <expression> = <selector> { <selector> }
    <selector>   = <conditions> { '=' <colour number> }
    <conditions> = <attrval> { ',' <conditions> }
    <attrval>    = <attribute> { '[' <value> { ',' <value> } ']' }
    <attribute>  = p | ch | sb | bl | fq | src | time
    <value>      = <number> | <text> | ' <text> ' | <regex>
    <regex>      = '/' <text> '/' { 'i' }
    
In this grammer, <number> and <text> are what you think they are: digits and
characters (excluding white space). When attribute values are compared with
<text> it is done case-insensitive. The <text> inside a <regex> is, arguably,
the regular expression text and may contain embedded spaces. It may be suffixed
by the character 'i' for case-insensitive regex matching. Regular <text> may be
put inside single quotes to support embedded spaces.

In human readable form this grammar reads:

    An <expression> consists of one or more <selectors>.
    
    Each <selector> selects data sets with a label that matches a set of
    conditions and optionally assigns a specific colour index to those.

    A <selector> is a comma separated sequence of one or more <conditions>.

    Each <condition>, in turn, adresses a data set's attribute and, optionally,
    matches only if the attribute value is a member of the list of given
    values. Otherwise it evaluates to the value of the attribute.

    If no explicit <colour number> is assigned to the <selector>, the system
    automatically assigns the first unused colour number.

Examples:

    # assign data set colours just based on the unique combinations of values
    # for P, SB found in the data sets
    ckey p,sb

    # one of the easiest applications: colour the data sets with 'll'
    # polarization with colour 2 and the 'rr' polarization with colour 3. 
    ckey p[ll]=2 p[rr]=3

    # suppose the even subbands are LSB and the odd ones USB, let's
    # assign colour based on 'sideband'. The system assigns the colours.
    ckey sb[0,2,4,6]  sb[1,3,5,7]

    # and maybe add the polarization as key
    ckey p,sb[0,2,4,6]  p,sb[1,3,5,7]

    # colour baselines to Wb with colour 2, the rest gets automatically
    # assigned a colour based on subband
    ckey bl[/(^wb)|(wb$)/i]=2 sb

""",

    ##################################################################
    # draw
    ##################################################################
"draw":
"""draw [drawing options]
    set or display drawing styles for the datasets

Without argument(s) "draw" displays the current drawing styles for the panel(s)
of the current plot type. With arguments the drawing style(s) can be changed.

jplotter supports two independent and three drawing styles in total:
    Lines, Points and Both

The drawing style(s) are kept per plot type and per panel-within-plot. As a
consequence, the drawing style can be changed per plot type and per subplot
within a plot.

"[drawing options]" is an optional list of drawing style commands. Each drawing
style command is:
        
    (<panel>:)[lines|points|both]

If the drawing style is prefixed with "<panel>:" the entry is called
'qualified' - meaning that it changes only the setting for that panel (e.g. the
"phase" panel in a amplitude-and-phase-versus-* plot).

The draw command accepts only all qualified or all unqualified arguments. With
just one unqualified argument, it sets the drawing style for all panels. The
other option is to provide exactly the amount of styles as there are panels, in
which case style N will be set for subplot N.

Examples:

    # styles are kept per plot type so let's choose one with two subplots:
    # "amplitude" and "phase"
    > pt anptime

    # let's see what the current defaults are
    > draw
    drawers[anptime]: amplitude:Points phase:Points

    # 'one unqualified argument sets all drawing styles'
    > draw lines
    drawers[anptime]: amplitude:Lines phase:Lines

    # 'exactly N panel unqualified arguments set the styles in order'
    > draw both points
    drawers[anptime]: amplitude:Both phase:Points

    # using 'qualified arguments' we can set one (or more) panels'
    # drawing style explicitly
    > draw phase:lines
    drawers[anptime]: amplitude:Both phase:Lines

""",

    ##################################################################
    # filter
    ##################################################################
"filter":
"""filter [[<panel>':'] <expression> | none]
    set or display post-reading pre-plotting filter condition(s)

Without arguments it displays the current filter(s) that have been set for the
current plot type. With argument it (re)sets a filtering condition for (a
subplot of) the current plot type. The filter condition is applied to the values
of each data set's properties "P", "CH", "SB", "BL", "SRC"  (see the jplotter
cookbook pdf for background).

Using this command it possible to filter data sets to be plotted after they've
been read from disk but before they're displayed. All data sets remain in memory
but only the ones for which the filter condition is true will be plotted. The
condition(s) can be set separately for different subplots of a multi-panel plot.

This is particularly useful if a large-ish data set has been read from disk to
prevent having to re-read the whole data set in case only a subset is to be
displayed.

Another use case is amplitude-and-phase versus channel plots: in the amplitude
panel the users wanted to display all polarizations but in the phase panel only
the parallel ones were deemed useful (cross baselines cross polarization
products usually carry very low signal.)

The <expression> follows the grammar below; examples follow after that. The
syntax is mostly consistent with the syntax from the 'ckey' command or the
'scan' selection's "where" clause.

    An <expression> may be prefixed with
    
    <address>':'  
        where <address> = '0' | '1' | 'amplitude' | 'phase' | 'real' | 'imaginary'
        basically the address of the (sub)panel of a potentially multi-panel
        plot. The numerical indices address from bottom panel to top.

        If a plot type only has one panel (e.g. "amptime" - amplitude vs time)
        then "0:" or "amplitude:" are still valid addresses but no observable
        difference in behaviour of the command is expected compared to had it
        been called with the address left out.

    <expression> = <condition> { 'and'|'or' <expression> } |
                   'not' <condition> | '(' <expression> ')'

    <condition>  = <attribute> '~' <match>  | 
                   <attribute> <compare> <value> |
                   <attribute> 'in' <list>

    <attribute>  = 'p' 'ch' 'sb' 'src' 'bl'

    <list>       = '[' <list items> ']'
    <match>      = '/'<regular expression>'/' | <text>
    <compare>    = '<' '<=' '=' '>' '>='
    <value>      = <number> | <text>

The general idea of the filter command is that it is possible to express
intentions like:

    "if a data set's attribute X has a value which is one of the set [a,b,c]
    then plot it", or

    "if a data set's attribute Y has a value which is less than z then plot it"

In the grammer, <number> and <text> are what you think they are: digits and
characters (excluding white space). When attribute values are compared with
<text> it is done case-insensitive. The <text> inside a <regex> is, arguably,
the regular expression text and may contain embedded spaces. It may be suffixed
by the character 'i' for case-insensitive regex matching. Regular <text> may be
put inside single quotes to support embedded spaces.

In human readable form this grammar reads:

    An <expression> consists of one or more <conditions>.
    
    Each <condition> selects data sets for which this condition returns true. It
    is possible to negate a <condition> using the 'not' operator.

    It is possible to use logical operators 'and' and/or 'or' to join
    <conditions>.

    It is possible to use parentheses to combine multiple 'and'ed or 'or'ed
    <conditions>.

Examples:

    # The second use case described above can be scripted like this.
    # It is assumed that some data has been selected with all polarizations in.
    # Select plot type with two panels, "amplitude" and "phase":
    pt anpchan
    # In the phase panel only display parallel hands of polarization
    filter phase: p in [ll,rr]

""",
}

