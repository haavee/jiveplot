# HV: Contains parsers for querying list of scans 
import re, hvutil, operator, math, itertools, inspect, plots, plotiterator, copy, numpy

haveQuanta = False
try:
    import pyrap.quanta
    haveQuanta = True
except ImportError:
    pass 

class token_type(object):
    __slots__ = ['type', 'value', 'position']
    # tokens must AT LEAST have a type. Most other things are optional
    def __init__(self, tp, val=None, **kwargs):
        self.type     = tp
        self.value    = val
        self.position = kwargs.get('position', -1)

    def __str__(self):
        return "(type={0}, val={1}, pos={2})".format(self.type, self.value, self.position)

    def __repr__(self):
        return str(self)

def mk_tokenizer(tokens, **env):
    def do_tokenize(string):
        pos = 0
        while pos<len(string):
            # Run all known regexps against the current string and filter out which one matched
            moList = map(lambda (mo, tp): (mo, tp(mo, position=pos, **env)),
                         filter(lambda tup: None if tup[0] is None else tup,
                                map(lambda (rx, tp): (rx.match(string, pos), tp), tokens)))
            if len(moList)==0:
                raise RuntimeError, "\n{0}\n{1:>{2}s}^\n{3} tokens matched here".format(string, "", pos, len(moList))
            # extract match-object and the token from the result list
            (mo, tok) = moList[0]
            pos      += (mo.end() - mo.start())
            # Ignore tokens that say they are nothing (e.g. whitespace)
            #print "TOKEN: ",tok
            if tok is not None:
                yield tok
        yield token_type(None, None)
    return do_tokenize

# helper functions
def mk_number(val):
    try:
        return int(val)
    except ValueError:
        return float(val)

# transform a matched date/time object into MJD seconds
def mk_mjdsecs(mo, **kwargs):
    if not haveQuanta:
        raise RuntimeError,"pyrap.quanta module not available for date/time to seconds functionality"
    # we must have 'year', 'month' and 'day'
    # 'h', 'm' and 's' are optional and default to 0
    # We know that CASA's quantity parses
    #   day-month-yearThhHmmMss.sssS correctly
    # so transform our input to that form
    G      = mo.group
    # Be careful to not access groups that may not be present at all!
    option = lambda group: G(group) if group in mo.groupdict() and mo.groupdict()[group] else 0
    return pyrap.quanta.quantity(
              "{0}-{1}-{2}T{3}H{4}M{5}S".format(G('day'), G('month'), G('year'), option('h'), option('m'), option('s'))
           ).get_value("s")

# Transform relative day time stamp and durations into seconds
def mk_seconds(mg, **kwargs):
    rv     = 0.0
    secday = 86400
    grps   = mg.groupdict()
    if 'd' in grps and mg.group('d'):
        rv += float(mg.group('d'))*secday
    if 'h' in grps and mg.group('h'):
        rv += float(mg.group('h'))*3600.0
    if 'm' in grps and mg.group('m'):
        rv += float(mg.group('m'))*60.0
    if 's' in grps and mg.group('s'):
        rv += float(mg.group('s'))
    if 'offset' in grps and mg.group('offset'):
        rv += (int(mg.group('offset')) + (math.floor(float(kwargs.get('start', 0))/secday))) * secday
    if 'neg' in grps and mg.group('neg'):
        rv *= -1
    return rv

resolver = lambda group: lambda mg, **env: env[mg.group(group)]

def mk_operator(which):
    ops = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv, '^':operator.pow,
           '&&': operator.and_, '||': operator.or_, '!':operator.not_,
           'and': operator.and_, 'or': operator.or_, 'not':operator.not_,
           'in': lambda x, y: operator.contains(y, x), '=':operator.eq,
           '<': operator.lt, '<=':operator.le, '>=':operator.ge, '>':operator.gt }
    return ops[which]


# Token makers
token_def  = lambda pattern, fn: (re.compile(pattern), fn)
ignore_t   = lambda      : lambda o, **k: None
keyword_t  = lambda      : lambda o, **k: token_type(o.group(0), **k)
simple_t   = lambda tp   : lambda o, **k: token_type(tp, **k)
value_t    = lambda tp   : lambda o, **k: token_type(tp, o.group(0), **k)
# extract a given match group
extract_t  = lambda tp, g: lambda o, **k: token_type(tp, o.group(g), **k)
# xform    = transform match group 0 [ie the whole regex match]
xform_t    = lambda tp, f: lambda o, **k: token_type(tp, f(o.group(0)), **k)
# xformmg  = transform the match object; ie you have access to all match groups
# xfrommge = same as previous but you have the environment passed in as well
xformmg_t  = lambda tp, f: lambda o, **k: token_type(tp, f(o), **k)
xformmge_t = lambda tp, f: lambda o, **k: token_type(tp, f(o, **k), **k)
number_t   = lambda      : xform_t('number', mk_number)
operator_t = lambda tp   : xform_t(tp, mk_operator)
datetime_t = lambda f    : xformmge_t('datetime', f)
resolve_t  = lambda tp, g: xformmge_t(tp, resolver(g))
#resolve_t  = lambda tp, g: lambda o, **k: token_type(tp, resolver(g)(o, **k), **k)

# helper functions to help build token regexes
MAYBE = lambda x: r"("+x+r")?"
NAMED = lambda n, x: r"(?P<"+n+r">"+x+r")"

# Patterns for supported datetime formats
YMD      = r"(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})"
DMY      = r"(?P<day>\d{1,2})-(?P<month>([a-zA-Z]{3}|\d{1,2}))-(?P<year>\d{4})"
DMY_EUR  = r"(?P<day>\d{1,2})/(?P<month>([a-zA-Z]{3}|\d{1,2}))/(?P<year>\d{4})"
SEP      = r"[/T]"
TIME     = r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2}(\.\d*)?)"
HMS      = r"(?P<h>\d{1,2})[hH](?P<m>\d{1,2})[mM](?P<s>\d{1,2}(\.\d*)?)[sS]"
RELDAY   = r"(?P<offset>-?\d+)/"
# Float needs a decimal point somewhere in there, int never, number is either.
# note that numbers never include a leading '-' so parsers are responsible for
# supporting unary '-'
FLOAT    = r"((\d+\.\d*)|\.\d+)([eE]-?\d+)?"
INT      = r"\d+"
NUMBER   = r"((\d+(\.\d*)?)|\.\d+)([eE]-?\d+)?"

# combine a date+time format and generate a token definition out of it
datetime_token = lambda date_fmt, time_fmt: token_def(date_fmt+SEP+time_fmt, datetime_t(mk_mjdsecs))

# Always nice to have. Note: if you want to support int and float at the same time
# care should be taken about the order. If you put 'int_token' before 'float_token'
# then you'll most likely never get a floating point token because the part
# before the decimal point will match the int token
float_token    = lambda : token_def(FLOAT , xform_t('float', float))
int_token      = lambda : token_def(INT   , xform_t('int',   int))
number_token   = lambda : token_def(NUMBER, number_t())


def parse_scan(qry, **kwargs):
    # Helper functions

    def mk_intrange(txt):
        return hvutil.expand_string_range(txt, rchar='-')

    # take a string and make a "^...$" regex out of it,
    # doing escaping of regex special chars and 
    # transforming "*" into ".*" and "?" into "."
    # (basically shell regex => normal regex)
    def pattern2regex(s):
        s = reduce(lambda acc, x: re.sub(x, x, acc), ["\+", "\-", "\."], s)
        s = reduce(lambda acc, (t, r): re.sub(t, r, acc), [("\*+", ".*"), ("\?", ".")], s)
        return re.compile("^"+s+"$")

    def regex2regex(s):
        flagmap = {"i": re.I, None:0}
        mo = re.match(r"(.)(?P<pattern>.+)\1(?P<flag>.)?", s)
        if not mo:
            raise RuntimeError,"'{0}' does not match the regex pattern /.../i?".format(s)
        return re.compile(mo.group('pattern'), flagmap[mo.group('flag')])

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        # keywords
        token_def(r"\b(to|not|where|limit)\b",  keyword_t()),
        # can't use 'keyword_t()' for the next one because we may need to accept whitespace between "order" and "by"
        token_def(r"\b(asc|desc)\b",            keyword_t()),
        token_def(r"\border\s+by\b",            simple_t('order by')), 
        token_def(r"\bin\b",                    operator_t('in')),
        token_def(r"\b(and|or|in)\b",           operator_t('relop')),
        # Date + time formats
        datetime_token(YMD,     TIME),
        datetime_token(YMD,     HMS),
        datetime_token(DMY,     TIME),
        datetime_token(DMY,     HMS),
        datetime_token(DMY_EUR, TIME),
        datetime_token(DMY_EUR, HMS),
        # Relative day offset - note: assume that the global variable
        # 'start' is set correctly ...
        token_def(RELDAY+TIME,                datetime_t(mk_seconds)),
        token_def(RELDAY+HMS,                 datetime_t(mk_seconds)),
        # Time durations
        token_def(r"(?P<neg>-)?(?P<d>\d+)d((?P<h>\d+)[hH])?((?P<m>\d+)[mM])?((?P<s>\d+(\.\d*)?)[sS])?", xformmg_t('duration', mk_seconds)),
        token_def(r"(?P<neg>-)?(?P<h>\d+)[hH]((?P<m>\d+)[mM])?((?P<s>\d+(\.\d*)?)[sS])?", xformmg_t('duration', mk_seconds)),
        token_def(r"(?P<neg>-)?(?P<m>\d+)[mM]((?P<s>\d+(\.\d*)?)[sS])?", xformmg_t('duration', mk_seconds)),
        token_def(r"(?P<neg>-)?(?P<s>\d+(\.\d*)?)[sS]", xformmg_t('duration', mk_seconds)),
        # regex      
        token_def(r"/[^/]+/i?",               xform_t('regex', regex2regex)),
        token_def(r"[0-9]+-[0-9]+(:[0-9]+)?", xform_t('irange', mk_intrange)),
        float_token(),
        int_token(),
        # Operators that just stand for themselves
        #token_def("~",                        simple_t('regexmatch')),
        token_def(r"(~|\blike\b)",            xform_t('regexmatch', lambda o, **k: lambda x, y: not re.match(y, x) is None)),
        token_def(r"(<=|>=|=|<|>)",           operator_t('compare')),
        token_def(r"-|\+|\*|/",               operator_t('operator')),
        token_def(r"\(",                      simple_t('lparen')),
        token_def(r"\)",                      simple_t('rparen')),
        token_def(r"\[",                      simple_t('lbracket')),
        token_def(r"\]",                      simple_t('rbracket')),
        token_def(r",",                       simple_t('comma')),
        # Textual stuff
        token_def(r"\$(?P<sym>[a-zA-Z][a-zA-Z_]*)",  resolve_t('external', 'sym')),
        token_def(r"'[^']*'",                 value_t('literal')),
        token_def(r"[:@\#%!\.a-zA-Z0-9_?|]+", value_t('text')),
        token_def(r"\s+",                     ignore_t())
    ]

    tokenizer  = mk_tokenizer(tokens, **kwargs)

    # The output of the parsing is a filter function that returns
    # True or False given a scan object

    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    ######  Our grammar

    # query    = modifier [ 'where' condition ['order by' sorting ] ['limit' int] ] eof
    # modifier = expr 'to' expr 
    # expr     = term '+' term | term '-' term | term '*' term | term '/' term | '(' expr ')'
    # term     = duration | number | property | external
    # duration = \d+ 'd'[\d+ 'h'][\d+ 'm'] [\d+ ['.' \d* ] 's'] |
    #                     \d+ 'h'[\d+ 'm'] [\d+ ['.' \d* ] 's'] |
    #                              \d+ 'm' [\d+ ['.' \d* ] 's'] |
    #                                        \d+ ['.' \d* ] 's'
    # number   = int|float
    # property = alpha char {alpha char | digit | '_'}    # will get property from scan object
    # external = '$' property                             # will look up value of property in global namespace

    # condition  = condexpr {relop condexpr} | 'not' condexpr | '(' condexpr ')'
    # sorting    = sortterm {',' sortterm}
    # sortterm   = identifier ['asc','desc' ]
    # condexpr   = property '~' (regex|text) | property compare expr | property 'in' list
    # compare    = '=' | '>' | '>=' | '<' | '<=' ;
    # relop      = 'and' | 'or' ;
    # list       = '[' [value {',' value}] ']'
    # value      = anychar {anychar}


    # regex      = '/' {anychar - '/'} '/' ['i']  ('i' is the case-insensitive match flag)
    # identifier = alpha {character} 
    # anychar    = character | symbol
    # character  = alpha | digit 
    # alpha      = [a-zA-Z_] ;
    # digit      = [0-9] ;

    # query    = expr 'to' expr [ 'where' condition ] eof
    def parse_query(s):
        if tok(s).type is None:
            raise SyntaxError, "empty query"
        # parse the scan start/end time modification function first
        perscan_f = parse_modifier(s)
        # if the parse left off at the 'where' keyword, we know what to do
        # note: the 'where' clause is optional and defaults to "all scans"
        # WHERE
        where     = tok(s)
        filter_f  = parse_condition(next(s)) if where.type=='where' else lambda x: True

        # "ORDER BY"
        orderby   = tok(s)
        if orderby.type=='order by':
            # only allow comma separated list of identifiers
            # 'parse_sort_list' will return a list of sort functions, in the order
            # in they were given
            sortlist = parse_sort_list(next(s))
            if not sortlist:
                raise SyntaxError, "No list of sort keys found (%s)" % tok(s)
            # good, 'sortlist' is a list of sort functions that need to be applied
            # on the list of filtered items
            orderby.value = lambda x: reduce(lambda acc, sortfn: sortfn(acc), sortlist, x)
        else:
            # no sorting
            orderby.value = lambda x: x
        # "LIMIT"
        limit     = tok(s)
        if limit.type=='limit':
            # we MUST be followed by an int
            next(s)
            ival = tok(s)
            if ival.type!='int':
                raise SyntaxError, "Only an integer is allowed after limit, not %s" % ival
            # consume the integer
            next(s)
            count = itertools.count()
            limit.value = lambda x: itertools.takewhile(lambda obj: count.next()<ival.value, x)
        else:
            limit.value = lambda x: x

        # the only token left should be 'eof' AND, after consuming it,
        # the stream should be empty. Anything else is a syntax error
        try:
            if tok(s).type is None:
                next(s)
        except StopIteration:
            return (perscan_f, lambda scans: limit.value(orderby.value(filter(filter_f, scans))))
        raise SyntaxError, "Tokens left after parsing %s" % tok(s)

    # modifier = expr 'to' expr 
    def parse_modifier(s):
        # we require two functions to be generated, the start_time_fn (before 'to')
        # and the end_time_fn (after 'to' ...)
        depth         = s.depth
        start_time_fn = parse_expr(s)
        #start_time_fn = parse_expr(s, None)
        if s.depth!=depth:
            raise SyntaxError, "Unbalanced parenthesis %s" % tok(s)
        # we now MUST see the 'to' keyword
        to = tok(s)
        if to is None or to.type!='to':
            raise SyntaxError, "Unexpected token %s (expected 'to' keyword)" % tok(s)
        # do not forget to consume the 'to' keyword ...
        depth         = s.depth
        end_time_fn   = parse_expr(next(s))
        if s.depth!=depth:
            raise SyntaxError, "Unbalanced parentheses %s (expect depth %d, found %d)" % (tok(s), depth, s.depth)
        def this_fn(scan):
            s_time = start_time_fn(scan)
            e_time = end_time_fn(scan)
            if e_time<s_time:
                raise RuntimeError, "Scan time selection error: end time is before start time in scan\n   {0}".format(scan)
            return (s_time, e_time)
        return this_fn
        #return lambda scan: (start_time_fn(scan), end_time_fn(scan))

    # expr     = term | expr '+' expr | expr '-' expr | expr '*' expr | expr '/' expr | '(' expr ')' | '-' expr
    def parse_expr(s, unary=False):
        t = tok(s)
        depth = s.depth

        if t.type in ['lparen', 'rparen']:
            lterm = parse_paren(s)
        elif t.type=='operator' and t.value is mk_operator('-'):
            # unary '-'
            tmpexpr = parse_expr(next(s), unary=True)
            lterm   = lambda scan: operator.neg( tmpexpr(scan) )
        else:
            lterm = parse_term(s)

        # If we see an operator, we must parse the right-hand-side
        # (our argument is the left-hand-side
        # Well ... not if we're doing unary parsing!
        # if we saw unary '-' then we should parse parens and terms up until
        # the next operator
        oper = tok(s)
        if oper.type=='operator':
            if unary:
                return lterm
            if lterm is None:
                raise SyntaxError, "No left-hand-side to operator %s" % oper
            rterm = parse_expr(next(s))
            if rterm is None:
                raise SyntaxError, "No right-hand-side to operator %s" % oper
            return lambda scan: oper.value(lterm(scan), rterm(scan))
        elif oper.type in ['int', 'float', 'duration', 'datetime']:
            # negative numbers as right hand side are not negative numbers
            # but are operator '-'!
            # so, subtracting a number means adding the negative value (which we already
            # have god)
            # Consume the number and return the operator add
            next(s)
            return lambda scan: operator.add(lterm(scan), oper.value)
        # neither parens, terms, operators?
        return lterm

    def parse_paren(s):
        lparen = tok(s)
        if lparen.type!='lparen':
            raise RuntimeError, "Entered parse_paren w/o left paren but %s" % lparen
        depth   = s.depth
        s.depth = s.depth + 1
        # recurse into parsing the expression - and do NOT forget to consume the lparen!
        expr    = parse_expr(next(s))
        # now we should be back at the same depth AND we should be seeing rparen
        rparen  = tok(s)
        if rparen.type=='rparen':
            s.depth = s.depth - 1
            next(s)
        return expr

    # term     = duration | number | property | external
    def parse_term(s):
        term = tok(s)
        # The easy bits first
        if term.type in ['int', 'float', 'external', 'duration', 'datetime', 'text']:
            if term.type=='text':
                def attrib_or_value(scan):
                    if hasattr(scan, term.value):
                        return getattr(scan, term.value)
                    else:
                        return term.value
                rv = attrib_or_value
            else:
                rv = lambda scan: term.value
            # all's well - eat this term
            next(s)
            return rv
        elif term.type=='literal':
            # ok, allowed to consume it
            next(s)
            # note that we strip the leading and closing single quote
            rv = lambda scan: term.value[1:-1]
            return rv
        return None

    # condition  = condexpr {relop condexpr} | 'not' condition | '(' condition ')'
    # relop      = 'and' | 'or' ;
    def parse_condition(s):
        token = tok(s)

        # Recurse if we need to
        if token.type in ['lparen', 'rparen']:
            lterm = parse_paren_condition(s)
        # 'not' expr
        elif token.type=='not':
            # parse the next expr and negate it
            # we MUST have a next one
            condition = parse_condition(next(s))
            if condition is None:
                raise SyntaxError, "Missing expression after 'not' %s" % condition
            lterm = lambda scan: operator.not_( condition(scan) )
        else:
            # it must be a condexpr
            lterm = parse_cond_expr(s)

        # If we now see a relop, we have to parse another condition
        relop = tok(s)
        if relop.type!='relop':
            return lterm

        # consume the relop & parse the condition
        rterm = parse_condition(next(s))

        if lterm is None:
            raise SyntaxError, "Missing left-hand-condition to relational operator (%s)", relop
        if rterm is None:
            raise SyntaxError, "Missing right-hand-condition to relational operator (%s)", relop

        # and return the combined operation
        return lambda scan: relop.value(lterm(scan), rterm(scan))


    # condexpr   = expr '~' (regex|text) | expr compare expr | expr 'in' list
    # compare    = '=' | '>' | '>=' | '<' | '<=' ;
    def parse_cond_expr(s):
        token = tok(s)
        # No matter what, we have a left and a right hand side
        # separated by an operator
        lterm   = parse_expr(s)
        if lterm is None:
            raise SyntaxError, "Failed to parse left-hand-term of cond_expr (%s)" % tok(s)
        # Now we must see a comparator
        compare = tok(s)
        if not compare.type in ['compare', 'regexmatch', 'in']:
            raise SyntaxError, "Expected a comparison operator, regex match or 'in' keyword, got %s" % compare
        # consume the comparison
        next(s)
        # do some processing based on the type of operator
        if compare.type=='in':
            rterm  = parse_list(s)
        elif compare.type=='compare':
            rterm = parse_expr(s)
        else:
            # must've been regexmatch
            rterm = parse_rx(s)
        # it better exist
        if rterm is None:
            raise SyntaxError, "Failed to parse right-hand-term of cond_expr (%s)" % tok(s)
        return lambda scan: compare.value(lterm(scan), rterm(scan))

    def parse_paren_condition(s):
        lparen = tok(s)
        if lparen.type!='lparen':
            raise RuntimeError, "Entered parse_paren_condition w/o left paren but %s" % lparen
        depth   = s.depth
        s.depth = s.depth + 1
        # recurse into parsing the expression - and do NOT forget to consume the lparen!
        expr    = parse_condition(next(s))
        # now we should be back at the same depth AND we should be seeing rparen
        rparen  = tok(s)
        if rparen.type=='rparen':
            s.depth = s.depth - 1
            next(s)
        return expr

    def parse_rx(s):
        # we accept string, literal and regex and return an rx object
        rx = tok(s)
        if not rx.type in ['regex', 'text', 'literal']:
            raise SyntaxError, "Failed to parse string matching regex (not regex, text or literal but %s)" % rx
        # consume the token
        next(s)
        if rx.type=='literal':
            # extract the pattern from the literal (ie strip the leading/trailing "'" characters)
            rx.value = rx.value[1:-1]
        if rx.type in ['text', 'literal']:
           rx.value = pattern2regex(rx.value) 
        return lambda scan: rx.value

    def parse_list(s):
        bracket = tok(s)
        if bracket.type != 'lbracket':
            raise SyntaxError, "Expected list-open bracket ('[') but found %s" % bracket
        rv = []
        # keep eating text + ',' until we read 'rbracket'
        next(s)
        while tok(s).type!='rbracket':
            # if we end up here we KNOW we have a non-empty list because
            # the next token after '[' was NOT ']'
            # Thus if we need a comma, we could also be seeing ']'
            needcomma        = len(rv)>0
            #print " ... needcomma=",needcomma," current token=",tok(s)
            if needcomma:
                if tok(s).type=='rbracket':
                    continue
                if tok(s).type!='comma':
                    raise SyntaxError, "Badly formed list at {0}".format(tok(s))
                # and eat the comma
                next(s)
            # now we need a value. 'identifier' is also an acceptable blob of text
            rv.extend( parse_list_item(s) )
            #print "parse_list: ",rv
        # and consume the rbracket (if not rbracket a syntax error is raised above)
        next(s)
        return lambda scan: rv

    # always returns a list-of-items; suppose the list item was an irange
    def parse_list_item(s):
        t = tok(s)
        # current token must be 'text' or 'irange'
        if not t.type in ['text', 'irange', 'int', 'float', 'literal']:
            raise SyntaxError, "Failure to parse list-item {0}".format(t)
        next(s)
        # for a literal, strip the leading and closing single quote
        if t.type == 'literal':
            t.value = t.value[1:-1]
        return t.value if t.type == 'irange' else [t.value]

    # attribute list = identifier {',' identifier}
    def parse_sort_list(s):
        rv = []
        rxAttribute = re.compile("^[a-zA-Z][a-zA-Z0-9_]*$")
        while True:
            item = tok(s)
            if item.type!='text':
                raise SyntaxError, "attribute list may only contain strings, found %s" % item
            if not rxAttribute.match(item.value):
                raise SyntaxError, "%s is not a valid attribute name" % item
            # Peek at the next token. If it's asc/desc take that into account
            next(s)
            order = tok(s)
            if order.type in ['asc', 'desc']:
                # consume it
                next(s)
            else:
                order.type = 'asc'
            # create a sorting function
            def mk_sf(attr, order):
                def do_it(x):
                    return sorted(x, key=operator.attrgetter(attr), reverse=(order=='desc'))
                return do_it
            rv.append( mk_sf(item.value, order.type) )

            #if we don't see a comma next, we break
            if tok(s).type!='comma':
                break
            # consume the comma
            next(s)
        # primary sort key is now first in list but for the sorting to work in steps
        # (see https://wiki.python.org/moin/HowTo/Sorting ) we must apply the sorting
        # functions in reverse order
        return reversed(rv)
        

    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

    tokenizer  = mk_tokenizer(tokens, **kwargs)
    return parse_query(state_type(tokenizer(qry)))


# Time (range) grammar - we must support some arithmetic
#
# timerange  = expr { 'to' ('+' duration | expr )}
# expr       = term '+' term | term '-' term |  term '*' term | term '/' term | '(' expr ')'
# term       = number | identifier | datetime | duration | reltime
# datetime   = year '/' month '/' day [T/] timestamp | day '-' month '-' year [T/] timestamp | day '-' monthstr '-' year [T/] timestamp
# reltime    = {'-'} digit {digit} '/' timestamp
# reltime    = {'-'} digit {digit} '/' duration
# year       = 4 * digit 
# month      = 2 * digit
# monthstr   = 3 * alpha char
# day        = 2 * digit
# timestamp  = digit {digit} [hH] digit {digit} [mM] digit {digit} {'.' digits } [sS]) | 
#              digit {digit} ':' digit {digit} ':' digit {digit} {'.' digits}
# duration   = number [hH] { number [mM] { number ['.' number] [sS] } } | number [mM] { number ['.' number] [sS] } | number ['.' number] [sS]
# scan prop  = 'scan' digits '.' identifier
# number     = {'-'} {[0-9]+}{'.'}[0-9]+{[eE]{-}[0-9]+}
# identifier = alpha char {alpha char | digit | '_' }

# digits    = digit {digit}
# digit     = [0-9]
# alpha char = [a-zA-Z]
SEC   = r"(?P<s>\d+(\.\d*)?)[sS]"
MIN   = r"(?P<m>\d+)[mM]"
HR    = r"(?P<h>\d+)[hH]"
DAY   = r"(?P<d>\d+)d"

DUR4  = DAY+MAYBE(HR)+MAYBE(MIN)+MAYBE(SEC)
DUR3  = HR+MAYBE(MIN)+MAYBE(SEC)
DUR2  = MIN+MAYBE(SEC)
DUR1  = SEC

def parse_time_expr(txt, **env):
    # Helper functions

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        # keywords
        token_def(r"\bto\b",                  keyword_t()),
        # Date + time formats
        datetime_token(YMD,     TIME),
        datetime_token(YMD,     HMS),
        datetime_token(DMY,     TIME),
        datetime_token(DMY,     HMS),
        datetime_token(DMY_EUR, TIME),
        datetime_token(DMY_EUR, HMS),
        # Relative day offset - note: assume that the global variable
        # 'start' is set correctly ...
        token_def(RELDAY+TIME,  datetime_t(mk_seconds)),
        token_def(RELDAY+HMS,   datetime_t(mk_seconds)),
        # Time durations
        token_def(RELDAY+DUR3,  datetime_t(mk_seconds)),
        token_def(RELDAY+DUR2,  datetime_t(mk_seconds)),
        token_def(RELDAY+DUR1,  datetime_t(mk_seconds)),
        token_def(DUR4,         xformmg_t('duration', mk_seconds)),
        token_def(DUR3,         xformmg_t('duration', mk_seconds)),
        token_def(DUR2,         xformmg_t('duration', mk_seconds)),
        token_def(DUR1,         xformmg_t('duration', mk_seconds)),

        # Operators and semantic elements that just stand for themselves
        token_def(r"-|\+|\*|/",                     operator_t('operator')),
        token_def(r"\(",                            simple_t('lparen')),
        token_def(r"\)",                            simple_t('rparen')),
        token_def(r',',                             simple_t('comma')),
        # we don't care about int's or float's - any number we'll accept
        number_token(),
        token_def(r"\$(?P<sym>[a-zA-Z][a-zA-Z_]*)", resolve_t('external', 'sym')),
        token_def(r"\s+",                           ignore_t())
    ]

    # shorthands that work on the parser state 's'
    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    # timeranges = timerange {',' timerange}
    # timerange = expr ['to' expr)]
    def parse_time_ranges(s):
        rv = []
        while True:
            rv.append( parse_time_range(s) )
            # Check for more time ranges
            nxt = tok(s)
            if nxt.type=='comma':
                # consume the comma and continue
                next(s)
                continue
            break
        # Ok, we should now see 'eof' and an empty stream
        try:
            if tok(s).type is None:
                next(s)
        except StopIteration:
            return rv
        raise SyntaxError, "Tokens left after parsing %s" % tok(s)

    # timerange = expr ['to' expr ]
    def parse_time_range(s):
        # we require two functions to be generated, the start_time_fn (before 'to')
        # and the end_time_fn (after 'to' ...)
        depth      = s.depth
        start_time = parse_expr(s)
        if s.depth!=depth:
            raise SyntaxError, "Unbalanced parenthesis %s" % tok(s)
        if start_time is None:
            raise SyntaxError, "Missing start-time expression %s" % tok(s)
        # If we don't see the 'to' keyword, it's a single time stamp
        to = tok(s)
        if to.type!='to':
            return (start_time, start_time)

        # insert the current rterm value in the environment such that
        # the 'end_time' may use "+ duration" 
        env['$#parsed:start^time#$'] = start_time
        # parse the end time after consuming the 'to' keyword
        depth      = s.depth
        end_time   = parse_expr(next(s))
        # remove the value from the environment again
        del env['$#parsed:start^time#$']
        if s.depth!=depth:
            raise SyntaxError, "Unbalanced parentheses %s (expect depth %d, found %d)" % (tok(s), depth, s.depth)
        if end_time is None:
            raise SyntaxError, "Missing end-time expression %s" % tok(s)
        if end_time<start_time:
            t = tok(s)
            raise RuntimeError, "{0}\n{1:>{2}s}^\nend time is before start time here".format(txt, "", len(txt) if t.position<0 else t.position)
        return (start_time, end_time)

    # expr     = term | expr '+' expr | expr '-' expr | expr '*' expr | expr '/' expr | '(' expr ')' | '-' expr
    def parse_expr(s, unary=False):
        t = tok(s)
        depth = s.depth

        if t.type in ['lparen', 'rparen']:
            lterm = parse_paren(s)
        elif t.type=='operator' and t.value is mk_operator('-'):
            # unary '-'
            tmpexpr = parse_expr(next(s), unary=True)
            lterm   = operator.neg( tmpexpr )
        elif t.type=='operator' and t.value is mk_operator('+'):
            # unary '+' - may be followed by an expression; we add the parsed time
            # to whatever the start time was
            next(s)
            duration = parse_expr(s)
            return env['$#parsed:start^time#$'] + duration
        else:
            lterm = parse_term(s)

        # If we see an operator, we must parse the right-hand-side
        # (our argument is the left-hand-side
        # Well ... not if we're doing unary parsing!
        # if we saw unary '-' then we should parse parens and terms up until
        # the next operator
        oper = tok(s)
        if oper.type=='operator':
            if unary:
                return lterm
            if lterm is None:
                raise SyntaxError, "No left-hand-side to operator %s" % oper
            rterm = parse_expr(next(s))
            if rterm is None:
                raise SyntaxError, "No right-hand-side to operator %s" % oper
            return oper.value(lterm, rterm)
        elif oper.type in ['int', 'float', 'duration', 'datetime']:
            # negative numbers as right hand side are not negative numbers
            # but are operator '-'!
            # so, subtracting a number means adding the negative value (which we already
            # have god)
            # Consume the number and return the operator add
            next(s)
            return operator.add(lterm, oper.value)
        # neither parens, terms, operators?
        return lterm

    def parse_paren(s):
        lparen = tok(s)
        if lparen.type!='lparen':
            raise RuntimeError, "Entered parse_paren w/o left paren but %s" % lparen
        depth   = s.depth
        s.depth = s.depth + 1
        # recurse into parsing the expression - and do NOT forget to consume the lparen!
        expr    = parse_expr(next(s))
        # now we should be back at the same depth AND we should be seeing rparen
        rparen  = tok(s)
        if rparen.type=='rparen':
            s.depth = s.depth - 1
            next(s)
        return expr

    # term     = duration | number | external
    def parse_term(s):
        term = tok(s)
        # The easy bits first
        if term.type in ['number', 'external', 'duration', 'datetime']:
            # all's well - eat this term
            next(s)
            return term.value
        return None

    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

    tokenizer  = mk_tokenizer(tokens, **env)
    return parse_time_ranges(state_type(tokenizer(txt)))

# Parse a simple duration (sort of VEX format):
#    ...y..d..h..m....s
# there must be at least one unit present, trailing fields after the highest order unit are optional.
# only accepts units in this order; e.g. cannot say 10s1h 
MINf   = r"(?P<m>\d+(\.\d*)?)[mM]"
HRf    = r"(?P<h>\d+(\.\d*)?)[hH]"
DAYf   = r"(?P<d>\d+(\.\d*)?)d"
DUR4f  = DAYf+MAYBE(HRf)+MAYBE(MINf)+MAYBE(SEC)
DUR3f  = HRf+MAYBE(MINf)+MAYBE(SEC)
DUR2f  = MINf+MAYBE(SEC)

def parse_duration(txt, **env):
    # Helper functions

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        # Time durations
        token_def(DUR4f,         xformmg_t('duration', mk_seconds)),
        token_def(DUR3f,         xformmg_t('duration', mk_seconds)),
        token_def(DUR2f,         xformmg_t('duration', mk_seconds)),
        token_def(DUR1,          xformmg_t('duration', mk_seconds)),
        token_def(r"\S+",        lambda o, **kwargs: token_type('gunk', o.group(0)))
    ]

    # shorthands that work on the parser state 's'
    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    # duration = ..y..d..h..m..s
    def parse_time_duration(s):
        dur = tok(s)
        if dur.type!='duration':
            raise SyntaxError, "This is not a duration - %s" % dur
        next(s)
        try:
            if tok(s).type is None:
                next(s)
        except StopIteration:
            return dur.value
        raise SyntaxError, "Tokens left after parsing %s" % tok(s)

    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

    tokenizer  = mk_tokenizer(tokens, **env)
    return parse_time_duration(state_type(tokenizer(txt)))



########################################################################################################
#
#  data set expression parser
#  allows manipulation of results of plotiterator - e.g. differencing
#
#########################################################################################################

#   <dscmd>  = 'store' {<expr>} 'as' <id> | 'load' <expr>

#   <expr>      = <expr> '+' <term> | <expr> '-' <term> | <term>
#   <term>      = <term> '*' <factor> | <term> '/' <factor> | <factor>
#   <factor>    = <exponent> '^' <factor> | <exponent>
#   <exponent>  = '-' <exponent> | <final>
#   <final>     = <number> | <id> | <id> '(' <arglist> ')' | '(' <expr> ')' | <id> '[' <subscript> ']'
#   <arglist>   = <empty> | <expr> {',' <expr> }
#   <subscript> = <filter> {',' <filter> }
#   <filter>    = <attribute> '=' <value>
#   <attribute> = 'p' | 'ch' | 'sb' | 'src' | 'time' | 'bl' | 'fq'
#                 'P' | 'CH' | 'SB' | 'SRC' | 'TIME' | 'BL' | 'FQ'
#   <value>     = <int> | <alnum>  # may/should/will depend on type of attribute!
#
#   <number> = <int> | <float>
#   <int>    = [0-9]+
#   <float>  = [0-9]+'.'[0-9]* | '.'[0-9]+
#   <id>     = <alnum>{'.'<alnum>}
#   <alnum>  = [a-zA-Z_][a-zA-Z0-9_]*

# turn datasets[<name>] into an iterator over all contained data sets,
# yielding (label, dataset) tuples
def ds_iter(value):
    for plot in value.keys():
        for dataset in value[plot].keys():
            yield (plots.join_label(plot, dataset), value[plot][dataset])
    raise StopIteration


methodwrappert = type({}.__delitem__)
def isAttr(o):
    # in fact, 'inspect.is<predicate>' predicates are useless. They still return all
    # the members of e.g. a "dict()". Because all of the methods of 'dict' are now
    # of type <method-wrapper>. Gah!
    #return not (inspect.isfunction(o) or inspect.ismethod(o))
    return not (inspect.isbuiltin(o) or o is methodwrappert)

def copy_attributes(outp, inp):
    map(lambda (a,v): setattr(outp, a, v), \
            map(lambda (a, tp): (a, getattr(inp, a)), \
                filter(lambda (nm, tp): not nm.startswith('__'), inspect.getmembers(inp, isAttr))))
    return outp

def ds_flat_filter(value, tp=None, subscript=None):
    rv                 = copy_attributes(plots.Dict(), value)
    (mklabf, subquery) = (lambda x: x, lambda x: True) if subscript is None else subscript
    for ds in (value.keys() if tp is None else filter(lambda k: k.TYPE==tp, value.keys())):
        # Do we accept this dataset?
        if not subquery(ds):
            continue
        dsref      = value[ds]
        # sort by x-value!
        (xs, ys)   = zip( *sorted(zip(dsref.xval, dsref.yval), key=operator.itemgetter(0)) )
        dsref.xval = numpy.array(xs)
        dsref.yval = numpy.array(ys)
        # if anames is set, it means we've filtered/subindexed so we must create a new label with
        # the indicated anames set to None [such that the crossmatching on those won't fail]
        nds        = mklabf(ds)
        rv[nds]    = value[ds]
    return rv

def ds_key_filter(value, keys):
    rv = copy_attributes(plots.Dict(), value)
    for k in keys:
        rv[k] = value[k]
    return rv

dictType  = type({})
isDataset = lambda x: isinstance(x, dictType)

def normal_apply(l, f, r):
    return (None, (l.xval, f(l.yval, r.yval)))

def shortest_apply(l, f, r):
    n = min(len(l.yval), len(r.yval))
    return ("Truncated to {0} elements".format(n), (l.xval[:n], f(l.yval[:n], r.yval[:n])))

isect_table = {
    # ( <lengths equal>, <one of 'm has length '1'>)
    (True,  False): normal_apply,   # when both lengths are equal it doesn't matter how long
    (True,   True): normal_apply,   #                   ..
    (False,  True): normal_apply,   # if unequal lengths but one of'm has is length '1'
    (False, False): shortest_apply  # only apply to first 'n' elements
}

def do_isect(d0, f, d1):
    # we know both d0 and d1 are flattened datasets
    # so we must iterate over the set of identical keys
    # for each key we apply the operation to the y-part of the datasets
    def app(acc, key):
        # make sure they're numarrays
        # [note: .as_numarray() does not create a new object, just returns 'self']
        ds0 = d0[key]#.as_numarray()
        ds1 = d1[key]#.as_numarray()
        l0  = len(ds0.yval)
        l1  = len(ds1.yval)
        # compare lengths (...) and decide what to do
        (msg, res) = isect_table[(l0==l1, l0==1 or l1==1)](ds0, f, ds1)
        if msg is not None:
            print "{0}: {1}".format(key, msg)
        pref = acc.setdefault(key, plots.Dict())
        pref.xval = res[0]
        pref.yval = res[1]
        return acc
    return reduce(app, set(d0.keys()) & set(d1.keys()), copy_attributes(plots.Dict(), d0))

# implement infix operator 'f' on two datums
def immediate_apply(l, f, r):
    return f(l, r)

def do_iterate(d0, f, d1):
    # we know that either d0 or d1 is a dataset
    d0isdata = isDataset(d0)
    gen      = d0.iteritems() if d0isdata else d1.iteritems()
    # apply in the correct order!
    app      = (lambda d: f(d, d1)) if d0isdata else (lambda d: f(d0, d))
    def reductor(acc, (k, ds)):
        pref = acc.setdefault(k, plots.Dict())
        pref.xval = ds.xval
        pref.yval = app(ds.yval)
        return acc
    return reduce(reductor, gen, copy_attributes(plots.Dict(), d0 if d0isdata else d1))

applicator_table = { 
    # infix:  lhs <operator> rhs
    # table key is:
    #      ( <isDataset lhs>, <isDataset rhs> )
    (False, False): immediate_apply , # short-circuit direct evaluation of non-datasets
    (True,  False): do_iterate,       # one of'm is a dataset
    (False,  True): do_iterate,       #      id.
    (True,   True): do_isect          # both are datasets, must intersect
}

def applicator(d0, f, d1):
    # for both arguments we want a set of keys such that we can get the intersection
    # of identical keys. But that's only if both of 'm are datasets
    # otherwise it's either just numbers that are combined or one of them is a data set
    return applicator_table[(isDataset(d0), isDataset(d1))](d0, f, d1)


def mk_dataset(ds, an):
    if not isDataset(ds):
        return ds
    if an is not None:
        ds.msname = copy.deepcopy(an)
    return ds


def parse_dataset_expr(txt, datasets, **env):
    ident       = r"[a-zA-Z_][a-zA-Z0-9_]*"
    identifier  = NAMED("name", ident)+MAYBE(r"\."+NAMED("type", ident))
    unary_minus = mk_operator('-')

    # extract the expression 
    annotation = hvutil.sub(txt, [(r"^\s*load\s*", ""), (r"^\s*store\s*", ""), (r"\bas\b.*", "")]).strip()

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        # keywords
        token_def(r"\b(store|load|as)\b",           keyword_t()),
        # the attribute names
        #(re.compile(r"\b(p|ch|sb|fq|bl|time|src)\b", re.I), value_t('attrname')),
        (re.compile(r"\b(p|ch|sb|bl|src)\b", re.I), value_t('attrname')),
        # operators
        token_def(r"-|\+",                          operator_t('additive')),
        token_def(r"\*|/",                          operator_t('multiplicative')),
        token_def(r"\^",                            operator_t('exponent')),
        token_def(r",",                             simple_t('comma')),
        token_def(r"=",                             simple_t('equal')),
        token_def(r"\(",                            simple_t('lparen')),
        token_def(r"\)",                            simple_t('rparen')),
        token_def(r"\[",                            simple_t('lbracket')),
        token_def(r"\]",                            simple_t('rbracket')),
        # identifiers (function call) and variables: <name>{.<type>} or <name>{.<type>}
        token_def(identifier,                       xformmg_t('id',     lambda mo: mo.groupdict())),
        token_def(ident,                            value_t('text')),
        #token_def(r"\$"+dsid,                       value_t('dsid')),
        # numbers
        number_token(),
        # not particularly interested in whitespace
        token_def(r"\s+",                           ignore_t())
        # Time durations
#        token_def(DUR4f,         xformmg_t('duration', mk_seconds)),
#        token_def(DUR3f,         xformmg_t('duration', mk_seconds)),
#        token_def(DUR2f,         xformmg_t('duration', mk_seconds)),
#        token_def(DUR1,          xformmg_t('duration', mk_seconds)),
#        token_def(r"\S+",        lambda o, **kwargs: token_type('gunk', o.group(0)))
    ]

    # shorthands that work on the parser state 's'
    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

#    def parse_dataset_expr_impl(s):
#        try:
#            while True:
#                t = tok(s)
#                print "[{0}/{1}] ".format(t.type, t.value)
#                if t.type is None:
#                    break
#                next(s)
#            next(s)
#            raise SyntaxError, "Tokens left after 'None' [tp={0}]".format(tok(s).type)
#        except StopIteration:
#            pass
#        #raise RuntimeError,"Not implemented yet"

    # entry point of grammar
    def parse_dataset_expr_impl(s):
        # only 'load' or 'store' are supported at this point
        supported = { 'load': parse_load_dataset, 
                      'store': parse_store_dataset }
        cur = tok(s)
        if cur.type not in supported:
            raise SyntaxError, "Unexpected token {0} in stead of {1}".format(cur.type, supported.keys())
        else:
            # eat up the token and dive in
            rv = supported[cur.type]( next(s) )
            # After parsing we should see EOF and then nothing
            try:
                next(s)
                if tok(s).type is None:
                    next(s)
                else:
                    raise SyntaxError, "Trailing token(s) {0}".format( tok(s).value )
            except StopIteration:
                # fine
                pass
            return rv

    def parse_store_dataset(s):
        # 'store' {expr} 'as' <id>
        # 'store' <expr> {'as' <id>}
        # look at the next token, it could be either 
        # an expression or the 'as' keyword, to indicate
        # that the current plots need to be stored
        cur  = tok( s )
        expr = None
        if cur.type != 'as':
            # we expect an expression here
            expr = parse_expr( s )
            if s.depth!=0:
                raise SyntaxError, "Unbalanced parenthesis"
        cur  = tok( s )
        name = None
        if cur.type == 'as':
            # eat up the 'as' keyword and expect an id
            next(s)
            cur = parse_id( s )
            if cur.type != 'id':
                raise SyntaxError, "Unexpected token {0} [expected variable name]".format( cur.value )
            # verify that dsid does not address a subset $id.id but just $id
            if cur.value['type'] is not None:
                raise SyntaxError, "Cannot store an expression as subset {0}".format( cur.value['type'] )
            if 'filter' in cur.value:
                raise SyntaxError, "Cannot store an expression as filtered subset of {0}".format( cur.value['name'] )
            name = cur.value['name']
        elif cur.type is not None:
            raise SyntaxError, "Unexpected token {0} [expected 'as' or nothing]".format( cur.value )

        # If both name and expr as None, that is a syntax error
        if expr is None and name is None:
            raise SyntaxError, "Empty store command?!"
        # if either is None, we can provide defaults for that
        def mk_f(expression, nm):
            def do_it(ds):
                an   = annotation
                expr = expression
                if expr is None:
                    expr = lambda ds: ds_flat_filter(ds['_'])
                    an   = None
                else:
                    if re.match(r"^\$[a-zA-Z_\.]+$", an):
                        an = None
                n = nm
                if n is None:
                    n = '_'
                # decorate with annotation if necessary
                ds[n] = mk_dataset(expr(ds), an)
                return ds[n]
            return do_it
        return mk_f(expr, name)

    def parse_load_dataset(s):
        # 'load' expr
        def mk_f(expression):
            def do_it(ds):
                an = annotation
                if re.match(r"^\$[a-zA-Z_\.]+$", an):
                    an = None
                # decorate with annotation, if necessary
                ds['_'] = mk_dataset(expression(ds), an)
                return ds['_']
            return do_it
        expr = parse_expr( s )
        if s.depth!=0:
            raise SyntaxError, "Unbalanced parenthesis"
        return mk_f( expr )


    #@argprint
    #   <name>{.<type>}{'[' <filter> ']'}
    def parse_id(s):
        print "parse_id"
        # we should *at least* be looking at an 'id' token
        cur = tok( s )
        if cur.type != 'id':
            raise SyntaxError, "Expected an identifier here"
        # look ahead to see if we find '[' which would indicate subscripting/filtering
        next(s)
        if tok(s).type == 'lbracket':
            cur.value['filter'] = parse_filter( s )
        return cur

    #@argprint
    #   '[' <filter> ']' 
    def parse_filter(s):
        # check if we indeed are looking at start of filter/subscripting and if so eat up that token
        if tok(s).type != 'lbracket':
            raise RuntimeError, "Entered parse_filter() but not looking at '['?"
        next(s)
        # now we must see a comma separated list of <attr> = <value>
        rv = []
        while tok(s).type!='rbracket':
            # if we end up here we KNOW we have a non-empty list because
            # the next token after '[' was NOT ']'
            # Thus if we need a comma, we could also be seeing ']'
            needcomma        = len(rv)>0
            if needcomma:
                if tok(s).type=='rbracket':
                    continue
                if tok(s).type!='comma':
                    raise SyntaxError, "Badly formed list at {0}".format(tok(s))
                # and eat the comma
                next(s)
            # now we need a list item "<attr> = <value>"
            rv.append( parse_list_item(s) )
        # and consume the rbracket (if not rbracket a syntax error is raised above)
        next(s)
        # convert the list of attribute matchers into a single match fn
        def mk_match_f(l):
            def do_it(ds):
                return all([cond(ds) for cond in l])
            return do_it
        def mk_lab_f(l):
            def do_it(ds):
                rv = copy.deepcopy(ds)
                map(lambda y: setattr(rv, y, None), l)
                return rv
            return do_it
        (anames, matchfns) = zip(*rv)
        return (mk_lab_f(anames), mk_match_f(matchfns))

    #@argprint
    #  <attribute> '=' <value>
    # returns function which matches label attribute value to <value>
    def parse_list_item(s):
        attrtype_dict  = { 'P':str,         'CH':int,         'SB':int,         'BL':re.compile, 'SRC':re.compile }
        attrmatch_dict = { 'P':operator.eq, 'CH':operator.eq, 'SB':operator.eq, 'BL':re.match,   'SRC':re.match }
        attrname = tok(s)
        if attrname.type!='attrname':
            raise SyntaxError, "Expected attribute name but found {0}".format( attrname.type )
        attrname = attrname.value.upper()
        # must see '='
        if tok(next(s)).type!='equal':
            raise SyntaxError, "Expected '=' but found {0}".format( tok(s).type )
        # next we should see <int> or <alnum>
        attrval = tok( next(s) )
        if attrval.type not in ['id', 'number', 'text']:
            raise SyntaxError, "Expected a number or text but found {0}".format( attrval.type )
        # ok, eat that one up
        next(s)
        def mk_amatch_f(aname, aval):
            # convert once
            aval = attrtype_dict[aname](aval)
            def do_it(ds):
                return attrmatch_dict[aname](aval, getattr(ds, aname))
            return do_it
        return (attrname, mk_amatch_f(attrname, attrval.value if attrval.type in ['number','text'] else attrval.value['name']))

    #@argprint
    def parse_expr( s ):
        # <expr> '+' <term> | <expr> '-' <term> | <term>
        expr = parse_term( s )
        if expr is None:
            return None

        # If we're looking at an additive operator now,
        # we must parse the rhs
        cur = tok( s )
        if cur.type=='additive':
            # eat this token
            next( s )
            rhs = parse_expr( s )
            if rhs is None:
                raise SyntaxError, "Expected a term, got {0}".format( tok(s).value )
            def mk_f(l, o, r):
                def do_it(ds):
                    return applicator(l(ds), o, r(ds))
                return do_it
            expr =  mk_f(expr, cur.value, rhs)
        return expr

    #@argprint
    def parse_term( s ):
        #   <term>   = <term> '*' <factor> | <term> '/' <factor> | <factor>
        term = parse_factor( s )
        # if no lhs (yet) we must check for a factor
        if term is None:
            return None

        # Now we could be looking a multiplicative operator
        cur = tok( s )
        if cur.type=='multiplicative':
            # eat it and try to parse a factor
            next( s )
            rhs = parse_term( s )

            # oh noes!
            if rhs is None:
                raise SyntaxError, "Expected a factor, got {0}".format( tok(s).value )

            def mk_f(l, o, r):
                def do_it(ds):
                    return applicator(l(ds), o, r(ds))
                return do_it
            term = mk_f(term, cur.value, rhs)
        return term

    #@argprint
    def parse_factor(s):
        #   <factor> = <exponent> '^' <factor> | <exponent>
        # if no exponent yet, look for one
        exponent = parse_exponent(s)
        if exponent is None:
            return None

        # if we're looking at '^', we must parse a factor
        cur = tok( s )
        if cur.type=='exponent':
            # eat the token and try to parse another exponent
            next( s )
            factor = parse_factor(s)
            if factor is None:
                raise SyntaxError, "Expected an exponent, got {0}".format( tok(s).value )

            def mk_f(e, o, f):
                def do_it(ds):
                    return applicator(e(ds), o, f(ds))
                return do_it
            exponent = mk_f(exponent, cur.value, factor)
        return exponent

    #@argprint
    def parse_exponent(s):
        #   <exponent>  = '-' <exponent> | <final>
        cur = tok( s )
        if cur.type=='additive' and cur.value==unary_minus:
            # eat the minus sign
            next( s )
            exponent = parse_exponent(s)
            def mk_f(e):
                def do_it(ds):
                    return applicator(-1, operator.mul, e(ds))
                return do_it
            exponent = mk_f(exponent)
        else:
            # must be a final then?
            exponent = parse_final(s)
        return exponent

    #@argprint
    def parse_final(s):
        #   <final>  = <number> | <id> | <id> '(' <expr> ')' | '(' <expr> ')'
        # look at current token
        final = tok( s )

        # Check if we recognize it
        if final.type=='number':
            # ok, we know we recognize the token, let's eat the number 
            next( s )
            def mk_f(v):
                def do_it(ds):
                    return v
                return do_it
            return mk_f(final.value)
        elif final.type=='id':
            # ok, we know we recognize the token, let's eat it up [this was the 'id']
            final = parse_id( s )

            # now, before returning something, check the new current token
            # if it happens to be 'lparen', we're looking at a functioncall!
            if tok(s).type=='lparen':
                # if the id had a filter, then it cannot be a functioncall!
                #   aap.phase[p=ll] (...)   should not parse
                #   mod.fn (...)            might be ok: python "module.function" in stead of "variable.type"
                if 'filter' in final.value:
                    raise SyntaxError,"A subscripted variable cannot be used as function call?!"
                # eat the paren, then parse the argument list
                next( s )
                arglist = parse_arglist(s, [])

                # after parsing the arglist we MUST see 'rparen' orelse the user's a fool
                if tok(s).type!='rparen':
                    raise SyntaxError, "Expected ')' after functioncall, got {0}".format( tok(s).value )

                # eat the rparen
                next( s )

                # and return the useful bits
                def mk_f(fn, al):
                    def do_it(ds):
                        # lookup fn
                        #callable_obj = do_resolve(fn)
                        print "{0}({1})".format(fn, ",".join(map(lambda x: repr(x(ds)), al)))
                        #return apply_fn(callable_obj, ds, al)
                        return 42
                    return do_it
                return mk_f(final.value, arglist)
            else:
                # no functioncall, just variable addressment
                def mk_f(nm):
                    def do_it(ds):
                        nam  = nm['name']
                        typ  = nm['type']
                        if nam not in ds:
                            raise RuntimeError, "Variable '{0}' does not exist".format(nam)
                        if isDataset(ds[nam]):
                            return ds_flat_filter(ds[nam], typ, nm.get('filter', None))
                        else:
                            return ds[nam][typ] if typ is not None else ds[nam]
                    return do_it
                return mk_f(final.value)
        elif final.type=='lparen':
            # ah. parenthesis! eat up the '('
            next( s )

            s.depth = s.depth + 1
            expr = parse_expr(s)
            # now we should see ')'
            cur = tok( s )
            if cur.type!='rparen':
                raise SyntaxError, "Expected ')' but got {0}".format( cur.value )
            s.depth = s.depth - 1
            next( s )
            return expr
        else:
            # we don't actually recognize this token here ?
            return None

    #@argprint
    def parse_arglist(s, al):
        #   <arglist>   = <empty> | <list>
        #   <list>      = <expr> {',' <expr> }
        
        # Basically we're done if we see 'rparen'
        # don't eat the token because the upper level needs to see it to
        # make sure that parens are balanced
        cur = tok( s )
        if cur.type=='rparen':
            return al
        return parse_list(s, al)

    #@argprint
    def parse_list(s, al):
        # we MUST see an expression now
        arg = parse_expr(s)

        if arg is None:
            raise SyntaxError, "Empty argument is not allowed (current token={0})".format( tok(s).value )
        
        # append it to the argument list
        al.append( arg )

        # inspect current token; if it's a comma we recurse
        if tok(s).type=='comma':
            next(s)
            return parse_list(s, al)
        # if we don't see a comma, we return and let the upper levels decide wether the current token is a nice one.
        return al



    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

        def __str__(self):
            return "<{0}/{1}>".format(self.depth, self.token)

    tokenizer  = mk_tokenizer(tokens, **env)
    return parse_dataset_expr_impl(state_type(tokenizer(txt)))(datasets)




########################################################################################################
#
#  colorkey expression parser
#  allows for assigning specific color indices to data sets, based on attribute value(s)
#
#########################################################################################################

# The idea is that the user can type an expression:
#
# > ckey P[LL]=1 P[RR]=2
#
# Such that all data sets where the 'P' (olarization) attribute has the value 'LL' get color index '1'
# and those with 'RR' get color index '2'.
#
# More detailed selections are possible:
#
# > ckey SB[0], BL[/wb*/]=1
#
# without constraints it does 'iota' = automatic counting 
# > ckey P[RR],SB
#
# When a label is passed that doesn't match any of the criteria an exception is thrown
#
# Grammar:
#
# expr     = selector {' ' selector} EOF
# selector = attribs {'=' colorkey}
# attribs  = attrib {',' attrib}
# attrib   = attrname {'[' attrvallist ']'}
# attrname = 'P' | 'CH' | 'SB' | 'FQ' | 'BL' | 'TIME' | 'SRC' |
#            'p' | 'ch' | 'sb' | 'fq' | 'bl' | 'time' | 'src'
# attrvallist = attrval {',' attrvallist }
# attrval     = number | string | regex
# number      = [0-9]+
# string      = 'text'
# colorkey    = number
# regex       = '/' text '/'
# text        = all characters except the termination (http://stackoverflow.com/a/5455705/26083)
#
#  The regex from http://stackoverflow.com/a/5455705/26083:
#  (this one looks for single-quote quoted strings but can easily be modified to support
#   other delimiting characters)
#     re.compile( r"""(?<!\\)(?:\\\\)*'([^'\\]*(?:\\.[^'\\]*)*)'""", re.DOTALL )

def mk_escaped_rx(ch, suf=None):
    return re.compile( r"""(?<!\\)(?:\\\\)*{0}([^{0}\\]*(?:\\.[^{0}\\]*)*){0}{1}""".format(ch, "" if suf is None else suf), re.DOTALL )

def mk_regex(rx):
    flagmap = {'i': re.I }
    flag    = 0
    # strip flag characters
    while rx[-1]!=rx[0]:
        flag |= flagmap.get(rx[-1], 0)
        rx = rx[:-1]
    return re.compile(rx[1:-1], flag)

def parse_ckey_expr(expr):
    # our tokens
    tokens = [
        # the attribute names we support
        (re.compile(r"\b(p|ch|sb|fq|bl|time|src)\b", re.I), value_t('attrname')),
        # attribute values
        #    @regex and text: we get the terminating start, end characters as well so must strip them off
        (mk_escaped_rx('/', "i?"),                          xform_t('regex', mk_regex)),
        number_token(),
        # '=', '[', ']' and ','
        token_def(r"\[",                                    simple_t('lbracket')),
        token_def(r"\]",                                    simple_t('rbracket')),
        token_def(r'=',                                     simple_t('equal')),
        token_def(r',',                                     simple_t('comma')),
        token_def(r"[^][ '\t]+",                              value_t('text')),
        (mk_escaped_rx("'"),                                xform_t('text',  lambda v: v[1:-1])),
        token_def(r"\s+",                                   ignore_t())
        #token_def(r"[a-zA-Z0-9\+\-]"
    ]

    # shorthands that work on the parser state 's'
    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    # ---- implementation
    def parse_ckey_expr_impl(s):
        # the wrapper function that generates the color index for a given label
        def mk_ckey_fn(lst):
            def do_it(label, keycoldict, **opts):
                # run the label through all the filters and see if something sticks
                #print "parse_ckey_expr:ckey_fn label={0}".format( str(label) )
                cks = filter(lambda x: x is not None, [ck(label, keycoldict) for ck in lst])
                # if no color for the label ... that's a bad thing isn't it?!
                if not cks:
                    raise RuntimeError, "None of the colour filters matched label {0}".format( label )
                #print "parse_ckey_expr:ckey_fn => found a match: colour = {0} [{1}]".format( cks[0], cks )
                return cks[0]
            return do_it
        selectors = []
        # a valid ckey expr is a sequence of valid selector assignments
        while True:
            selectors.append( parse_selector(s) )
            # After having succesfully parsed a selector,
            # we should see EOF or another selector
            if tok(s).type is None:
                break
        return mk_ckey_fn(selectors)

    def parse_selector(s):
        def mk_cond(attrnm, attrlist, vallist):
            # list of values, turn into one string
            valstr = ",".join(map(str, vallist)) if vallist else ""
            def do_it(label):
                aval = getattr(label, attrnm)
                #print "parse_selector:do_it({0}) - attrnm[{1}] => {2}".format(str(label), attrnm, aval)
                # 'attrlist' is a list of functions that we'll pass the attribute value to see if it 
                # matches the condition(s). Return true if at least one matches
                if (aval is None) or (attrlist and [pred(aval) for pred in attrlist].count(True)==0):
                    return (None, None)
                if vallist:
                    return (attrnm, valstr)
                else:
                    return (attrnm, aval)
            return do_it

        # we have any number of "attrval {'[' ... ']'}" before the '='
        condlist = []
        while True:
            # a selector starts with an attrname
            attrname = tok(s)
            if attrname.type!='attrname':
                raise SyntaxError, "Unexpected token '{0}', expected attribute name".format(attrname)
            # safe to consume
            next(s)
            # if we see a '[' we must parse an attrivallist
            t = tok(s)
            alist = []
            vlist = []
            if t.type=='lbracket':
                (alist, vlist) = parse_attrvallist(s)
            # excellent! add another condition to the list of conditions
            condlist.append( mk_cond(attrname.value.upper(), alist, vlist) )
            # Valid: either ',' (another "attrval"), "=", 'EOF' or another
            # selector "p sb" (two selectors) is different from "p,sb" (one selector)
            # note: could also be end-of-input
            t = tok(s)
            if t.type not in ['equal', 'comma', 'attrname', None]:
                raise SyntaxError, "Unexpected token '{0}', expected '=', ',' or an attribute name".format( t )
            # Ok, let's see what to do now
            if t.type=='comma':
                # consume the comma and continue: we should see another attrname!
                next(s)
                continue
            # all other valid tokens cause a break
            break

        # Ok, we may see an '=' sign, another attributename or EOF
        equal    = tok(s)

        def mk_colidxfn():
            def do_it(keycoldict, key):
                # the default colour key function: if key not in keycoldict yet, 
                # insert new one with new colour
                ck = keycoldict.get(key, None)
                if ck is None:
                    # find values in the keycoldict and choose one that isn't there already
                    colours = sorted([v for (k,v) in keycoldict.iteritems()])
                    # find first unused colour index, skip colour 0 and 1 (black & white)
                    ck = 2
                    while colours and ck<=colours[-1]:
                        if ck not in colours:
                            break
                        ck = ck + 1
                    keycoldict[key] = ck
                #print "parse_selector:colidxfn[default]: key={0} => colour={1}".format(key, ck)
                return ck
            return do_it
        colidxfn = mk_colidxfn()
        #colidxfn = lambda keycoldict, key: keycoldict.setdefault(key, len(keycoldict))
       
        if equal.type=='equal':
            # must see an integer
            next(s)
            cval = tok(s)
            if cval.type!='number':
                raise SyntaxError, "Unexpected token '{0}', expected a colour index (integer)"
            def mk_colidxfn(cv):
                def do_it(keycoldict, key):
                    #print "parse_selector:colidxfn[constval]=",cv
                    if keycoldict.setdefault(key, cv)!=cv:
                        raise RuntimeError, "Key {0} was already present with other colour index {1} [request to set to {2}]".format(key, keycoldict.get(key), cv)
                    return cv
                return do_it
                #return lambda keycoldict, key: cv
            colidxfn = mk_colidxfn(cval.value)
            # ok, eat the integer
            next(s)
        # from the conditionlist and colorvalue, we may construct the
        # colorselection function
        def mk_colselect(conds, colouridxfn):
            # return a function that, given a label and a colorkey dict,
            # returns either None or the color index for the label
            def do_it(l, keycoldict):
                #print "parse_selector:colorselector[{0} conds] for label {1}".format(len(conds), str(l))
                # run all our conditions on the label; they ALL must match
                ms   = map(lambda cond: cond(l), conds)
                # if any of them are (None, None) - this label doesn't meet
                # all our criteria
                #print "parse_selector:colorselector[{0} conds] ms={1}".format(len(conds), ms)
                if (None, None) in ms:
                    return None
                # create new label with all attributes
                # taken from the conditions
                nl = reduce(lambda acc, (nm, v): setattr(acc, nm, v) or acc, ms, plots.label({}, []))
                # str representation is key
                return colouridxfn(keycoldict, str(nl))
            return do_it
        return mk_colselect(condlist, colidxfn)

    # parse an attribute-value list!
    # return a tuple of list of match functions and a list of values (the strings)
    # for later display purposes
    def parse_attrvallist(s):
        # we should see '['
        lbrack = tok(s)
        if lbrack.type!='lbracket':
            raise SyntaxError, "Unexpected token '{0}', expect '['".format(lbrack)
        # eat the '['
        next(s)
        valfnlist  = []
        valvallist = []
        # now we should be eating comma-separated entries until we see ']'
        musthaveitem = False
        while True:
            cur = tok(s)
            # only accept the closing ']' if we do not expect another item
            if cur.type=='rbracket':
                if musthaveitem:
                    raise SyntaxError, "Missing item after ',' in list at {0}".format(cur.pos-1)
                # consume the ']'
                next(s)
                break
            # not close of list so must see at least a supported item
            if cur.type == 'number':
                # equality-compare
                def mk_comp(v):
                    def do_it(aval):
                        #print "parse_attrvallist:attribute-value comparator: {0}=={1}?".format(aval, v)
                        return aval==v
                    return do_it
                valfnlist.append( mk_comp(cur.value) )
            elif cur.type == 'text':
                # case insensitive equality-compare
                def mk_comp_i(v):
                    def do_it(aval):
                        #print "parse_attrvallist:attribute-value case insensitive text compare: {0}=={1}?".format(aval, v)
                        return aval.lower()==v
                    return do_it
                valfnlist.append( mk_comp_i(cur.value.lower()) )
            elif cur.type == 'regex':
                # run the regex against the attribute value
                def mk_reg_exec(rg):
                    def do_it(aval):
                        #print "parse_attrvallist:attribute-value regex matcher: {0} matches {1}?".format(aval, rg.pattern)
                        return rg.search(aval) is not None
                    return do_it
                valfnlist.append( mk_reg_exec(cur.value) )
            else:
                # unsupported type!
                raise SyntaxError, "Unsupported list item '{0}', expected number, text or regex".format(cur)
            # append the actual value to the list
            valvallist.append( cur.value )
            # eat the item
            next(s)
            cur = tok(s)
            if cur.type=='comma':
                # eat the comma and indicate that we NEED an extra item
                musthaveitem = True
                next(s)
            else:
                musthaveitem = False
        return (valfnlist, valvallist)

    ## debuggert - just show all the tokens
    def parse_ckey_expr_impl_tokens(s):
        while True:
            t = tok(s)
            print t
            if t.type is None:
                break
            next(s)
        return 42

    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

        def __str__(self):
            return "<{0}/{1}>".format(self.depth, self.token)

    tokenizer  = mk_tokenizer(tokens, **{})
    return parse_ckey_expr_impl(state_type(tokenizer(expr)))


#########################################################################################################
#
#   filter parser
#
#   Let the user filter datasets to plot based on dataset attribute value(s)
#   *after* they've been read from disk but before they're plotted.
#
#   This can be useful if data reading takes a lot of time and then be able
#   to (re)plot subset(s) of that data without having to re-read the raw data
#   from disk
#
#########################################################################################################

######  Our grammar

# filter     = condition eof
# condition  = condexpr {relop condition} | 'not' condition | '(' condition ')'
# #condexpr   = attribute '~' (regex|text) | attribute compare expr | attribute 'in' list
# condexpr   = attribute '~' (regex|text) | attribute compare number | attribute 'in' list
# attribute  = 'P' | 'CH' | 'SB' | 'FQ' | 'BL' | 'TIME' | 'SRC' |
#              'p' | 'ch' | 'sb' | 'fq' | 'bl' | 'time' | 'src'
#
# compare    = '=' | '>' | '>=' | '<' | '<=' ;
# relop      = 'and' | 'or' ;
# list       = '[' [value {',' value}] ']'
# value      = number | text
# regex      = '/' {anychar - '/'} '/' ['i']  ('i' is the case-insensitive match flag)
# number     = digit { number }
# digit      = [0-9]
# text       = char { alpha }
# char       = [a-zA-Z_]
# alpha      = char | digit { alpha }
#
mk_attribute_getter = lambda a: lambda obj: getattr(obj, a.upper())

def parse_filter_expr(qry, **kwargs):
    # Helper functions

    def mk_intrange(txt):
        return hvutil.expand_string_range(txt, rchar='-')

    # take a string and make a "^...$" regex out of it,
    # doing escaping of regex special chars and 
    # transforming "*" into ".*" and "?" into "."
    # (basically shell regex => normal regex)
    def pattern2regex(s):
        s = reduce(lambda acc, x: re.sub(x, x, acc), ["\+", "\-", "\."], s)
        s = reduce(lambda acc, (t, r): re.sub(t, r, acc), [("\*+", ".*"), ("\?", ".")], s)
        return re.compile("^"+s+"$")

    def regex2regex(s):
        flagmap = {"i": re.I, None:0}
        mo = re.match(r"(.)(?P<pattern>.+)\1(?P<flag>.)?", s)
        if not mo:
            raise RuntimeError,"'{0}' does not match the regex pattern /.../i?".format(s)
        return re.compile(mo.group('pattern'), flagmap[mo.group('flag')])

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        # the attribute names we support
        #(re.compile(r"\b(p|ch|sb|fq|bl|time|src)\b", re.I), value_t('attribute')),
        (re.compile(r"\b(p|ch|sb|fq|bl|time|src)\b", re.I), xform_t('attribute', mk_attribute_getter)),
        # operators
        token_def(r"\bnot\b",                               operator_t('not')),
        token_def(r"\bin\b",                                operator_t('in')),
        token_def(r"\b(and|or)\b",                          operator_t('relop')),
        token_def(r"(<=|>=|=|<|>)",                         operator_t('compare')),
        token_def(r"(~|\blike\b)",                          xform_t('regexmatch', lambda o, **k: lambda x, y: re.match(y, x) is not None)),
        # parens + list stuff
        token_def(r"\(",                                    simple_t('lparen')),
        token_def(r"\)",                                    simple_t('rparen')),
        token_def(r"\[",                                    simple_t('lbracket')),
        token_def(r"\]",                                    simple_t('rbracket')),
        token_def(r",",                                     simple_t('comma')),
        # values + regex
        int_token(),
        token_def(r"/[^/]+/i?\b",                           xform_t('regex', regex2regex)),
        token_def(r"[:@\#%!\.a-zA-Z0-9_?|]+",               value_t('text')),
        # and whitespace
        token_def(r"\s+",                                   ignore_t())
    ]

    tokenizer  = mk_tokenizer(tokens, **kwargs)

    # The output of the parsing is a filter function that returns
    # True or False given a dataset object

    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    ######  Our grammar

    # filter      = condition eof
    def parse_filter(s):
        if tok(s).type is None:
            raise SyntaxError, "empty filter"

        filter_f  = parse_condition(s) 
        # "LIMIT"
        #limit     = tok(s)
        #if limit.type=='limit':
        #    # we MUST be followed by an int
        #    next(s)
        #    ival = tok(s)
        #    if ival.type!='int':
        ##        raise SyntaxError, "Only an integer is allowed after limit, not %s" % ival
        #    # consume the integer
        #    next(s)
        #    count = itertools.count()
        #    limit.value = lambda x: itertools.takewhile(lambda obj: count.next()<ival.value, x)
        #else:
        #    limit.value = lambda x: x

        # the only token left should be 'eof' AND, after consuming it,
        # the stream should be empty. Anything else is a syntax error
        try:
            if tok(s).type is None:
                next(s)
        except StopIteration:
            return filter_f
        raise SyntaxError, "Tokens left after parsing %s" % tok(s)

    def parse_paren(s):
        lparen = tok(s)
        if lparen.type!='lparen':
            raise RuntimeError, "Entered parse_paren w/o left paren but %s" % lparen
        depth   = s.depth
        s.depth = s.depth + 1
        # recurse into parsing the expression - and do NOT forget to consume the lparen!
        expr    = parse_expr(next(s))
        # now we should be back at the same depth AND we should be seeing rparen
        rparen  = tok(s)
        if rparen.type=='rparen':
            s.depth = s.depth - 1
            next(s)
        return expr

    # condition  = condexpr {relop condition} | 'not' condition | '(' condition ')'
    # relop      = 'and' | 'or' ;
    def parse_condition(s):
        token = tok(s)

        # Recurse if we need to
        if token.type in ['lparen', 'rparen']:
            lterm = parse_paren_condition(s)
        # 'not' expr
        elif token.type=='not':
            # parse the next expr and negate it
            # we MUST have a next one
            condition = parse_condition(next(s))
            if condition is None:
                raise SyntaxError, "Missing expression after 'not' %s" % condition
            lterm = lambda scan: operator.not_( condition(scan) )
        else:
            # it must be a condexpr
            lterm = parse_cond_expr(s)

        # If we now see a relop, we have to parse another condition
        relop = tok(s)
        if relop.type!='relop':
            return lterm

        # consume the relop & parse the condition
        rterm = parse_condition(next(s))

        if lterm is None:
            raise SyntaxError, "Missing left-hand-condition to relational operator (%s)", relop
        if rterm is None:
            raise SyntaxError, "Missing right-hand-condition to relational operator (%s)", relop

        # and return the combined operation
        return lambda scan: relop.value(lterm(scan), rterm(scan))


    # condexpr   = attribute '~' (regex|text) | attribute compare number | attribute 'in' list
    # compare    = '=' | '>' | '>=' | '<' | '<=' ;
    def parse_cond_expr(s):
        attribute = tok(s)
        # No matter what, we have a left and a right hand side
        # separated by an operator
        if not (attribute.type == 'attribute'):
            raise SyntaxError, "Unexpected token {0}, expected attribute name".format( attribute.type )
        # consume the attribute value
        next(s)
        # Now we must see a comparator
        compare = tok(s)
        if not compare.type in ['compare', 'regexmatch', 'in']:
            raise SyntaxError, "Expected a comparison operator, regex match or 'in' keyword, got {0}".format( compare )
        # consume the comparison
        next(s)
        # do some processing based on the type of operator
        if compare.type=='in':
            rterm  = parse_list(s)
        elif compare.type=='compare':
            #rterm = parse_expr(s)
            # we only support numbers here
            rterm = tok(s)
            if not (rterm.type=='int'):
                raise SyntaxError, "Unexpected token {0}, expected a number here".format( rterm )
            # and consume the number
            rterm = rterm.value
            next(s)
        else:
            # must've been regexmatch
            rterm = parse_rx(s)
        # it better exist
        if rterm is None:
            raise SyntaxError, "Failed to parse right-hand-term of cond_expr (%s)" % tok(s)
        print "HAVE rterm=",rterm
        return lambda ds: compare.value(attribute.value(ds), rterm)

    def parse_paren_condition(s):
        lparen = tok(s)
        if lparen.type!='lparen':
            raise RuntimeError, "Entered parse_paren_condition w/o left paren but %s" % lparen
        depth   = s.depth
        s.depth = s.depth + 1
        # recurse into parsing the expression - and do NOT forget to consume the lparen!
        expr    = parse_condition(next(s))
        # now we should be back at the same depth AND we should be seeing rparen
        rparen  = tok(s)
        if rparen.type=='rparen':
            s.depth = s.depth - 1
            next(s)
        return expr

    def parse_rx(s):
        # we accept string, literal and regex and return an rx object
        rx = tok(s)
        if not rx.type in ['regex', 'text', 'literal']:
            raise SyntaxError, "Failed to parse string matching regex (not regex, text or literal but %s)" % rx
        # consume the token
        next(s)
        if rx.type=='literal':
            # extract the pattern from the literal (ie strip the leading/trailing "'" characters)
            rx.value = rx.value[1:-1]
        if rx.type in ['text', 'literal']:
           rx.value = pattern2regex(rx.value) 
        return rx.value

    def parse_list(s):
        bracket = tok(s)
        if bracket.type != 'lbracket':
            raise SyntaxError, "Expected list-open bracket ('[') but found %s" % bracket
        rv = []
        # keep eating text + ',' until we read 'rbracket'
        next(s)
        while tok(s).type!='rbracket':
            # if we end up here we KNOW we have a non-empty list because
            # the next token after '[' was NOT ']'
            # Thus if we need a comma, we could also be seeing ']'
            needcomma        = len(rv)>0
            #print " ... needcomma=",needcomma," current token=",tok(s)
            if needcomma:
                if tok(s).type=='rbracket':
                    continue
                if tok(s).type!='comma':
                    raise SyntaxError, "Badly formed list at {0}".format(tok(s))
                # and eat the comma
                next(s)
            # now we need a value. 'identifier' is also an acceptable blob of text
            rv.extend( parse_list_item(s) )
            #print "parse_list: ",rv
        # and consume the rbracket (if not rbracket a syntax error is raised above)
        next(s)
        return rv

    # always returns a list-of-items; suppose the list item was an irange
    def parse_list_item(s):
        t = tok(s)
        # current token must be 'text' or 'irange'
        if not t.type in ['text', 'irange', 'int', 'float', 'literal']:
            raise SyntaxError, "Failure to parse list-item {0}".format(t)
        next(s)
        # for a literal, strip the leading and closing single quote
        if t.type == 'literal':
            t.value = t.value[1:-1]
        return t.value if t.type == 'irange' else [t.value]

    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.next()

        def next(self):
            self.token       = self.tokenstream.next()
            return self

    tokenizer  = mk_tokenizer(tokens, **kwargs)
    return parse_filter(state_type(tokenizer(qry)))


#########################################################################################################
#
#   animation parser
#
#   Let the user animate datasets based on dataset attribute value(s)
#   *after* they've been read from disk 
#
#########################################################################################################

######  Our grammar

#    animate <selection> by <attributes> <eof>
#    (The 'animate' keyword is taken to be matched out in the command parser)
#
#    # empty selection means "current"
#    <selection>  = "" | <dataset>
#    <attributes> = <attribute> { ',' <attribute> }
#
#    <dataset>    = {<identifier> ':'} <expression>
#    <identifier> = [a-zA-Z][0-9a-zA-Z]*   # alphanumeric variable name
#
#    <attribute>  = <attrname> { <sortorder> }
#    <attrname>   = 'time' | 'src' | 'bl' | 'p' | 'sb' | 'ch' | 'type'
#    <sortorder>  = 'asc' | 'desc'
#
#    <expression> = <expr> { 'and' <expression> | 'or' <expression> }
#    <expr>       = <condition> | 'not' <expression> | '(' <expression> ')'
#    <condition>  = <attrname> <relop> <value> |
#                   <attrname> 'in' <list> |
#                   <attrname> 'like' <regex> |
#                   <attrname> 'like' <text>
#    <relop>      = '<' | '<=' | '=' | '>' | '>=' 
#    <list>       = '[' <listitems> ']' | <intrange>
#    <listitems>  = <listitem> {',' <listitems> }
#    <listitem>   = <value> | <intrange>
#    <intrange>   = <int>':'<int>
#    <value>      = <number> | <text> 
#    <text>       = ''' <characters> '''
#    <regex>      = '/' <text> '/'




# condition  = condexpr {relop condition} | 'not' condition | '(' condition ')'
# #condexpr   = attribute '~' (regex|text) | attribute compare expr | attribute 'in' list
# condexpr   = attribute '~' (regex|text) | attribute compare number | attribute 'in' list
# attribute  = 'P' | 'CH' | 'SB' | 'FQ' | 'BL' | 'TIME' | 'SRC' |
#              'p' | 'ch' | 'sb' | 'fq' | 'bl' | 'time' | 'src'
#
# compare    = '=' | '>' | '>=' | '<' | '<=' ;
# relop      = 'and' | 'or' ;
# list       = '[' [value {',' value}] ']'
# value      = number | text
# regex      = '/' {anychar - '/'} '/' ['i']  ('i' is the case-insensitive match flag)
# number     = digit { number }
# digit      = [0-9]
# text       = char { alpha }
# char       = [a-zA-Z_]
# alpha      = char | digit { alpha }
#


def parse_animate_expr(qry, **kwargs):
    # Helper functions

    def mk_intrange(txt):
        return hvutil.expand_string_range(txt, rchar='-')

    # take a string and make a "^...$" regex out of it,
    # doing escaping of regex special chars and 
    # transforming "*" into ".*" and "?" into "."
    # (basically shell regex => normal regex)
    def pattern2regex(s):
        s = reduce(lambda acc, x: re.sub(x, x, acc), ["\+", "\-", "\."], s)
        s = reduce(lambda acc, (t, r): re.sub(t, r, acc), [("\*+", ".*"), ("\?", ".")], s)
        return re.compile("^"+s+"$")

    def regex2regex(s):
        flagmap = {"i": re.I, None:0}
        mo = re.match(r"(.)(?P<pattern>.+)\1(?P<flag>.)?", s)
        if not mo:
            raise RuntimeError,"'{0}' does not match the regex pattern /.../i?".format(s)
        return re.compile(mo.group('pattern'), flagmap[mo.group('flag')])

    # basic lexical elements
    # These are the tokens for the tokenizer
    tokens = [
        token_def(r"\b(animate|by|asc|desc)\b",    keyword_t()),
        # the attribute names we support
        #(re.compile(r"\b(p|ch|sb|fq|bl|time|src|type)\b", re.I), xform_t('attribute', mk_attribute_getter)),
        (re.compile(r"\b(p|ch|sb|fq|bl|time|src|type)\b", re.I), xform_t('attribute', str.upper)),
        #(re.compile(r"\b(p|ch|sb|fq|bl|time|src|type)\b", re.I), value_t('attribute')),
        # Date + time formats
        datetime_token(YMD,     TIME),
        datetime_token(YMD,     HMS),
        datetime_token(DMY,     TIME),
        datetime_token(DMY,     HMS),
        datetime_token(DMY_EUR, TIME),
        datetime_token(DMY_EUR, HMS),
        # Relative day offset - note: assume that the global variable
        # 'start' is set correctly ...
        token_def(RELDAY+TIME,  datetime_t(mk_seconds)),
        token_def(RELDAY+HMS,   datetime_t(mk_seconds)),
        # Time durations
        token_def(RELDAY+DUR3,  datetime_t(mk_seconds)),
        token_def(RELDAY+DUR2,  datetime_t(mk_seconds)),
        token_def(RELDAY+DUR1,  datetime_t(mk_seconds)),
        token_def(DUR4,         xformmg_t('duration', mk_seconds)),
        token_def(DUR3,         xformmg_t('duration', mk_seconds)),
        token_def(DUR2,         xformmg_t('duration', mk_seconds)),
        token_def(DUR1,         xformmg_t('duration', mk_seconds)),
        # operators
        token_def(r"\bnot\b",                               operator_t('not')),
        token_def(r"\bin\b",                                operator_t('in')),
        token_def(r"\b(and|or)\b",                          operator_t('relop')),
        token_def(r"(<=|>=|=|<|>)",                         operator_t('compare')),
        token_def(r"(~|\blike\b)",                          xform_t('regexmatch', lambda o, **k: lambda x, y: re.match(y, x) is not None)),
        # parens + list stuff
        token_def(r"\(",                                    simple_t('lparen')),
        token_def(r"\)",                                    simple_t('rparen')),
        token_def(r"\[",                                    simple_t('lbracket')),
        token_def(r"\]",                                    simple_t('rbracket')),
        token_def(r",",                                     simple_t('comma')),
        token_def(r":",                                     simple_t('colon')),
        token_def(r"-|\+|\*|/",                             operator_t('operator')),
        # values + regex
        int_token(),
        token_def(r"/[^/]+/i?\b",                           xform_t('regex', regex2regex)),
        token_def(r"\$(?P<sym>[a-zA-Z][a-zA-Z_]*)",         resolve_t('external', 'sym')),
        token_def(r"'([^']*)'",                             extract_t('literal', 1)),
        token_def(r"[a-zA-Z][a-zA-Z0-9_]*",                 value_t('identifier')),
        token_def(r"\S+",                                   value_t('text')),
        #token_def(r"[:@\#%!\.\*\+\-a-zA-Z0-9_?|]+",         value_t('text')),
        # and whitespace
        token_def(r"\s+",                                   ignore_t())
    ]

    tokenizer  = mk_tokenizer(tokens, **kwargs)

    # The output of the parsing is a filter function that returns
    # True or False given a dataset object

    next    = lambda s: s.next()
    tok     = lambda s: s.token
    tok_tp  = lambda s: s.token.type
    tok_val = lambda s: s.token.value

    ######  Our grammar

    # animate      = 'animate' [<selection>] 'by' <attributes> <eof>
    def parse_animate(s):
        if tok(s).type != 'animate':
            raise SyntaxError,"The animate expression does not start with the keyword 'animate' but with {0}".format(tok(s))
        # skip that one
        next(s)
        # now we may see a selection
        selection_f = parse_selection(s)
        # check mismatched parentheses in the expression(s)
        if s.depth!=0:
            raise SyntaxError, "Mismatched parentheses"
        # now we MUST see the 'by' keyword
        if tok(s).type!='by':
            raise SyntaxError,"Unexpected token {0}, expected the 'by' keyword".format(tok(s))
        # and skip that one
        next(s)
        # now we must parse the <attributes>
        groupby_f = parse_attributes(s)
        # the only token left should be 'eof' AND, after consuming it,
        # the stream should be empty. Anything else is a syntax error
        try:
            if tok(s).type is None:
                next(s)
        except StopIteration:
            return (selection_f, groupby_f)
        raise SyntaxError, "(at least one)token left after parsing: {0}".format(tok(s))
    
    #    <selection>  = "" | <dataset>
    def parse_selection(s):
        # try to parse the dataset identifier, if it is None then there was none
        # and we default to "_"
        dataset_id = parse_dataset_id(s)
        #if dataset_id is None:
        #    dataset_id = '_'
        # now we may see an expression
        filter_f = parse_expression(s)
        # build a filtering function on the indicated dataset
        return (dataset_id, filter_f)

    #    <dataset>    = {<identifier> ':'} <expression>
    #    Note: let's allow 'attribute' here as well - the ':' following
    #          the dataset_id would be the disambiguator between
    #          data set identifier and attribute name?
    def parse_dataset_id(s):
        # if we don't see an identifier that means there's no identifier at all
        dataset_id = tok(s)
        if dataset_id.type not in[ 'identifier', 'attribute']:
            return None
        # if we see attribute we need to peek ahead to see if the disambiguating ':' is there
        if dataset_id.type == 'attribute' and s.peek().type!='colon':
            # rite, not a data set id
            return None
        dataset_id = dataset_id.value
        # eat it up and then we MUST see ':'
        next(s)
        if tok(s).type != 'colon':
            raise SyntaxError, "Expected ':' following data set identifier"
        # consume and return the actual identifier
        next(s)
        return dataset_id

    #    <expression> = <expr> { 'and' <expression> | 'or' <expression> }
    #    <expr>       = <condition> | 'not' <expression> | '(' <expression> ')'
    #    <condition>  = <attrname> <relop> <number> |
    def parse_expression(s):
        left = parse_expr(s)
        if left is None:
            return None
        # OK. We have a left hand side
        # if we're looking at a relop we may have to parse a right hand side
        if tok(s).type != 'relop':
            return left
        # ok looking at relop, save it for later use and move on
        relop = tok(s).value
        # now we MUST see a right hand side
        right = parse_expression( next(s) )
        if right is None:
            raise SyntaxError, "Missing right hand side to logical operator and or or"
        def mk_f(l, op, r):
            def do_it(ds):
                return op(l(ds), r(ds))
            return do_it
        return mk_f(left, relop, right)

    def parse_expr(s):
        # depending on what token we're looking at choose the appropriate action
        tp = tok(s).type
        if tp == 'attribute':
            # depending on the type of the attribute ...
            return parse_cond_expr(s)
            #return parse_cond_expr(s)
        if tp == 'not':
            # consume the 'not' and return whatever is following
            f = parse_expression( next(s) )
            return lambda ds: not f(ds)
        if tp == 'lparen':
            # remove the '('
            s.depth = s.depth + 1
            expr = parse_expression( next(s) )
            # now we MUST see ')' [and if we do skip it]
            if tok(s).type != 'rparen':
                raise SyntaxError, "Mismatched parenthesis"
            s.depth = s.depth - 1
            next(s)
            if expr is None:
                raise SyntaxError, "An empty expression is not an expression"
            return expr
        return None

    # term     = duration | number | external
    def parse_time_term(s):
        term = tok(s)
        # The easy bits first
        if term.type in ['number', 'external', 'duration', 'datetime']:
            # all's well - eat this term
            next(s)
            return term.value
        return None
    # support datetime or an expression involving datetime?
    # expr     = term | expr '+' expr | expr '-' expr | expr '*' expr | expr '/' expr | '(' expr ')' | '-' expr
    def parse_time_cond(s, unary=False):
        t = tok(s)
        depth = s.depth

        print "parse_time_cond/tok=", t, " depth=", depth
        if t.type == 'lparen':
            s.depth = s.depth + 1
            lterm   = parse_time_cond( next(s) )
            # now we MUST see ')' [and if we do skip it]
            if tok(s).type != 'rparen':
                raise SyntaxError, "Mismatched parenthesis"
            s.depth = s.depth - 1
            next(s)
            if lterm is None:
                raise SyntaxError, "An empty expression is not an expression"
            return lterm
        elif t.type=='operator' and t.value is mk_operator('-'):
            # unary '-'
            tmpexpr = parse_time_cond(next(s), unary=True)
            lterm   = operator.neg( tmpexpr )
        else:
            print "parsing time term?"
            lterm = parse_time_term(s)
            print "  yields: ",lterm

        # If we see an operator, we must parse the right-hand-side
        # (our argument is the left-hand-side
        # Well ... not if we're doing unary parsing!
        # if we saw unary '-' then we should parse parens and terms up until
        # the next operator
        oper = tok(s)
        if oper.type=='operator':
            if unary:
                return lterm
            if lterm is None:
                raise SyntaxError, "No left-hand-side to operator {0}".format(oper)
            rterm = parse_time_cond(next(s))
            if rterm is None:
                raise SyntaxError, "No right-hand-side to operator {0}".format(oper)
            return oper.value(lterm, rterm)
        elif oper.type in ['int', 'float', 'duration', 'datetime']:
            # negative numbers as right hand side are not negative numbers
            # but are operator '-'!
            # so, subtracting a number means adding the negative value (which we already
            # have god)
            # Consume the number and return the operator add
            next(s)
            return operator.add(lterm, oper.value)
        # neither parens, terms, operators?
        return lterm

    # condexpr   = attribute '~' (regex|text) | attribute compare number | attribute 'in' list
    # compare    = '=' | '>' | '>=' | '<' | '<=' ;
    def parse_cond_expr(s):
        attribute = tok(s)
        # No matter what, we have a left and a right hand side
        # separated by an operator
        if not (attribute.type == 'attribute'):
            raise SyntaxError, "Unexpected token {0}, expected attribute name".format( attribute.type )
        # consume the attribute value
        next(s)
        # Now we must see a comparator
        # for the time attribute 'regexmatch' and 'in' don't make sense
        compare = tok(s)
        if not compare.type in (['compare'] if attribute.value == 'TIME' else ['compare', 'regexmatch', 'in']):
            raise SyntaxError, "Invalid comparison operator {0} for attribute {1}".format( compare, attribute.value )
        # consume the comparison
        next(s)
        # do some processing based on the type of operator
        if compare.type=='in':
            rterm  = parse_list(s)
        elif compare.type=='compare':
            # we must compare to a value
            # take care of when attribute is 'time'
            if attribute.value == 'TIME':
                rterm = parse_time_cond(s)
            else:
                rterm = parse_value(s)
        else:
            # must've been regexmatch
            rterm = parse_rx(s)
        # it better exist
        if rterm is None:
            raise SyntaxError, "Failed to parse right-hand-term of cond_expr (%s)" % tok(s)
        return lambda ds: compare.value(mk_attribute_getter(attribute.value)(ds), rterm)

    #    <value>      = <number> | <text> 
    def parse_value(s):
        value = tok(s)
        if value.type not in ['int', 'text', 'identifier']:
            raise SyntaxError, "Unsupported value type - only int or text allowed here, not {0}".format( value )
        # consume the value and return it
        next(s)
        return value.value

    def parse_rx(s):
        # we accept string, literal, identifier and regex and return an rx object
        rx = tok(s)
        if not rx.type in ['regex', 'text', 'literal']:
            raise SyntaxError, "Failed to parse string matching regex (not regex, text or literal but %s)" % rx
        # consume the token
        next(s)
        if rx.type in ['text', 'literal']:
           rx.value = pattern2regex(rx.value) 
        return rx.value

    #    <list>       = '[' <values> ']' | <intrange>
    def parse_list(s):
        # could be actual list or int range
        if tok(s).type == 'lbracket':
            return parse_list_list(s)
        elif tok(s).type == 'int':
            return parse_int_range(s)
        # unexpected token
        raise SyntaxError, "Unexpected token {0} - not a list or int range".format(tok(s))

    def parse_int_range(s):
        # we *must* be looking at 'int'
        start = tok(s)
        if start.type != 'int':
            raise SyntaxError, "Expected an integer here, not a {0}".format(start)
        next(s)
        # now we must see colon
        if tok(s).type != 'colon':
            raise SyntaxError, "Expected ':' to form integer range"
        # eat up
        next(s)
        end = tok(s)
        if start.type != 'int':
            raise SyntaxError, "Expected an integer here, not a {0}".format(end)
        # don't forget to consume the number
        next(s)
        return range(start.value, end.value+1)

    def parse_list_list(s):
        bracket = tok(s)
        if bracket.type != 'lbracket':
            raise SyntaxError, "Expected list-open bracket ('[') but found %s" % bracket
        rv = []
        # keep eating text + ',' until we read 'rbracket'
        next(s)
        while tok(s).type!='rbracket':
            # if we end up here we KNOW we have a non-empty list because
            # the next token after '[' was NOT ']'
            # Thus if we need a comma, we could also be seeing ']'
            needcomma        = len(rv)>0
            #print " ... needcomma=",needcomma," current token=",tok(s)
            if needcomma:
                if tok(s).type=='rbracket':
                    continue
                if tok(s).type!='comma':
                    raise SyntaxError, "Badly formed list at {0}".format(tok(s))
                # and eat the comma
                next(s)
            # now we need a value. 'identifier' is also an acceptable blob of text
            rv.extend( parse_list_item(s) )
            #print "parse_list: ",rv
        # and consume the rbracket (if not rbracket a syntax error is raised above)
        next(s)
        return rv

    # always returns a list-of-items; suppose the list item was an irange
    def parse_list_item(s):
        t = tok(s)
        # current token must be 'text' or 'irange'
        if not t.type in ['text', 'irange', 'int', 'float', 'literal']:
            raise SyntaxError, "Failure to parse list-item {0}".format(t)
        next(s)
        return t.value if t.type == 'irange' else [t.value]

    
    #    <attributes> = <attribute> { ',' <attribute> }
    #    <attribute>  = <attrname> { <sortorder> }
    #    <attrname>   = 'time' | 'src' | 'bl' | 'p' | 'sb' | 'ch' | 'type'
    #    <sortorder>  = 'asc' | 'desc'
    def parse_attributes(s):
        groupby = set()
        sortfns = []
        while True:
            item = tok(s)
            if item.type!='attribute':
                raise SyntaxError, "Unexpected token {0}, expected an attribute".format(item)
            # check that the same attribute does not get mentioned twice
            if item.value in groupby:
                raise RuntimeError, "The attribute type {0} is mentioned more than once".format(item.value)
            groupby.add( item.value )
            # Peek at the next token. If it's asc/desc take that into account
            next(s)
            order = tok(s)
            if order.type in ['asc', 'desc']:
                order = order.type
                # consume it
                next(s)
            else:
                # default to asc
                order = 'asc'
            # create a sorting function
            def mk_sf(attr, order):
                def do_it(x):
                    return sorted(x, key=operator.attrgetter(attr), reverse=(order=='desc'))
                return do_it
            sortfns.append( mk_sf(item.value, order) )

            #if we don't see a comma next, we break
            if tok(s).type!='comma':
                break
            # consume the comma
            next(s)
        # primary sort key is now first in list but for the sorting to work in steps
        # (see https://wiki.python.org/moin/HowTo/Sorting ) we must apply the sorting
        # functions in reverse order
        return (operator.attrgetter(*groupby), lambda x: reduce(lambda acc, sortfn: sortfn(acc), reversed(sortfns), x))


    class state_type:
        def __init__(self, tokstream):
            self.tokenstream = tokstream
            self.depth       = 0
            self.lookAhead   = []
            self.next()

        def peek(self):
            self.lookAhead.append( self.tokenstream.next() )
            return self.lookAhead[-1]

        def next(self):
            if self.lookAhead:
                self.token     = self.lookAhead.pop(0) #self.lookAhead[0]
                #self.lookAhead = None
            else:
                self.token     = self.tokenstream.next()
            return self

    tokenizer  = mk_tokenizer(tokens, **kwargs)
    return parse_animate(state_type(tokenizer(qry)))







# expr     = term '+' term | term '-' term | term '*' term | term '/' term | '(' expr ')'
# term     = duration | number | property | external
# duration = \d+ 'd'[\d+ 'h'][\d+ 'm'] [\d+ ['.' \d* ] 's'] |
#                     \d+ 'h'[\d+ 'm'] [\d+ ['.' \d* ] 's'] |
#                              \d+ 'm' [\d+ ['.' \d* ] 's'] |
#                                        \d+ ['.' \d* ] 's'
# number   = int|float
# property = alpha char {alpha char | digit | '_'}    # will get property from scan object
# external = '$' property                             # will look up value of property in global namespace



# regex      = '/' {anychar - '/'} '/' ['i']  ('i' is the case-insensitive match flag)
# identifier = alpha {character} 
# anychar    = character | symbol
# character  = alpha | digit 
# alpha      = [a-zA-Z_] ;
# digit      = [0-9] ;

# selector = attribs {'=' colorkey}
# attribs  = attrib {',' attrib}
# attrib   = attrname {'[' attrvallist ']'}
# attrname = 'P' | 'CH' | 'SB' | 'FQ' | 'BL' | 'TIME' | 'SRC' |
#            'p' | 'ch' | 'sb' | 'fq' | 'bl' | 'time' | 'src'
# attrvallist = attrval {',' attrvallist }
# attrval     = number | string | regex
# number      = [0-9]+
# string      = 'text'
# colorkey    = number
# regex       = '/' text '/'
# text        = all characters except the termination (http://stackoverflow.com/a/5455705/26083)
