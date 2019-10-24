from __future__  import print_function
from functools   import partial, reduce
from itertools   import product, repeat
from operator    import truth, contains, eq, is_not, attrgetter, itemgetter, methodcaller, __add__, is_
from collections import deque

# why this isn't in stdlib ... is probably because Guido does't like functional programming ...
identity    = lambda x      : x
# everybody SHOULD love function composition :-)
compose     = lambda *fns   : (lambda x: reduce(lambda acc, f: f(acc), reversed(fns), x))
choice      = lambda p, t, f: (lambda x: t(x) if p(x) else f(x))  # branch
choice_kw   = lambda p, t, f: (lambda x, **kwargs: t(x, **kwargs) if p(x, **kwargs) else f(x, **kwargs))  # branch
ylppa       = lambda *args  : (lambda f: f(*args))                # ylppa is 'apply' in reverse ...
combine     = lambda f, *fns: (lambda x: f(*map(ylppa(x), fns)))  # f( fn[0](x), fn[1](x), ... )
swap_args   = lambda f      : (lambda a, b, *args, **kwargs: f(b, a, *args, **kwargs))
logic_or    = lambda x, y   : x or y                              # operator.__or__ / __and__ are /bitwise/ ops!
logic_and   = lambda x, y   : x and y                             #               ..
const       = lambda x      : (lambda _: x)                       # return the same value irrespective of input
between     = lambda a, b   : (lambda x: a<=x<b)                  # missing from module 'operator'?
m_itemgetter= lambda *idx   : (lambda x: map(x.__getitem__, idx)) # _ALWAYS_ returns [...], irrespective of #-of-indices
                                                                  #   for laughs, look up 'operator.itemgetter()' => 3 (three!)
                                                                  #   different types of return type depending on arguments! FFS!
# reorder_args: call f with the arguments indicated by idx:
# call f with args[idx[n]] for 0 <= n < len(idx)
# f will be called with len(idx) arguments. can also be used to select/repeat arguments
reorder_args= lambda f, *idx: (lambda *args, **kwargs: f(*m_itemgetter(*idx)(args), **kwargs))
hasattr_    = lambda a             : partial(reorder_args(hasattr, 1, 0), a)
getattr_    = lambda a             : partial(reorder_args(getattr, 1, 0), a)
# it is "setattr(o, a, v)" but we call it as "setattr_(a, v)(o)" thus a,v,o needs to be reorderd to o, a, v, i.e. 2,0,1
# note that setattr_ returns the object itself so it can be chained
setattr_    = lambda a, v          : combine(logic_or, partial(reorder_args(setattr, 2, 0, 1), a, v), identity)
delattr_    = lambda a             : combine(logic_or, partial(reorder_args(delattr, 1, 0), a), identity)
maybe_get   = lambda a, d=None     : choice(hasattr_(a), getattr_(a), const(d))
maybe_set   = lambda a, v          : choice(const(is_not_none(v)), setattr_(a, v), identity)
identity    = lambda x, *a, **kw   : x
#mk_query    = lambda c, t, w, *args: "SELECT {0} FROM {1} WHERE {2}{3}".format(c, t, w, "" if not args else " and "+args[0])
do_update   = lambda x, y          : x.update(y) or x
d_filter    = lambda keys          : (lambda d: dict(((k,v) for k,v in d.iteritems() if k in keys)))
d_filter_n  = lambda keys          : (lambda d: dict(((k,v) for k,v in d.iteritems() if k not in keys)))
# expose the print function as, well, a function
printf      = print
#collectf    = lambda x, y : x.collectfn(y)

filter_true = partial(filter, truth)
is_not_kw   = lambda x, y, **kwargs   : is_not(x, y)
is_not_none = partial(is_not, None)
is_not_none_kw = partial(is_not_kw, None)
is_iterable = combine(logic_or, hasattr_('__iter__'), hasattr_('__getitem__'))
listify     = choice(is_iterable, identity, lambda x: [x])
mk_list     = lambda *args: list(args)
mk_tuple    = lambda *args: args
truth_tbl   = lambda *args: tuple(map(truth, args))
# shorter
GetA        = attrgetter
GetN        = itemgetter
Map         = lambda f: partial(map, f)
Filter      = lambda f: partial(filter, f)

# In Py3 one must sometimes drain an iterable for its side-effect (thx guys).
# Py2:
#     # no print function
#     def p(x):
#       print x
#     # but this does something useful
#     map(p, list(...))
# Py3:
#     # we have print() ...
#     # but this don't do nuttin' (so to speak):
#     map(p, list(...))
#     # this does the printing but discards the results
#     drain( map(p, list(...)) )
# I like this approach: https://stackoverflow.com/a/9372429
drain       = deque(maxlen=0).extend

# Quite a few times I use map(...) for its sideeffects
# So why not have a shortcut "drap()" drain(map(...))
drap        = compose(drain, partial(map))

# A dynamically constructed List wrapper. Whenever the 2to3 tool wants to wrap map() or
# filter() in plain "list(...)", replace with this "List(...)".
# The code will then run optimal under both Py2 and Py3
# (see timing results below)
try:
    # Crude Py2 detection
    r = raw_input
    List = ensure_list = identity
except NameError:
    List = lambda x: list(x)
    ensure_list = lambda f: (lambda *args, **kwargs: list(f(*args, **kwargs)))

# The "_" versions evaluate to something that always yields a 
# list and does that efficiently under both Py2 and Py3
map_       = ensure_list(map)
zip_       = ensure_list(zip)
range_     = ensure_list(range)
filter_    = ensure_list(filter)
enumerate_ = compose(list, enumerate) # enumerate gives list() neither in 2 nor 3

# I've included a source listing of a file "tlist.py" which cleary illustrates this:
#
# tlist.py: # Our objective is construct something which makes our code
# tlist.py: # similar between Py2/Py3 -> such that map()/filter() expressions
# tlist.py: # return a list; our code assumes that len() and []-indexing are
# tlist.py: # immediately possible after map()/filter()
# tlist.py: #
# tlist.py: # The 2to3 tool does this by wrapping map()/filter() in plain "list(...)".
# tlist.py: # But for code which can be run under Py2 as well as Py3 this incurs
# tlist.py: # a penalty if that is done.
# tlist.py: #
# tlist.py: # The timing tests below clearly show this. The net result is that 
# tlist.py: # Py3 is always a factor 3 slower compared to Py2 map()/filter()
# tlist.py: # returning a direct list.
# tlist.py: from __future__ import print_function
# tlist.py: 
# tlist.py: test_list = filter(lambda x: x%2==0, range(1000))
# tlist.py: 
# tlist.py: try:
# tlist.py:     r = raw_input
# tlist.py:     List = lambda x: x
# tlist.py: except NameError:
# tlist.py:     List = lambda x: list(x)
# tlist.py: 
# tlist.py: # This is definitely fastest but under Py3 does not satisfy our
# tlist.py: # constraint that len() and []-indexing are possible on the return value
# tlist.py: def immediate():
# tlist.py:     return test_list
# tlist.py: 
# tlist.py: # This is what 2to3 does, wrap map()/filter() in "list(...)"
# tlist.py: def wrapped():
# tlist.py:     return list(test_list)
# tlist.py: 
# tlist.py: # This is what I propose to wrap with instead
# tlist.py: def dynamic():
# tlist.py:     return List(test_list)
# tlist.py: 
# tlist.py: 
# tlist.py: if __name__=='__main__':
# tlist.py:     import timeit
# tlist.py:     print("immedate: ", timeit.timeit("immediate()", setup="from __main__ import immediate"))
# tlist.py:     print("wrapped: ",  timeit.timeit("wrapped()", setup="from __main__ import wrapped"))
# tlist.py:     print("dynamic: ",  timeit.timeit("dynamic()", setup="from __main__ import dynamic"))
# tlist.py: 
# tlist.py: ########################################################################################
# tlist.py: #  Timing results using anaconda/Python2 
# tlist.py: #  immedate:  0.133291959763
# tlist.py: #   wrapped:  1.37940621376
# tlist.py: #   dynamic:  0.129576206207
# tlist.py: #
# tlist.py: #  Timing results using anaconda/Python3 
# tlist.py: #  immedate:  0.12062911898829043
# tlist.py: #   wrapped:  0.419401798164472
# tlist.py: #   dynamic:  0.39737311704084277
# tlist.py: #
# tlist.py: #  These were repeatable, done on the same hardware.
# tlist.py: #  So the "dynamic" approach is fastest compatible between Py2/Py3, with Py3 factor
# tlist.py: #  of ~3 slower!
