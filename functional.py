from functools import partial
from itertools import product, repeat
from operator  import truth, contains, eq, is_not, attrgetter, itemgetter, methodcaller, __add__, is_

# everybody SHOULD love function composition :-)
compose     = lambda *fns   : (lambda x: reduce(lambda acc, f: f(acc), reversed(fns), x))
choice      = lambda p, t, f: (lambda x: t(x) if p(x) else f(x))  # branch
choice_kw   = lambda p, t, f: (lambda x, **kwargs: t(x, **kwargs) if p(x, **kwargs) else f(x, **kwargs))  # branch
#ylppa       = lambda x      : (lambda f: f(x))                    # ylppa is 'apply' in reverse ...
ylppa       = lambda *args  : (lambda f: f(*args))                    # ylppa is 'apply' in reverse ...
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
#printf      = lambda x, y : x.printfn(y)
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

