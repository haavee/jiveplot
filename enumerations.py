""" 
Python can support enumerations/enum like types, a finite set of immutable symbolic values that either stand for themselves (the name IS the value) or refer to another, fixed, value.

An enumeration can be constructed using the Enum(...) function:
    axes = enumerations.Enum('x', 'y', 'z')

The returnvalue is an immutable object with attributes 'x', 'y', 'z' whose value is the attribute itself:

    # equality, irrespective how the enumerated name is adressed, is very equal; 'is' and '==' behave equal
    axes['x'] is axes[axes.x] is axes.x
    axes['x'] == axes[axes.x] == axes.x == 'x'

All enumerated names have an associated, read-only, value:
    assert axes.x.value == 'x'

It is possible to construct an enum where the associated value is not the default/generated 'str(...)' of the enumerated name:
    status = enumerations.Enum(OK=200, NotFound=404)

    assert status.OK.value == 200

Elements from the enumeration can be used as ordinary variables:
    axtp = axes.x

    if axtp is axes.y:
        # do something
    if axtp == axes.y:
        #...

    # they can be used as keys in dicts
    rot = {axes.x: /*stuff*/, axes.y: /*other stuff*/}

    # Enums can be tested for presence based on enumerated name or the enum itself:
    assert 'x' in axes
    assert axes.x in axes
    #    "<string> in Enum(...)" finds the enumeration with name <string>
    if input('axis?> ') not in axes:
        raise RuntimeError("Unknown axis type ...?")

    # can be iterated over
    for ax in axes:
        # do something for the specific enumeration

    # so they have length
    print len(axes)
    for i in range(len(axes)):
        print "enum element ",i," = ", axes[i]

    # looked up by value or by name of the enumeration element
    assert axes.index('z')    == 2
    assert axes.index(axes.z) == 2
    assert status[ status.index('OK') ]      is status.OK
    assert status[ status.index(200) ]       is status.OK
    assert status[ status.index(status.OK) ] is status.OK

    # The most important part: immutability!
    axes.q            => AttributeError: type object 'Enum(x,y,z)' has no attribute 'q'
    axes.x = 42       => RuntimeError: Attempt to overwrite enumerated value - cannot set Enum(x, y, z).x = 42
    axes[1] = 42      => TypeError: Seriously? Trying to assign into an enumeration?? Enum(x,y,z)[1] = 42 ... tssssk
    axes.foo = 42     => TypeError: You cannot set attributes on an enumeration
    axes.x.value = 42 => AttributeError: can't set attribute
    del axes.x        => TypeError: Cannot delete attribute x from enumeration

    # ValueError if attempt to get an invalid item:
    foo = Enum('bar')
    axes['bar']    => ValueError: Enum(x,y,z) does not contain the value bar
    axes[foo.bar]  => ValueError: Enum(x,y,z) does not contain the value bar
"""
from   six import with_metaclass
import operator, collections

# Python 2.6 don't have OrderedDict in the collections module. Sigh.
try:
    Dict = collections.OrderedDict
except AttributeError:
    # provide workaround that preserves order (insofar that's possible; kwargs are unordered)
    # and ignore duplicate keys
    class Dict(list):
        def __init__(self, *args, **kwargs):
            assert len(args) in [0,1]
            self.seen = set()
            self.update( *args, **kwargs )

        def update(self, *args, **kwargs):
            for (k,v) in (args[0] if args else ()):
                if k in self.seen:
                    continue
                self.append( (k,v) )
                self.seen.add( k )
            for (k,v) in kwargs.iteritems():
                if k in self.seen:
                    continue
                self.append( (k,v) )
                self.seen.add( k )

        def iteritems(self):
            return self



class EnumValueMeta(type):
    def __new__(cls, name, parents, dct):
        # Check if the meta'ed class is conform our expectations
        ev = dct.get('_enumvalue', None)
        if ev is None:
            raise TypeError("EnumValue/the modified class has no _enumvalue set!")
        mName = str(ev[0])
        dct['__str__'] = dct['__repr__'] = classmethod(lambda self: mName)
        dct['_enumtypes'] = tuple(map(type, ev))
        # Maybe you want to disable creating instances of the enum values?
        return super(EnumValueMeta, cls).__new__(cls, mName, parents, dct)

    # read-only property '.value' 
    @property
    def value(cls):
        return cls._enumvalue[1]

    def __call__(cls):
        return cls

    def __eq__(self, other):
        (ev, et) = (self._enumvalue, self._enumtypes)
        ot = type(other)
        return True if id(self)==id(other) else ((ev[0]==other if et[0] is ot else False) or (ev[1]==other if et[1] is ot else False))

    def __do_compare__(self, other, f):
        (ev, et) = (self._enumvalue, self._enumtypes)
        ot = type(other)
        try:
            t_idx = 1 if ot is type(self) else et.index(ot)
            return f(ev[t_idx], other)
        except ValueError:
            raise TypeError("Comparing invalid types: self={0} other={1}".format(et, ot))

    def __le__(self, other):
        return self.__do_compare__(other, operator.__le__)
    def __lt__(self, other):
        return self.__do_compare__(other, operator.__lt__)
    def __ge__(self, other):
        return self.__do_compare__(other, operator.__ge__)
    def __gt__(self, other):
        return self.__do_compare__(other, operator.__gt__)

    def __hash__(self):
        return hash(self._enumvalue[1])

    def __ne__(self, other):
        return not self == other 

    def __del__(self):
        pass
        #raise TypeError("Cannot delete an enumeration value")

    def __delattr__(self, a):
        raise TypeError("Cannot delete an enumeration value")

    def __str__(self):
        return self.__str__()
    __repr__ = __str__


# Keep a cache of known enumerations such that repeatedly making literally the same enumeration
# consistently returns literally the same enumerations (compares equal by id)
_knownEnums = dict()
class EnumMeta(type):
    # In the new phase:
    # - we form the new name
    # - we transform the _enums set into a dict [enum, value]
    # - disable some methods
    # - set some standard methods to predictable values
    def __new__(cls, name, parents, dct):
        global _knownEnums

        enums  = dct.get('_enums', ())
        # Prevent people trying to break things
        if sum(1 if kv[0] in dir(cls) else 0 for kv in enums):
            raise TypeError("Attempt to overwrite one or more of the class's methods with an enumerated name!")
        # check if we already have this set of enums
        eptr   = _knownEnums.get(enums, None)
        if eptr is not None:
            return eptr
        # OK don't have this specific key
        # form name for this one
        myname = "Enum({0})".format(','.join(map(str, map(operator.itemgetter(0), enums))))
        # build dict of enum name to value mapping
        # use list to preserve the order
        enumvals = list()
        for ev in enums:
            class EnumValueImpl(with_metaclass(EnumValueMeta)):
                _enumvalue    = ev
            enumvals.append( EnumValueImpl )
            dct[ ev[0] ] = EnumValueImpl

        # Now we can start modifying the class definition
        # - disable the ability to create instances of these things or to overwrite attributes
        for k in ['__setattr__', '__init__', '__new__']:
            if k in dct:
                del dct[k]
        # - string represention looks like class name
        dct['__repr__']    = dct['__str__']  = classmethod(lambda self: myname)
        # replace the _enums set with the tuple'd list of enums
        dct['_enums']      = tuple(enumvals)
        # now we can create the type instance
        return _knownEnums.setdefault(enums, super(EnumMeta, cls).__new__(cls, myname, parents, dct))

    def __call__(cls):
        return cls

    def __str__(self):
        return self.__str__()
    __repr__ = __str__

    def __contains__(self, x):
        return x in self._enums

    def __iter__(self):
        return iter(self._enums)

    def __len__(self):
        return len(self._enums)

    def index(self, x, *args):
        # thanks Python3 for having to explicitly make a list FFS!
        lst = list(map(str, self._enums) if isinstance(x, str) else self._enums)
        return lst.index(x, *args)

    def __getitem__(self, a):
        try:
            idx = a if isinstance(a, int) else self._enums.index(a)
            return self._enums[idx]
        except ValueError:
            raise ValueError("{0} does not contain the value {1}".format(self, a))

    def __setitem__(self, a, v):
        raise TypeError("Seriously? Trying to assign into an enumeration?? {0}[{1}] = {2} ... tssssk".format(self, a, v))

    def __delitem__(self, a):
        raise TypeError("Cannot delete item {0} from enumeration".format(a))

    def __setattr__(self, a, v):
        # We cannot allow overwriting enumerated values but we can allow other attributes to be set?
        if a in self:
            raise RuntimeError("Attempt to overwrite enumerated value - cannot set {0}.{1} = {2}".format(self, a, v))
        raise TypeError("You cannot set attributes on an enumeration")

    def __delattr__(self, a):
        raise TypeError("Cannot delete attribute {0} from enumeration".format(a))


def Enum(*names, **namedvalues):
    """
    *args     will be the enumerated names with associated value of args[i] == str(args[i])
    **kwargs  for assigning a non-default/generated associated value for the enumerated name"""
    # remove duplicates from enums but keep relative order
    #enums = collections.OrderedDict((n, str(n)) for n in names)
    enums = Dict(((n, str(n)) for n in names))
    if len(enums)!=len(names):
        raise TypeError("Duplicate names detected in enumerated names")
    # add the named values
    enums.update( **namedvalues )
    if len(enums)!=(len(names)+len(namedvalues)):
        raise TypeError("Duplicate names detected between names and named values")
    class EnumImpl(with_metaclass(EnumMeta, object)):
        _enums        = tuple((enums.iteritems if hasattr(enums, 'iteritems') else enums.items)())
    return EnumImpl



if __name__ == '__main__':
    import unittest
    mk = lambda x: type(x, (), {'attribute':42})

    class TestConstruction(unittest.TestCase):
        def test_noduplicate_names(self):
            self.assertRaises(TypeError, Enum, 'aap', 'noot', 'aap')

        def test_noduplicates_at_all(self):
            self.assertRaises(TypeError, Enum, 'aap', 'noot', aap=42)

        def test_no_overwrite_of_special_names(self):
            self.assertRaises(TypeError, Enum, '__new__', '__init__', '__len__', 'index')


    class TestBasics(unittest.TestCase):

        def setUp(self):
            self.a = Enum('aap', 'noot')
            self.b = Enum('mies')
            self.c = Enum('aap', 'noot')
            self.d = Enum('url', OK=200, NotFound=404)
            self.fbb = tuple(map(mk, ['Foo', 'Bar', 'Baz']))
            self.q = Enum(self.fbb[0], self.fbb[1], None)

        def test_lengtha(self):
            self.assertEqual(len(self.a), 2)

        def test_lengthb(self):
            self.assertEqual(len(self.b), 1)

        def test_lengthd(self):
            self.assertEqual(len(self.d), 3)

        def test_str_repr_equal(self):
            self.assertEqual( str(self.a), repr(self.a) )

        def test_expected_name(self):
            self.assertEqual( str(self.a), 'Enum(aap,noot)' )

        # can convert to list/iterate
        def test_as_list(self):
            # FFS! Py2.7 has assertItemsEqual but Py3 doesn't;
            #      there it's called assertCountEqual ARGH!
            #      So we gonna use the compatible recipe
            self.assertEqual(list(sorted(self.a)), list(sorted([self.a.noot, self.a.aap])))

        # a and c a constructed with same elements in same order
        # should yield identical enums
        def test_same_args_equal_same_enum(self):
            self.assertEqual(self.a, self.c)
        def test_same_args_IS_same_enum(self):
            self.assertIs(self.a, self.c)
        def test_diff_args_IS_not_same_enum(self):
            self.assertIsNot(self.a, self.b)

        def test_name_equal_str(self):
            self.assertTrue( self.a.aap   == 'aap' )
        def test_name_equal_str(self):
            self.assertFalse( self.a.aap  == 'noot' )
        def test_oper_not_equal_true(self):
            self.assertTrue( self.b.mies != 'aap' )
        def test_oper_not_equal_false(self):
            self.assertFalse( self.b.mies != 'mies' )
        def test_value_name(self):
            self.assertEqual( self.a.aap.value, 'aap' )
        def test_value_value(self):
            self.assertEqual( self.d.OK.value,  200 )

        def test_calling_yields_self_enum(self):
            # calling the enum or the enumerated value(s) yields identical objects
            self.assertIs(self.a,      self.a())
        def test_calling_yields_self_enum_value(self):
            self.assertIs(self.a.noot, self.a.noot())

        # test for membership can be done by enumerated constant or string; cross enum shouldn't work
        def test_contains_attr(self):
            self.assertIn(self.a.aap, self.a)
        def test_contains_attr_call(self):
            self.assertIn(self.a.aap(), self.a)
        def test_contains_str(self):
            self.assertIn('aap', self.a)
        def test_not_contains_str(self):
            self.assertNotIn('mies', self.a)
        def test_not_contains_other_enum(self):
            self.assertNotIn(self.b.mies, self.a)
        def test_contains_valued_enum(self):
            self.assertIn('OK', self.d)

        # getitem with invalid/not found name/value
        def test_getitem_attr_other_enum(self):
            with self.assertRaises(ValueError):
                self.a[ self.b.mies ]
        def test_getitem_of_not_present_name(self):
            with self.assertRaises(ValueError):
                self.a[ 'mies' ]
        def test_getitem_out_of_range(self):
            with self.assertRaises(IndexError):
                self.d[ 403 ]

        # indexing by number, name or string
        def test_getitem_number_IS_enum(self):
            self.assertIs(    self.a[0], self.a.aap )
        def test_getitem_number_equal_string(self):
            self.assertEqual( self.a[0], 'aap' )
        def test_getitem_string_IS_enum(self):
            self.assertIs( self.a['aap'],      self.a.aap )
        def test_getitem_string_equal_string(self):
            self.assertEqual( self.a['aap'],   'aap' )
        def test_getitem_enum_IS_enum(self):
            self.assertIs( self.a[self.a.aap], self.a.aap )
        def test_getitem_enum_equal_enum(self):
            self.assertEqual( self.a[self.a.aap], self.a.aap )
        def test_getitem_enum_equal_string(self):
            self.assertEqual( self.a[self.a.aap], 'aap' )

        # .index(...) (cf. builtin list.index(...))
        def test_index_str(self):
            self.assertEqual( self.a.index('aap'), 0 )
        def test_index_enum(self):
            self.assertEqual( self.a.index(self.a.aap), 0 )
        # lookup of named value (e.g. "OK=200")
        def test_index_value(self):
            self.assertEqual( self.d[self.d.index(200)], self.d.OK )

        def test_index_not_found_value(self):
            with self.assertRaises(ValueError):
                # HTML code 403 not 'defined' in self.d
                self.d.index( 403 )
        def test_index_not_found_name(self):
            with self.assertRaises(ValueError):
                self.a.index( 'not in a' )

        def test_no_attributes_on_enum(self):
            with self.assertRaises(TypeError):
                self.a.foo = 42 

        def test_no_assign_attribute(self):
            with self.assertRaises(RuntimeError):
                self.a.aap    = 42 
        def test_no_assign_valued_enum(self):
            with self.assertRaises(RuntimeError):
                self.d.NotFound = 42 
        def test_no_assign_index(self):
            with self.assertRaises(TypeError):
                self.a[0]     = 42 
        def test_no_assign_index_name(self):
            with self.assertRaises(TypeError):
                self.a['aap'] = 42

        def test_no_assign_value(self):
            with self.assertRaises(AttributeError):
                self.d.OK.value = 42
       
        # it should be impossible to delete an enum value,
        # an enumeration value's value - even through indexing
        def test_no_del_property(self):
            with self.assertRaises(TypeError):
                del self.d.OK.value
        def test_no_del_attribute(self):
            with self.assertRaises(TypeError):
                del self.a.aap
        def test_no_del_getitem(self):
            with self.assertRaises(TypeError):
                del self.a[1]

        # Enum with types as enumerated values?
        def test_none_in_enum(self):
            self.assertIn(None, self.q)

        def test_lookup_none_equal(self):
            self.assertEqual(self.q[None],    'None')

        def test_contains_type(self):
            self.assertIn( self.fbb[0], self.q )
        def test_contains_type_str(self):
            self.assertIn( str(mk('Foo')), self.q )
        def test_does_not_contain_str(self):
            self.assertNotIn( str(mk('Baz')), self.q )
        def test_does_not_contain_type(self):
            self.assertNotIn( self.fbb[2], self.q )

    unittest.main()
