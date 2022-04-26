# command line utility - abstracts handling of commands from defining/implementing them
# A rewrite from the glish based version from jivegui.ms2
from   __future__ import print_function
try:               import builtins
except ImportError:import __builtin__ as builtins
import re, sys, copy, pydoc, fcntl, termios, struct, os, itertools, math, tempfile, traceback, functional
import hvutil, glob
from   six        import iteritems
from   six.moves  import input as raw_input
from   functional import List, drain, drap, map_, filter_, GetA, GetN, compose
from   functools  import reduce

# if we have readline, go on, use it then!
# we attempt to save the history across invocations of this prgrm
haveReadline = None
try:
    import readline
    haveReadline = True
except:
    pass

class UnknownCommand(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return "the command '{0}' is unknown".format(self.msg)

## The command object will use a "line source" object which
## should expose (at least) the context protocol
## (http://docs.python.org/2/library/stdtypes.html#context-manager-types)
## and the iterator protocol
## (http://docs.python.org/2/library/stdtypes.html#iterator-types)
##
## The ".run( <linesource> )" method uses the line source object
## as:
##       self.run( <linesource> ):
##          with <linesource> as tmp:
##             for line in tmp:
##                execute( line )
##
## This allows the linesource object to create a context which
## will be automagically destroyed when the ".run()" is finished.
##
## The "readkbd" line source class uses this context to save &
## restore the current history [== Python interactive shell's history]
## and to restore and save the history of the command line environment,
## the application's interactive shell.
##
## This way the two histories remain nicely separated/unpolluted

## This is the base class implementing a push/pop of current history
## and push/pop of (temporary) alternative history. Classes which
## want their own readline history saved can derive from this one.
class newhistory(object):
    
    ## The actual history for the new environment will be stored/retrieved
    ## from ${HOME}/.<basename>.history
    def __init__(self, basename):
        self.basename = basename

    # support overload readline completion
    def completer(self, *args):
        return None

    def __enter__(self):
        # we only do something if we have readline
        if not haveReadline:
            return self
        ## Set up the new history context
        self.historyFile      = os.path.join( os.getenv('HOME'), ".{0}.history".format(self.basename)) 
        (h, self.oldhistFile) = tempfile.mkstemp(prefix=self.basename, suffix=".hist", dir="/tmp")
        # only need the filename, really
        os.close(h)
        readline.write_history_file(self.oldhistFile)
        readline.clear_history()

        # if reading the old history fails, fail silently
        # (the file might not exist yet)
        try:
            readline.read_history_file(self.historyFile)
        except:
            pass

        # store the old completer, install our own one
        readline.parse_and_bind("tab: complete")
        #readline.parse_and_bind("C-c: backward-kill-line")
        self.oldCompleter = readline.get_completer()
        readline.set_completer(self.completer)
        return self
  
    # clean up the context
    def __exit__(self, ex_tp, ex_val, ex_tb):
        if not haveReadline:
            return False
        try:
            # update the history for this project
            readline.write_history_file(self.historyFile)
        except Exception as E:
            print("Failed to write history to {0} - {1}".format(self.historyFile, E))
        # put back the original history!
        readline.clear_history()
        readline.read_history_file(self.oldhistFile)
        os.unlink(self.oldhistFile)

        # and put back the old completer
        readline.set_completer(self.oldCompleter)

# linegenerators, pass one of these to the "run"
# method of the CommandLineInterface object
#import ctypes
#rllib = ctypes.cdll.LoadLibrary("libreadline.so")
#rl_line_buffer = ctypes.c_char_p.in_dll(rllib, "rl_line_buffer")
#rl_done        = ctypes.c_int.in_dll(rllib, "rl_done")

class readkbd(newhistory):
    ## create an interactive keyboard reader with prompt <p>
    def __init__(self, p):
        super(readkbd, self).__init__(p)
        self.prompt   = p
        self.controlc = None
        # we want readline completer to give us the "/" as well!
        readline.set_completer_delims( readline.get_completer_delims().replace("/", "").replace("-","") )
        print("+++++++++++++++++++++ Welcome to cli +++++++++++++++++++")
        print("$Id: command.py,v 1.16 2015-11-04 13:30:10 jive_cc Exp $")
        print("  'exit' exits, 'list' lists, 'help' helps ")

    # add the iterator protocol (the context protocol is supplied by newhistory)
    def __iter__(self):
        return self

    def handle(self, exception):
        print(exception)

    def __next__(self):
        try:
            l = raw_input(self.prompt+"> ")
            if self.controlc:
                #rl_done.value = 0
                self.controlc = None
                return None
            return l
        except (EOFError):
            quit = True
            print("\nKTHXBYE!")
        except KeyboardInterrupt:
            # user pressed ctrl-c whilst something was 
            # in the buffer. Make the code skip the next line of input.
            if len(readline.get_line_buffer())>0:
                #self.controlc = readline.get_line_buffer()
                self.controlc = True
                #print "rlbuf: ",rl_line_buffer.value
                #rl_line_buffer.value = ""
                #print "rl_done = ",rl_done
                #rl_done.value = 1
                #print "rl_done = ",rl_done
                print("\nYour next line of input might be ignored.\nI have not understood readline's ^C handling good enough to make it work.\nFor the moment just type <enter> and ignore the displayed text.")
            return None
        if quit:
            raise StopIteration

    next = __next__

    def completer(self, text, state):
        if not haveReadline:
            return None
        if state==0:
            self.options = glob.glob(text+"*")
        try:
            return self.options[state]
        except IndexError:
            return None

## Read commands from a string. Doesn't need its own
## history context so the context __enter__/__exit__ are
## basically no-ops
class readstring:
    def __init__(self, s):
        self.lines = s.split('\n')

    def handle(self, exception):
        print(exception)
        sys.exit( -1 )

    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return False
    def __iter__(self):
        return iter(self.lines)

## Read commands from a file. Context identical to
## readstring() above.
class readfile:

    ## Allow creation with (optional) arguments - each line in 
    ## the file will be interpreted as a string formatting
    ## command (http://docs.python.org/2/library/string.html#format-string-syntax)
    ##
    ## The string from the file will be used as:
    ##     line = <file>.next().format( *self.args )
    ##
    ## Where "self.args" is, effectively,
    ##     "args[0].split()" 
    ## ie the first (optional) argument will be
    ##   1) interpreted as string
    ##   2) split into pieces at whitespace
    ##      (honours quoting - whitespace inside
    ##       quotes does not split)
    ##   3) this list of split pieces is given to
    ##      ".format( ... )"
    ##
    ## Example:
    ##    $> cat script.scr
    ##    hello {0}; set_temp {1}; mask={2}
    ##
    ##    Then, in a CommandLineInterface you can issue the command:
    ##       jcli> play script.scr aap 42 0xdead
    ##    
    def __init__(self, f, *args):
        if not f:
            raise ValueError("no filename given to readfile()")

        try:
            self.args     = quote_split(args[0], ' ') if args and args[0] else []
            self.file     = open(f)
            self.filename = f
        except IOError as e:
            raise ValueError("readfile '{0}' - {1}".format(f, e))

    def handle(self, exception):
        print(exception)

    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return False
    def __iter__(self):
        return self
    def __next__(self):
        line = None
        try:
            line = re.sub(r"#.*$", "", self.file.next().rstrip('\n'))
            return line.format(*self.args)
        except IndexError:
            print("readfile[{0}]: Not enough arguments for script-line:".format(self.filename))
            print(line)
            raise StopIteration
        except IOError as e:
            print("readfile[{0}]: {1}".format(self.filename, e))
            raise StopIteration

    next = __next__

class scripted:
    ## each script arg is supposed to be
    ## an iterable, producing commands to be interpreted
    def __init__(self, *scripts):
        self.commands = itertools.chain(*scripts)

    # any exception terminates the scribd
    def handle(self, exception):
        print(exception)
        sys.exit( -1 )

    ## Support the context protocol
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return False

    # the iterator protocol
    def __iter__(self):
        return self.commands




# in order to make a command call this function
# you have to fill in at least the "id" and "rx" fields
# and the "cb" callback function to execute if 
# "rx" matches. "rx" is a regular expression and
# "cb" a callable.
#
# You may add the "expand" member. Anything evaluating
# to False means macro substitution on the command is
# not done. If missing defaults to "True"
#
# It's advisable to also set the "hlp" member to,
# say, the docstring of the function - the system's
# builtin "help" command uses it.
#
# You _may_ specify an "args" function which will be 
# executed to transform the command into an argument 
# list for the callback function, otherwise the whole
# line will be passed as argument to the callback.
#
#  execute(txt) ->
#     for c in commands:
#       if c.rx.match(txt):
#         arguments = None
#         if c.args:
#           arguments = c.args(txt)
#         if arguments:
#           c.cb(arguments)
#         else:
#           c.cb()
#
# If your callback function has optional arguments 
# (not unthinkable) it's betst to define the callback as:
#
#  def my_callback(*args):
#       # find out how we were called
#       if not args:
#           # called w/o arguments
#       else:
#           for arg in args:
#              foobar(arg)
#
# Example:
#
#  the "foo" command takes an optional integer argument.
#  the "rx" takes care of matching this exactly; "foo bar" won't match
#    (and the "cb" not called) "foo 1" will.
#  if the "rx" matches, then the "args" function will be called to
#    sanitize the command - it strips the leading "foo" including all
#    whitespace - therefore leaving only the integer argument (in text
#    representation)
#  the "cb" function will be called with the - optional - argument
#    and if it was given, convert to int and add 42.
#    nothing happens - you can't print from a lambda (#GVD KUDTPYTHON!)
#    but suppose your "cb" is a real function (or the lambda calls a 
#    real function) then you _can_ make stuff happen.
#  the "id" field is used as the short form of the command, for
#    listing all commands
#
#   mkcmd(rx=re.compile("^foo(\s+[0-9]+)?$",
#         id="foo",
#         args=lambda x: re.sub("^foo\s+",""),
#         cb=lambda x: int(x)+42 if x else -1, 
#         hlp="foo [<number>]\nthis foo's the number, if given")
def mkcmd(**kwargs):
    # create an instance of an anonymous type
    o = type('', (), {})()
    drap(lambda a_v: setattr(o,a_v[0],a_v[1]), iteritems(kwargs))
    return o

## oneliner to discard all characters up until the comment character
stripcomment = lambda s, cmt='#': \
    ''.join(itertools.takewhile(lambda x: x!=cmt, s))

## quote_split  do not extract _words_ but split the
##              string at the indicated character,
##              provided the character is not inside
##              quotes
##
##      "aap  'noot 1; mies 2'; foo bar"
##  results in:
##      ["aap 'noot 1;mies 2'", "foo bar"]
def quote_split(s, splitchar=';'):
    rv = [""]
    inquote = False
    for i in builtins.range(len(s)):
        if not inquote and s[i]==splitchar:
            rv.append( "" )
            continue
        rv[-1] = rv[-1] + s[i]
        if s[i]=="'":
            inquote = not inquote
    return rv


## Return a compiled regex which matches the word-boundaried word 'x'
## See http://docs.python.org/2/library/re.html under the "\b"
## special matcher
wordmatch = lambda x : re.compile(r"\b{0}\b".format(x))

class CommandLineInterface:
    def __init__(self, **kwargs):
        self.exit     = None
        self.commands = []
        self.macros   = {}
        self.debug    = kwargs.get('debug', False)
        self.app      = kwargs.get('app', None)

        # add builtins
        self.commands.append( \
            mkcmd(id="exit", rx=re.compile(r"^exit$"), \
				  hlp='exit\n\texit the commandline interface',\
                  cb=lambda : self._exit()) )
        self.commands.append( \
            mkcmd(id="list", rx=re.compile(r"^list$"), \
				  hlp='list\n\tlist all commands',\
                  cb=lambda : self.listCommands()) )
        self.commands.append( \
            mkcmd(id="help", rx=re.compile(r"^help\b.*$"), \
				  hlp='help [command1 ... commandn]:\n\tdisplay help\nWithout arguments, display one line of help for each command, otherwise full help command1 .. commandn', \
                  expand=False,
                  args=lambda x: re.sub(r"^help\s*", "", x).split(), \
                  cb=lambda *args: self.help(*args)) )
        self.commands.append( \
            mkcmd(id="macro", rx=re.compile(r"^macro(\s+\S+(\s+('.+?(?<!\\)'|\S+))?)?$"),\
                  hlp=hlp_macro, expand=False,\
                  args=lambda x: re.sub(r"^macro\s*","",x), \
                  cb=lambda x : self.macro(x)) )
        self.commands.append( \
            mkcmd(id="del", rx=re.compile(r"^del(\s+\S+)?$"),\
				  hlp='del <name>:\n\tremove definition of macro <name>, if such a macro exists, otherwise no-op', \
                  expand=False,\
                  args=lambda x: re.sub(r"^del\s*","",x), \
                  cb=lambda x : self.delMacro(x)) )
        self.commands.append( \
            mkcmd(id="play", rx=re.compile(r"^play\s+\S+(\s+\S.*)?$"), \
                  hlp=hlp_play, \
                  args=lambda x: [re.match(r"^play\s*(?P<file>\S+)(\s+(?P<args>.*))?$", x)], \
                  cb=lambda x: self.run(readfile(x.group('file'), x.group('args'))) ) )
        self.commands.append( \
            mkcmd(id="!", rx=re.compile(r"^!.*"), \
                  hlp=hlp_shell, \
                  args=lambda x: re.sub(r"^!\s*", "", x),\
                  cb=lambda x: self.runShell(x)) )

        # load macros from file?
        self.loadMacros()

    def run(self, linesrc):
        self.exit = None
        with linesrc as l:
            for line in l:
                if not line:
                    continue
                try:
                    drap(lambda x: self.parseOneCmd(x) if not self.exit else None, \
                              quote_split(stripcomment(line)))
                except Exception as E:
                    if self.debug:
                        traceback.print_exc()
                    # delegate actual handling of the exception to the linesrc
                    linesrc.handle(E)
                if self.exit:
                    break

    def listCommands(self):
        s = []

        # compute maximum width of a command name
        ids   = List(map(GetA('id'), self.commands))
        width = max(map(len, ids))
        fmt   = "{{0:<{0}}}".format(width+2).format
        def p(x):
            if len(s)>0 and len(s[-1])<(6*width):
                s[-1] += x
            else:
                s.append(x)
        drap(compose(p, fmt), sorted(ids))

        s.append("=== Macros   ===")
        s.extend( map_("{0[0]} => '{0[1]}'".format, iteritems(self.macros)) )
        maybePage(s)

    def addCommand(self, cmd):
        if not hasattr(cmd, "id") or not hasattr(cmd, "cb") or not hasattr(cmd, "rx"):
            raise RuntimeError("Command MUST have an 'id' field, a 'cb' (callback) field and an 'rx' field")
        # check if there's not a macro/command with this name already
        for x in self.commands:
            if x.id==cmd.id:
                raise RuntimeError("Command '{0}' already exists".format(cmd.id))
        for (n,v) in iteritems(self.macros):
            if cmd.rx.match(n):
                raise RuntimeError("Attempt to add command '{0}', which is already defined as macro".format(cmd.id))
        self.commands.append(cmd)

    # attempt to add macro, which is tuple(name, value)
    def addMacro(self, macro, **kwargs):
        store     = kwargs.get('store', True)
        n, v      = macro
        # We need to do cycle detection - 
        # (1) create an updated macro definition set, including the
        #     new definition
        nmacro    = copy.deepcopy(self.macros)
        nmacro[n] = v
        # (2) transform into a graph:
        #       name => [list, of, entities]
        #     (for each macro, compile the list of 
        #      macro names that are found in the value)
        def reductor(acc, k_v):
            acc[k_v[0]] = filter_(lambda txt: re.search(wordmatch(txt), k_v[1]), nmacro.keys())
            return acc
        graph = reduce(reductor, iteritems(nmacro), {})
        # (3) detect cycles
        if hvutil.cycle_detect(graph):
            raise SyntaxError("The macro definition '{0} => {1}' would create a loop in macroexpansion!".format(n, v))
        #      ...
        # (5) Profit!
        self.macros = nmacro
        if store:
            self.storeMacros()

    def delMacro(self, x):
        if x in self.macros:
            del self.macros[x]
            self.storeMacros()
            print("deleted macro '{0}'".format(x))

    def parseOneCmd(self, txt):
        # strip whitespaces at the end and perform macrosubstitutions
        t  = copy.deepcopy(txt.strip())
        if len(t)==0:
            return
        # Ok, see if we find a match
        cmd    = self._isCmd(t)
        # Allow macro expansion? Defaults to True otherwise take the value of 
        # the ".expand" attribute, if it has it
        # At this point we may find a macro here, so it wouldn't be recognized 
        # as a command but after macro expansion it could!
        expand = cmd.expand if cmd and hasattr(cmd, "expand") else True
        if expand:
            t2  = self._doMacroSubstitution(t)
            # if macros were substituted, we may 
            # end up with >1 command. Process them 
            # as we do all other commands
            if t2 != t:
                self.run(readstring(t2))
                return
        if not cmd:
            raise UnknownCommand(t)
        # prepare the arguments, if any
        #   NOTE: if you use "def X(*args)" and use as follows:
        #            y = "some string" 
        #            X(*y)
        #         then X() gets called with len(y) arguments -
        #         the individual characters of the string 8-/
        args = None
        if hasattr(cmd, "args"):
            args = cmd.args(t)
        # and call the callback
        if args is None:
            cmd.cb()
        elif isinstance(args, str):
            # see above. prevent string expansion into individual characters
            # because that's hardly ever the right answer
            cmd.cb(args)
        else:
            #cmd.cb(args)
            # HV: TODO FIXME XXX 
            #     Need to rework the callbacks to use "*args" such that
            #     IF a command provides an "arg" function to create 
            #     (a list of) arguments, the function actually is called
            #     with that many arguments and not with one tuple
            cmd.cb(*args)

    def help(self, *args):
        # We need a function to extract the helptexts
        # of (1) the indicated commands or (2) all
        # commands. 
        # If all commands are selected we only print the
        # first line of the texts
        nohlp = lambda c : "{0} - no help available".format(c)
        txts  = []
        if not args:
            # display the first two lines of helptext for every command
            txts = [hvutil.strcut(x.hlp, 2, '\n') if hasattr(x, "hlp") else nohlp(x) for x in self.commands]
        else:
            def txt(c):
                try:
                    [cmd] = [x for x in self.commands if x.id==c]
                    # display full help for command if available
                    return cmd.hlp if hasattr(cmd, "hlp") else nohlp(c)
                except ValueError:
                    return "{0} - no help for unknown command".format(c)
            txts = map_(txt, args) 
        maybePage( txts )

    def macro(self, args, **kwargs):
        verbose = kwargs.get('verbose', True)
        store   = kwargs.get('store'  , True)
        # args is the string following the macro command, if any
        # we use the escape split on it - honouring quotation
        # three cases: len(args)==0 => just the 'macro' command, list all macros
        parts = hvutil.escape_split(args)
        if len(parts)>2:
            raise RuntimeError("internal error - too many arguments to 'macro' command")
        if len(parts)==0:
            # display all macro definitions
            drap(compose(print, "{0[0]} => '{0[1]}'".format), iteritems(self.macros))
        elif len(parts)==1:
            # display the definition of macro 'xxx' (if any)
            try:
                print("{0} => '{1}'".format(parts[0], self.macros[parts[0]]))
            except KeyError:
                print(parts[0]+" => no such macro defined")
        else:
            # attempt to define the macro
            # make sure no command of that name exists
            for cmd in self.commands:
                if cmd.rx.match(parts[0]):
                    raise RuntimeError("'{0}' already exists as command".format(parts[0]))
            self.addMacro( (parts[0], parts[1]), store=store )
            if verbose:
                print(parts[0]+" => "+parts[1])
        return None

    def runShell(self, s):
        exitcode = os.system(s)
        if exitcode!=0:
            raise RuntimeError("'{0}' exited with code {1}".format(s, exitcode))

    def _exit(self):
        self.exit = True

    def _isBuiltin(self, txt):
        for cmd in self.builtins:
            if cmd.rx.match(txt):
                return cmd
        return None

    def _isCmd(self,txt):
        for cmd in self.commands:
            if cmd.rx.match(txt):
                return cmd
        return None

    def _doMacroSubstitution(self, txt):
        # Run each macro over the text and
        # stop if the output is the same as the input
        while True:
            otext = copy.deepcopy(txt)
            txt = reduce(lambda acc, n_v: re.sub(wordmatch(n_v[0]), n_v[1], acc), iteritems(self.macros), txt)
            if txt==otext:
                break
        return txt

    def loadMacros(self):
        if not self.app:
            return
        mfn = os.path.join( os.getenv('HOME'), ".{0}.macros".format(self.app))
        try:
            with open(mfn, 'r') as mf:
                reduce(lambda acc, line: self.macro(line, verbose=False, store=False) or acc, mf, None)
        except IOError:
            # no macro file yet 
            pass

    def storeMacros(self):
        if not self.app:
            return
        with open(os.path.join(os.getenv('HOME'), ".{0}.macros".format(self.app)), 'w') as mf:
            reduce(lambda acc, n_v: acc.write("{0[0]} '{0[1]}'\n".format(n_v)) or acc, iteritems(self.macros), mf)
            mf.close()


# get the terminal/screen size
def getScreenSize():
    try:
        hw = struct.unpack('hh', fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, '1234'))
    except:
        try:
            hw = (os.environ['LINES'], os.environ['COLUMNS'])
        except:  
            hw = None
    return hw

# take a list of strings and display them or page them
def maybePage(lst):
    screenSize = getScreenSize()
    # be more intelligent about it
    doPage = None
    if screenSize:
        # count the number of lines the text would use up on the screen
        # (taking care of lines longer than the terminal width)
        doPage = sum( \
            [sum( [int(math.ceil(float(len(y) if len(y) else 1)/float(screenSize[1]))) for y in x.split('\n')] \
                ) for x in lst]) > screenSize[0]
    if doPage or not screenSize:
        pydoc.pager("\n".join(lst))
    else:
        drap(print, lst)


hlp_macro = \
"""macro [<name>[ <definition>]]
\tmacro inspection/definition

'macro' 
    Without arguments displays all currently defined macros.
'macro <name>'
    Display the definition of <name>, if any known
'macro <name> <definition>'
    Define the macro <name> to mean <definition>. To get
    embedded spaces in the definition use single quotes:
    macro <name> 'txt with spaces in'

Examples:
    macro x exit
    macro z 'help macro'
"""

hlp_play = \
"""play <file> [arg1 ... argN]:
\tread and execute the commands found in <file>

    The extra, white-space separated arguments are passed as
    arguments to the script. The arguments are passed as
    strings - no type coercion is attempted.

    Using the positional arguments in the scripts is done
    like Python string formatting; basically each line in a
    script file is taken to be a format string and the
    arguments are supplied to the ".format()" string member
    function.

    Thus, inside your script you can access the first
    argument using "{0}", the second using "{1}" etc. All
    formatting possibilities that Python offers therefore
    apply. For more details see:
    http://docs.python.org/2/library/string.html#format-string-syntax

    Example:

    $> cat mkplot
    # example script file
    #   expect parameters:
    #   <typ> <x> <y>
    plottype {0}
    nplot x {1}
    nplot y {2}

    From within the commandline interface:
    jcli> play mkplot amp 4 2


"""

hlp_shell = \
"""! <shell command>
\texecute the shell command in a subshell

Anything you type after the exlamation mark will be passed on as-is to a
subshell which means that e.g. environment variable expansion and '~' expansion
is available here. 

The command will be executed through Python's "os.system(...)" in favour of
subprocess.check() because the latter requires splitting the input and also
does not provide '~' or environment variable expansion.

"""
