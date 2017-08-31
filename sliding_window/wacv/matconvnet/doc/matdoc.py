# file: matdoc.py
# author: Andrea Vedaldi
# brief: Extact comments from a MATLAB mfile and generate a Markdown file

import sys, os, re, shutil
import subprocess, signal
import string, fnmatch

from matdocparser import *
from optparse import OptionParser

usage = """usage: %prog [options] <mfile>

Extracts the comments from the specified <mfile> and prints a Markdown
version of them."""

optparser = OptionParser(usage=usage)
optparser.add_option(
    "-v", "--verbose",
    dest    = "verb",
    default = False,
    action  = "store_true",
    help    = "print debug information")

findFunction = re.compile(r"^\s*(function|classdef).*$", re.MULTILINE)
getFunction = re.compile(r"\s*%\s*(\w+)\s*(.*)\n"
                          "((\s*%.*\n)+)")
cleanComments = re.compile("^\s*%", re.MULTILINE)

# --------------------------------------------------------------------
def readText(path):
# --------------------------------------------------------------------
    with open (path, "r") as myfile:
        text=myfile.read()
    return text

# --------------------------------------------------------------------
class MatlabFunction:
# --------------------------------------------------------------------
    def __init__(self, name, nature, brief, body):
        self.name = name
        self.nature = nature
        self.brief = brief
        self.body = body

    def __str__(self):
        return "%s (%s)" % (self.name, self.nature)

# --------------------------------------------------------------------
def findNextFunction(test, pos):
# --------------------------------------------------------------------
    if pos == 0 and test[0] == '%':
        # This is an M-file with a MEX implementation
        return (pos, 'function')
    m = findFunction.search(test, pos)
    if m:
        return (m.end()+1, m.group(1))
    else:
        return (None, None)

# --------------------------------------------------------------------
def getFunctionDoc(text, nature, pos):
# --------------------------------------------------------------------
    m = getFunction.match(text, pos)
    if m:
        name = m.group(1)
        brief = m.group(2).strip()
        body = clean(m.group(3))
        return (MatlabFunction(name, nature, brief, body), m.end()+1)
    else:
        return (None, pos)

# --------------------------------------------------------------------
def clean(text):
# --------------------------------------------------------------------
    return cleanComments.sub("", text)

# --------------------------------------------------------------------
def extract(text):
# --------------------------------------------------------------------
    funcs = []
    pos = 0
    while True:
        (pos, nature) = findNextFunction(text, pos)
        if nature is None: break
        (f, pos) = getFunctionDoc(text, nature, pos)
        if f:
            funcs.append(f)
    return funcs

# --------------------------------------------------------------------
class Frame(object):
# --------------------------------------------------------------------
    prefix = ""
    before = None
    def __init__(self, prefix, before = None, hlevel = 0):
        self.prefix = prefix
        self.before = before
        self.hlevel = hlevel

# --------------------------------------------------------------------
class Context(object):
# --------------------------------------------------------------------
    frames = []

    def __init__(self, hlevel = 0):
        self.hlevel = hlevel

    def __str__(self):
        text =  ""
        for f in self.frames:
            if not f.before:
                text = text + f.prefix
            else:
                text = text + f.prefix[:-len(f.before)] + f.before
                f.before = None
        return text

    def pop(self):
        f = self.frames[-1]
        del self.frames[-1]
        return f

    def push(self, frame):
        self.frames.append(frame)

def render_L(tree, context):
    print "%s%s" % (context,tree.text)

def render_L_from_indent(tree, context, indent):
    print "%s%s%s" % (context," "*max(0,tree.indent-indent),tree.text)

def render_SL(tree, context):
    print "%s%s %s" % (context,
                      "#"*(context.hlevel+tree.section_level),
                       tree.inner_text)

def render_S(tree, context):
    for n in tree.children: render_SL(n, context)

def render_DH(tree, context):
    if len(tree.inner_text.strip()) > 0:
        print "%s**%s** [*%s*]" % (context, tree.description.strip(), tree.inner_text.strip())
    else:
        print "%s**%s**" % (context, tree.description.strip())

def render_DI(tree, context):
    context.push(Frame("    ", "*   "))
    render_DH(tree.children[0], context)
    print context
    if len(tree.children) > 1:
        render_DIVL(tree.children[1], context)
    context.pop()

def render_DL(tree, context):
    for n in tree.children: render_DI(n, context)

def render_P(tree, context):
    for n in tree.children: render_L(n, context)
    print context

def render_B(tree, context):
    print context

def render_V(tree, context):
    context.push(Frame("    "))
    for n in tree.children:
        if n.isa(L): render_L_from_indent(n, context, tree.indent)
        elif n.isa(B): render_B(n, context)
    context.pop()

def render_BL(tree, context):
    for n in tree.children:
        context.push(Frame("    ", "+   "))
        render_DIVL(n, context)
        context.pop()

def render_DIVL(tree, context):
    for n in tree.children:
        if n.isa(P): render_P(n, context)
        elif n.isa(BL): render_BL(n, context)
        elif n.isa(DL): render_DL(n, context)
        elif n.isa(V): render_V(n, context)
        elif n.isa(S): render_S(n, context)
        context.before = ""

def render(func, brief, tree, hlevel):
    print "%s `%s` - %s" % ('#' * hlevel, func.upper(), brief)
    render_DIVL(tree, Context(hlevel))

if __name__ == '__main__':
    (opts, args) = optparser.parse_args()
    if len(args) != 1:
        optparser.print_help()
        sys.exit(2)
    mfilePath = args[0]

    # Get the function
    text = readText(mfilePath)
    funcs = extract(text)
    if len(funcs) == 0:
        print >> sys.stderr, "Could not find a MATLAB function"
        sys.exit(-1)

    parser = Parser()
    if funcs[0].nature == 'classdef':
        # For MATLAB classes, look for other methods outside
        # the classdef file
        components = mfilePath.split(os.sep)
        if len(components)>1 and components[-2][0] == '@':
            classDir = string.join(components[:-1],os.sep)
            for x in os.listdir(classDir):
                if fnmatch.fnmatch(x, '*.m') and not x == components[-1]:
                    text = readText(classDir + os.sep + x)
                    funcs_ = extract(text)
                    if len(funcs_) > 0:
                        funcs.append(funcs_[0])
    else:
        # For MATLAB functions, do not print subfuctions
        funcs = [funcs[0]]

    hlevel = 1
    for f in funcs:
        lexer = Lexer(f.body.splitlines())
        tree = parser.parse(lexer)
        if opts.verb:
            print >> sys.stderr, tree
        render(f.name, f.brief, tree, hlevel)
        hlevel = 2
