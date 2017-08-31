#!/usr/bin/python
# file: matdocparser.py
# author: Andrea Vedaldi
# description: Utility to format MATLAB comments.

# Copyright (C) 2014-15 Andrea Vedaldi.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

"""
MatDocParser is an interpreter for the MatDoc format. This is a simplified and
stricter version of Markdown suitable to commenting MATLAB functions. the format
is easily understood from an example:

A paragraph starts on a new line.
And continues on following lines.

Indenting with a whitespace introduces a verbatim code section:

   Like this
    This continues it

Different paragraphs are separated by blank lines.

* The *, -, + symbols at the beginning of a line introduce a list.
  Which can be continued on follwing paragraphs by proper indentation.

  Multiple paragraphs in a list item are also supported.

* This is the second item of the same list.

It is also possible to have definition lists such as

Term1:: Short description 2
   Longer explanation.

   Behaves like a list item.

Term2:: Short description 2
Term3:: Short description 3
  Longer explanations are optional.

# Lines can begin with # to denote a title
## Is a smaller title
"""

import sys
import os
import re

__mpname__           = 'MatDocParser'
__version__          = '1.0-beta24'
__date__             = '2015-09-20'
__description__      = 'MatDoc MATLAB inline function description interpreter.'
__long_description__ = __doc__
__license__          = 'BSD'
__author__           = 'Andrea Vedaldi'

# --------------------------------------------------------------------
# Input line types (terminal symbols)
# --------------------------------------------------------------------

# Terminal symbols are organized in a hierarchy. Each line in the
# input document is mapped to leaf in this hierarchy, representing
# the type of line detected.

class Symbol(object):
    indent = None
    def isa(self, classinfo, indent = None):
        return isinstance(self, classinfo) and \
            (indent is None or self.indent == indent)
    def __str__(self, indent = 0):
        if self.indent is not None: x = "%d" % self.indent
        else: x = "*"
        return " "*indent + "%s(%s)" % (self.__class__.__name__, x)

# Terminal symbols
# Note that PL, BH, DH are all subclasses of L; the fields .text and .indent
# have the same meaning for all of them.
class Terminal(Symbol): pass
class EOF (Terminal): pass # end-of-file
class B (Terminal): pass # blank linke
class L (Terminal): # non-empty line: '<" "*indent><text>'
    text = ""
    def __str__(self, indent = 0):
        return "%s: %s" % (super(L, self).__str__(indent), self.text)
class PL (L): pass # regular line
class BH (L): # bullet: a line of type '  * <inner_text>'
    inner_indent = None
    inner_text = None
    bullet = None
class DH (L):  # description: a line of type ' <description>::<inner_text>'
    inner_text = None
    description = None
    def __str__(self, indent = 0):
        return "%s: '%s' :: '%s'" % (super(L, self).__str__(indent),
                           self.description, self.inner_text)
class SL (L): # section: '<#+><text>'
    section_level = 0
    inner_text = None
    def __str__(self, indent = 0):
        return "%s: %s" % (super(L, self).__str__(indent), self.inner_text)

# A lexer object: parse lines of the input document into terminal symbols
class Lexer(object):
    def __init__(self, lines):
        self.lines = lines
        self.pos = -1

    def next(self):
        self.pos = self.pos + 1
        # no more
        if self.pos > len(self.lines)-1:
            x = EOF()
            return x
        line = self.lines[self.pos]
        # a blank line
        match = re.match(r"\s*\n?$", line) ;
        if match:
            return B()
        # a line of type '  <#+><inner_text>'
        match = re.match(r"(\s*)(#+)(.*)\n?$", line)
        if match:
            x = SL()
            x.indent = len(match.group(1))
            x.section_level = len(match.group(2))
            x.inner_text = match.group(3)
            #print x.indent, x.section_level, x.inner_text
            return x
        # a line of type '  <content>::<inner_text>'
        match = re.match(r"(\s*)(.*)::(.*)\n?$", line)
        if match:
            x = DH()
            x.indent = len(match.group(1))
            x.description = match.group(2)
            x.inner_text = match.group(3)
            x.text = x.description + "::" + x.inner_text
            return x
        # a line of type '  * <inner_contet>'
        match = re.match(r"(\s*)([-\*+]\s*)(\S.*)\n?$", line)
        if match:
            x = BH()
            x.indent = len(match.group(1))
            x.bullet = match.group(2)
            x.inner_indent = x.indent + len(x.bullet)
            x.inner_text = match.group(3)
            x.text = x.bullet + x.inner_text
            return x
        # a line of the type  '   <content>'
        match = re.match(r"(\s*)(\S.*)\n?$", line)
        if match:
            x = PL()
            x.indent = len(match.group(1))
            x.text = match.group(2)
            return x

# --------------------------------------------------------------------
# Non-terminal
# --------------------------------------------------------------------

# DIVL is a consecutive list of blocks with the same indent and/or blank
# lines.
#
# DIVL(indent) -> (B | SL(indent) | P(indent) | V(indent) |
#                  BL(indent) | DL(indent))+
#
# S(indent) -> SL(indent)
#
# A P(indent) is a paragraph, a list of regular lines indentent by the
# same amount.
#
# P(indent) -> PL(indent)+
#
# A V(indent) is a verbatim (code) block. It contains text lines and blank
# lines that have indentation strictly larger than `indent`:
#
# V(indent) -> L(i) (B | L(j), j > indent)+, for all i > indent
#
# A DL(indent) is a description list:
#
# DL(indent) -> DH(indent) DIVL(i)*,  i > indent
#
# A BL(indent) is a bullet list. It contains bullet list items, namely
# a sequence of special DIVL_BH(indent,inner_indent) whose first block
# is a paragaraph P_BH(indent,inner_indent) whose first line is a
# bullet header BH(indent,innner_indent). Here the bullet identation
# inner_indent is obtained as the inner_indent of the
# BH(indent,inner_indent) symbol. Formalising this with grammar rules
# is verbose; instead we use the simple `hack' of defining
#
# BL(indent) -> (DIVL(inner_indent))+
#
# where DIVL(inner_indent) are regular DIVL, obtaine after replacing
# the bullet header line BH with a standard paragraph line PL.

class NonTerminal(Symbol):
    children = []
    def __init__(self, *args):
        self.children = list(args)
    def __str__(self, indent = 0):
        s = " "*indent + super(NonTerminal, self).__str__() + "\n"
        for c in self.children:
            s += c.__str__(indent + 2) + "\n"
        return s[:-1]

class S(NonTerminal): pass
class DIVL(NonTerminal): pass
class DIV(NonTerminal): pass
class BL(NonTerminal): pass
class DL(NonTerminal): pass
class DI(NonTerminal): pass
class P(DIV): pass
class V(DIV): pass

# --------------------------------------------------------------------
class Parser(object):
    lexer = None
    stack = []
    lookahead = None

    def shift(self):
        if self.lookahead:
            self.stack.append(self.lookahead)
        self.lookahead = self.lexer.next()

    def reduce(self, X, n, indent = None):
        #print "reducing %s with %d" % (S.__name__, n)
        x = X(*self.stack[-n:])
        del self.stack[-n:]
        x.indent = indent
        self.stack.append(x)
        return x

    def parse(self, lexer):
        self.lexer = lexer
        self.stack = []
        while True:
            self.lookahead = self.lexer.next()
            if not self.lookahead.isa(B): break
        self.parse_DIVL(self.lookahead.indent)
        return self.stack[0]

    def parse_SL(self, indent):
        self.shift()
        self.reduce(S, 1, indent)

    def parse_P(self, indent):
        i = 0
        if indent is None: indent = self.lookahead.indent
        while self.lookahead.isa(PL, indent):
            self.shift()
            i = i + 1
        self.reduce(P, i, indent)

    def parse_V(self, indent):
        i = 0
        while (self.lookahead.isa(L) and self.lookahead.indent > indent) or \
              (self.lookahead.isa(B)):
            self.shift()
            i = i + 1
        self.reduce(V, i, indent)

    def parse_DIV_helper(self, indent):
        if self.lookahead.isa(SL, indent):
            self.parse_SL(indent)
        elif self.lookahead.isa(PL, indent):
            self.parse_P(indent)
        elif self.lookahead.isa(L) and (self.lookahead.indent > indent):
            self.parse_V(indent)
        elif self.lookahead.isa(BH, indent):
            self.parse_BL(indent)
        elif self.lookahead.isa(DH, indent):
            self.parse_DL(indent)
        elif self.lookahead.isa(B):
            self.shift()
        else:
            return False
        # leaves with B, P(indent), V(indent), BL(indent) or DL(indent)
        return True

    def parse_BI_helper(self, indent):
        x = self.lookahead
        if not x.isa(BH, indent): return False
        indent = x.inner_indent
        self.lookahead = PL()
        self.lookahead.text = x.inner_text
        self.lookahead.indent = indent
        self.parse_DIVL(indent)
        # leaves with DIVL(inner_indent) where inner_indent was
        # obtained from the bullet header symbol
        return True

    def parse_BL(self, indent):
        i = 0
        while self.parse_BI_helper(indent): i = i + 1
        if i == 0: print "Error", sys.exit(1)
        self.reduce(BL, i, indent)

    def parse_DI_helper(self, indent):
        if not self.lookahead.isa(DH, indent): return False
        self.shift()
        if self.lookahead.indent > indent:
            self.parse_DIVL(self.lookahead.indent)
            self.reduce(DI, 2, indent)
        else:
            self.reduce(DI, 1, indent)
        return True

    def parse_DL(self, indent):
        i = 0
        while self.parse_DI_helper(indent): i = i + 1
        if i == 0: print "Error", sys.exit(1)
        self.reduce(DL, i, indent)

    def parse_DIVL(self, indent = None):
        i = 0
        while self.parse_DIV_helper(indent):
            if indent is None: indent = self.stack[-1].indent
            i = i + 1
        self.reduce(DIVL, i, indent)

if __name__ == '__main__':
    str="""

Some text describing a MATLAB function F().
The function F() does nothing.

It has the following options:

CarryOn:: True
  Keep doing nothing for the time being.

Stop:: 'here'
  Stop doing whathever here. Example:

    % call the function
    f('stop', 'there')

    % contemplate the results

So in short we conclude that:

* This does nothing
*   It could do something,
    but still does not.

   #

See also: hope for the best.

# Section number one

Bla

## More Sect
### Even more

blo
"""
    parser = Parser()
    lexer = Lexer(str.split('\n'))
    tree = parser.parse(lexer)
    print tree

