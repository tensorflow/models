#!/usr/bin/env python
#-*- coding: utf8 -*-

import os
import sys
from   optparse import OptionParser
import time

# --verbose
VERBOSE = 0

if __name__ == '__main__':

	parser = OptionParser()
	parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
	(options, args) = parser.parse_args()
	
	if options.verbose == 1 : VERBOSE = 1

	startTime = time.time()

	data = {}

	while 1 :
		try : line = sys.stdin.readline()
		except KeyboardInterrupt : break
		if not line : break

		line = line.strip()
		if not line :
			print ''
			continue
		if line[0] == '#' :
			print line
			continue

		tokens = line.split('\t')
		if tokens[4] == '_' :
			tokens[4] = tokens[3] # there is no XPOS
		else :
			tokens[3] = tokens[4] # UPOS <- XPOS
		print '\t'.join(tokens)

	durationTime = time.time() - startTime
	sys.stderr.write("duration time = %f\n" % durationTime)
