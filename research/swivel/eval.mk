# -*- Mode: Makefile -*-
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This makefile pulls down the evaluation datasets and formats them uniformly.
# Word similarity evaluations are formatted to contain exactly three columns:
# the two words being compared and the human judgement.
#
# Use wordsim.py and analogy to run the actual evaluations.

CXXFLAGS=-std=c++11 -m64 -mavx -g -Ofast -Wall
LDLIBS=-lpthread -lm

WORDSIM_EVALS=	ws353sim.ws.tab \
		ws353rel.ws.tab \
		men.ws.tab	\
		mturk.ws.tab \
		rarewords.ws.tab \
		simlex999.ws.tab \
		$(NULL)

ANALOGY_EVALS=	mikolov.an.tab \
		msr.an.tab \
		$(NULL)

all: $(WORDSIM_EVALS) $(ANALOGY_EVALS) analogy

ws353sim.ws.tab: ws353simrel.tar.gz
	tar Oxfz $^ wordsim353_sim_rel/wordsim_similarity_goldstandard.txt > $@

ws353rel.ws.tab: ws353simrel.tar.gz
	tar Oxfz $^ wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt > $@

men.ws.tab: MEN.tar.gz
	tar Oxfz $^ MEN/MEN_dataset_natural_form_full | tr ' ' '\t' > $@

mturk.ws.tab: Mtruk.csv
	cat $^ | tr -d '\r' | tr ',' '\t' > $@

rarewords.ws.tab: rw.zip
	unzip -p $^ rw/rw.txt | cut -f1-3 -d $$'\t' > $@

simlex999.ws.tab: SimLex-999.zip
	unzip -p $^ SimLex-999/SimLex-999.txt \
	| tail -n +2 | cut -f1,2,4 -d $$'\t' > $@

mikolov.an.tab: questions-words.txt
	egrep -v -E '^:' $^ | tr '[A-Z] ' '[a-z]\t' > $@

msr.an.tab: myz_naacl13_test_set.tgz
	tar Oxfz $^ test_set/word_relationship.questions | tr ' ' '\t' > /tmp/q
	tar Oxfz $^ test_set/word_relationship.answers | cut -f2 -d ' ' > /tmp/a
	paste /tmp/q /tmp/a > $@
	rm -f /tmp/q /tmp/a


# wget commands to fetch the datasets.  Please see the original datasets for
# appropriate references if you use these.
ws353simrel.tar.gz:
	wget http://alfonseca.org/pubs/ws353simrel.tar.gz

MEN.tar.gz:
	wget http://clic.cimec.unitn.it/~elia.bruni/resources/MEN.tar.gz

Mtruk.csv:
	wget http://tx.technion.ac.il/~kirar/files/Mtruk.csv

rw.zip:
	wget http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip

SimLex-999.zip:
	wget http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip

questions-words.txt:
	wget http://word2vec.googlecode.com/svn/trunk/questions-words.txt

myz_naacl13_test_set.tgz:
	wget http://research.microsoft.com/en-us/um/people/gzweig/Pubs/myz_naacl13_test_set.tgz

analogy: analogy.cc

clean:
	rm -f *.ws.tab *.an.tab analogy *.pyc

distclean: clean
	rm -f *.tgz *.tar.gz *.zip Mtruk.csv questions-words.txt
