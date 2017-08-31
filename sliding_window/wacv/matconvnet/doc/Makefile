PYTHON = python
MARKDOWN = markdown2
PDFLATEX = pdflatex
BIBTEX = bibtex
MKDOCS = mkdocs
SVG2PDF = svg2pdf
NPM = npm
MATLAB = $(MATLABROOT)/bin/matlab
DOT = dot

site=doc/site/docs

mfiles=\
vl_argparse.m \
vl_compilenn.m \
vl_imreadjpeg.m \
vl_nnbnorm.m \
vl_nnbilinearsampler.m \
vl_nnconcat.m \
vl_nnconv.m \
vl_nnconvt.m \
vl_nncrop.m \
vl_nndropout.m \
vl_nnloss.m \
vl_nnnoffset.m \
vl_nnnormalize.m \
vl_nnnormalizelp.m \
vl_nnpdist.m \
vl_nnpool.m \
vl_nnrelu.m \
vl_nnroipool.m \
vl_nnsigmoid.m \
vl_nnsoftmax.m \
vl_nnsoftmaxloss.m \
vl_nnspnorm.m \
vl_rootnn.m \
vl_setupnn.m \
vl_imreadjpeg.m \
vl_imreadjpeg.m \
vl_taccum.m \
vl_tmove.m \
vl_tshow.m \
simplenn/vl_simplenn.m \
simplenn/vl_simplenn_diagnose.m \
simplenn/vl_simplenn_tidy.m \
simplenn/vl_simplenn_display.m \
simplenn/vl_simplenn_move.m \
+dagnn/@DagNN/DagNN.m

mfile_src=$(addprefix matlab/src/, $(mfiles))
mfile_doc=$(patsubst %.m,%.md,$(addprefix $(site)/mfiles/, $(mfiles)))
mds=$(wildcard doc/site/docs/*.md)
svg_src=$(wildcard doc/figures/svg/*.svg)
svg_tgt=$(patsubst %.svg, %.pdf, $(svg_src))

models=$(wildcard data/models/*.mat)
model_svg_tgt=$(patsubst %.mat, %.svg, $(models))

# Web components
node_modules = doc/site/.build/node_modules
node_modules-dir = $(node_modules)/.stamp
jquery=$(node_modules)/jquery/dist/jquery.min.js
fontawesome=$(node_modules)/font-awesome/css/font-awesome.min.css
bootstrap=$(node_modules)/bootstrap/dist/css/bootstrap.min.css

doc: doc-site doc-manual
doc-site: doc/site/site/index.html
doc-manual: doc/matconvnet-manual.pdf
doc-arxiv: doc/matconvnet-manual.tar.gz

doc/matconvnet-manual.pdf : doc/matconvnet-manual.tex doc/*.tex doc/*.bib $(svg_tgt)
	mkdir -p doc/.build
	ln -sf ../references.bib doc/.build/references.bib
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cd doc/.build ; $(BIBTEX) matconvnet-manual || true ;
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cp -f doc/.build/matconvnet-manual.pdf doc/

doc/matconvnet-manual.tar.gz : doc/*.tex doc/figures/*.pdf doc/figures/svg/*.pdf
	cd doc ; cp .build/matconvnet-manual.bbl .;\
	tar czvf matconvnet-manual.tar.gz $(^:doc/%=%) matconvnet-manual.bbl
	rm doc/matconvnet-manual.bbl

doc/figures/svg/%.pdf : doc/figures/svg/%.svg
	$(SVG2PDF) "$(<)" "$(@)"

$(site)/mfiles/%.md : matlab/%.m $(site)/mfiles/.stamp doc/matdoc.py doc/matdocparser.py
	mkdir -p $(dir $(@))
	$(PYTHON) doc/matdoc.py "$(<)" > "$(@)"

$(node_modules-dir):
	mkdir -p $(node_modules)
	touch -t 197301010000 $(node_modules-dir)

$(bootstrap): $(node_modules-dir)
	cd $(node_modules) ; $(NPM) install bootstrap@3.3.6

$(fontawesome): $(node_modules-dir)
	cd $(node_modules) ; $(NPM) install font-awesome@4.5.0

$(jquery): $(node_modules-dir)
	cd $(node_modules) ; $(NPM) install jquery@2.1.4

doc/site/site/index.html : doc/site/mkdocs.yml \
  $(mfile_doc) $(mds) \
  $(bootstrap) $(fontawesome) $(jquery)
	mkdir -p doc/site/theme/{css,js,fonts}
	cd doc/site/theme/css ; ln -sfv ../../.build/node_modules/bootstrap/dist/css/bootstrap.min.css{,.map} ./
	cd doc/site/theme/css ; ln -sfv ../../.build/node_modules/font-awesome/css/font-awesome.min.css ./
	cd doc/site/theme/fonts ; ln -sfv ../../.build/node_modules/font-awesome/fonts/fontawesome-webfont.{eot,svg,ttf,woff,woff2} ./
	cd doc/site/theme/js ; ln -sfv ../../.build/node_modules/bootstrap/dist/js/{bootstrap.min.js,npm.js} ./
	cd doc/site/theme/js ; ln -sfv ../../.build/node_modules/jquery/dist/jquery.min.{js,map} ./
	cd doc/site ; $(MKDOCS) build --clean

doc-clean:
	rm -f doc/matdocparser.pyc doc/matdoc.pyc
	rm -f $(svg_tgt)

doc-distclean:
	rm -f doc/matconvnet-manual.pdf
	rm -rf doc/site/site
	rm -rf doc/site/.build
	rm -f $(mfile_doc)

doc-info:
	@echo "mds=$(mds)"
	@echo "mfile_src=$(mfile_src)"
	@echo "mfile_doc=$(mfile_doc)"
	@echo "svg_tgt=$(svg_tgt)"

doc-serve: doc
	cd doc/site ; $(MKDOCS) serve

data/models/%.dot : data/models/%.mat
	$(MATLAB) -nosplash -nodesktop -r "addpath('utils'); model2dot('$(<)', '$(@)'); exit;"

data/models/%.svg : data/models/%.dot
	$(DOT) -Tsvg -o "$(@)" "$(<)"

model-svg : $(model_svg_tgt)
