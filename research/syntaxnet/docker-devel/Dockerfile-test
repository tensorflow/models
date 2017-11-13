FROM dragnn-oss-test-base:latest

RUN rm -rf \
  $SYNTAXNETDIR/syntaxnet/dragnn \
  $SYNTAXNETDIR/syntaxnet/syntaxnet \
  $SYNTAXNETDIR/syntaxnet/third_party \
  $SYNTAXNETDIR/syntaxnet/util/utf8
COPY dragnn $SYNTAXNETDIR/syntaxnet/dragnn
COPY syntaxnet $SYNTAXNETDIR/syntaxnet/syntaxnet
COPY third_party $SYNTAXNETDIR/syntaxnet/third_party
COPY util/utf8 $SYNTAXNETDIR/syntaxnet/util/utf8
