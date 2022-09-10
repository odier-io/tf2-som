#!/bin/sh

########################################################################################################################

pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./docs/ --force --html tf_som

sed '/<header>/,/<\/header>/d' ./docs/tf_som/index.html > ./docs/index.html

rm -fr ./docs/tf_som/

########################################################################################################################
