#!/bin/sh

########################################################################################################################

pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./docs/ --force --html tf_som

sed '/<header>/,/<\/header>/d' ./docs/tf_som/index.html > ./docs/index.html

rm -fr ./docs/tf_som/

########################################################################################################################

if command -v html-minifier-terser &> /dev/null
then
  html-minifier-terser --collapse-whitespace --remove-comments --minify-css true --minify-js true -o ./docs/index.min.html ./docs/index.html

  mv ./docs/index.min.html ./docs/index.html
else
  echo 'npm i -g html-minifier-terser'
fi

########################################################################################################################
