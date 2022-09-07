#!/bin/bash

rm -fr ./dist/ ./build/ ./tf2_som.egg-info/

./setup.py install

if twine check dist/*
then
    twine upload dist/*
fi
