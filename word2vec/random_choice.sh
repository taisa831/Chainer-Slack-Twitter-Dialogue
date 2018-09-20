#!/bin/bash
# ------------------------------------------------------------------
# [Masaya Ogushi] Choose Random in the shell file
#
#          library for Unix shell scripts.
#            Shell template
#               http://stackoverflow.com/questions/14008125/shell-script-common-template
#
# ------------------------------------------------------------------
# --- Option processing --------------------------------------------
if [ $# == 0 ] ; then
    echo $USAGE
    exit 1;
fi

WIKI_DATA=$1
GET_NUMBER=5000

# -- Body ---------------------------------------------------------

cat $WIKI_DATA | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' | head -n $GET_NUMBER
jot -r "$(wc -l $WIKI_DATA)" 1 | paste - $1 | sort -n | cut -f 2- | head -n $GET_NUMBER
