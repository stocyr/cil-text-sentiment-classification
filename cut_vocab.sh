#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
if command -v pv >/dev/null 2>&1 ;then
    # pv exists
	pv -l scratch/vocab.txt | sed "s/^\s\+//g" | sed "s/\s\+/ /g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > scratch/vocab_cut.txt
else
    # work without pv
    cat scratch/vocab.txt | sed "s/^\s\+//g" | sed "s/\s\+/ /g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > scratch/vocab_cut.txt
fi
