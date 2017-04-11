#!/usr/bin/env python3
from scipy.sparse import *
import pickle
import sys

usage = 'Usage: python3 pickle_version_converter.py [input file] [output protocol version]\n' \
        '  input file                pickle object file (ending with .pkl)\n' \
        '  output protocol version   desired pickle protocol version of output file: 0..4' \

if len(sys.argv) != 3:
    print('Not given 2 arguments.\n' + usage)
    sys.exit(1)
if int(sys.argv[2]) not in range(5):
    print('Please specify a valid pickle protocol version: 0..4\n' + usage)
    sys.exit(1)

old_name = str(sys.argv[1])
new_name = old_name[:-4] + '_new.pkl'
protocol = int(sys.argv[2])

print('Loading object from pickle file', old_name)
with open(sys.argv[1], 'rb') as f:
    obj = pickle.load(f)

print('Dumping to new pickle file', new_name, 'in pickle protocol', protocol)
with open(new_name, 'wb') as f:
    pickle.dump(obj, f, protocol)

print("Conversion done.")
