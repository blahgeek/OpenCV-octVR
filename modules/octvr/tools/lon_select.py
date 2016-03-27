#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-03-22

import sys
import json
import math

deg_to_rad = lambda s: float(s) / 180.0 * math.pi

if len(sys.argv) < 5:
    print('Usage: {} <input.json> <min-longitude-of-1st-img> <lon-range-per-img> <lon-offset-from-2-to-1>'.format(sys.argv[0]))
    sys.exit(1)

data = json.load(open(sys.argv[1]))

min_lon = deg_to_rad(sys.argv[2])
range_lon = deg_to_rad(sys.argv[3])
offset_lon = deg_to_rad(sys.argv[4])

for x in data['inputs'][:16]:
    x['options']['longitude_selection'] = (min_lon, min_lon + range_lon)
    min_lon += offset_lon

print(json.dumps(data, indent=4))
