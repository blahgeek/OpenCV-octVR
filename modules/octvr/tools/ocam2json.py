#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-03-10

import json
import sys

opts = {}

lines = iter(open(sys.argv[1]))
lines = map(lambda x: x.strip(), lines)
lines = filter(lambda line: line and not line.startswith('#'), lines)
lines = list(lines)

S = lambda n: [float(x) for x in lines[n].split(' ')]

opts['pol'] = S(0)[1:]
opts['inv_pol'] = S(1)[1:]
opts['xc'], opts['yc'] = S(2)
opts['c'], opts['d'], opts['e'] = S(3)
opts['height'], opts['width'] = [int(x) for x in lines[4].split(' ')]

print(json.dumps({
        "output": {
            "type": "equirectangular",
            "options": {
                "rotation": {
                    "roll": 0,
                    "yaw": 0,
                    "pitch": 0,
                }
            }
        }, 
        "inputs": [{
            "type": "ocam_fisheye",
            "options": opts
        }],
      }, indent=4))
