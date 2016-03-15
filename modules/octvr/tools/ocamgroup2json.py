#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-03-15

import sys
import json

inputs = [{
    "type": "ocam_fisheye",
    "options": {"file": x}
} for x in sys.argv[1:]]

output = {
    "type": "perspective",
    "options": {
        "aspect_ratio": 1.333333,
        "sf": 2.0
    }
}

for i in range(len(inputs)):
    with open('defish_{}.json'.format(i), 'w') as f:
        f.write(json.dumps({"output": output, "inputs": [inputs[i], ]}, indent=4))

with open('stitch.json', 'w') as f:
    f.write(json.dumps({"output": {"type": "equirectangular"}, "inputs": inputs}, indent=4))
