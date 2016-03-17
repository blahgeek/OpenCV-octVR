#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-03-15

import os
import sys
import json

inputs = [{
    "type": "ocam_fisheye",
    "options": {"file": os.path.abspath(x)}
} for x in sys.argv[1:]]

output = {
    "type": "perspective",
    "options": {
        "aspect_ratio": 1.6,
        "sf": 2.0
    }
}

for i in range(len(inputs)):
    with open('defish_{}.json'.format(i + 1), 'w') as f:
        json.dump({"output": output, "inputs": [inputs[i], ]}, f, indent=4)

# with open('stitch.json', 'w') as f:
#     json.dump({"output": {"type": "equirectangular"}, "inputs": inputs}, f, indent=4)
