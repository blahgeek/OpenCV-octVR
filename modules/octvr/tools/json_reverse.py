#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-02-29

import sys
import json

if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    print(json.dumps({
                         "inputs": [data["output"], ],
                         "output": data["inputs"][int(sys.argv[2])]
                     }, indent=4))
