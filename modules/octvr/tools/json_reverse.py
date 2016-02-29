#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-02-29

import sys
import json

if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    for i in range(len(data['inputs'])):
        new_data = {
            'inputs': [data['output'], ],
            'output': data['inputs'][i]
        }
        json.dump(new_data, open(sys.argv[1] + '_{}.json'.format(i), 'w'), indent=4)
