#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by i@BlahGeek.com at 2016-01-16

import logging
import sys
import re
import json

# Support .pts(PTGui project file) and .pto(Hugin project file)
class PTXParser:

    def __init__(self):
        self.inputs = []
        self.processing_input = dict()

    def process_input_line(self, line):
        fields = line.strip().split(' ')
        for field in fields:
            match = re.match(r'([a-zA-Z]+)(.+)', field)
            if match is None:
                continue
            key, val = match.groups()
            if val[0] == '=':
                val = self.inputs[int(val[1:])][key]
            self.processing_input[key] = val

        self.inputs.append(self.processing_input)
        self.processing_input = dict()
        logging.info("New input added")

    def process_mask_line(self, line):
        match = re.match(r'k i(\d+) t(\d+) p"(.*)"', line.strip())
        if match is None:
            return
        assert match.group(2) == '0', "Currently only support negative hugin mask"
        img = self.inputs[int(match.group(1))]
        if 'exclude_masks' not in img:
            img['exclude_masks'] = list()
        img['exclude_masks'].append({
            'type': 'polygonal',
            'args': list(map(float, match.group(3).split(' ')))
        })

    def process_input_meta_dummyimage(self, args):
        self.processing_input['dummyimage'] = True

    def process_input_meta_imgfile(self, args):
        fields = args.strip().split(' ')
        self.processing_input['w'] = int(fields[0])
        self.processing_input['h'] = int(fields[1])

    def process_input_meta_imgcrop(self, args):
        fields = list(map(float, args.strip().split(' ')))
        for x in fields[:5]:
            assert x == 0, 'Only circle crop is supported now'
        assert len(fields) == 8
        self.processing_input["circular_crop"] = fields[5:]

    def process_input_meta_viewpoint(self, args):
        for viewpoint in map(float, args.strip().split(' ')):
            assert viewpoint == 0, 'Viewpoint is not supported yet'

    def process_input_meta(self, line):
        '''Process line startswith '#-' '''
        cmd, _, args = line.strip().partition(' ')

        method = getattr(self, 'process_input_meta_' + cmd, None)
        if callable(method):
            method(args)
        else:
            self.processing_input['meta-' + cmd] = args

    def process_line(self, line):
        if line.startswith('#-'):
            self.process_input_meta(line[2:])
        elif line.startswith('#'):
            pass
        elif line.startswith('o') or line.startswith('i'):
            self.process_input_line(line)
        elif line.startswith('k'):
            self.process_mask_line(line)

    def dump(self):
        deg_to_rad = lambda s: float(s) / 180.0 * 3.1415926
        for img in self.inputs:
            if 'dummyimage' in img:
                continue
            assert img['f'] == '3' or img['f'] == '2', "Currently only support fisheye"
            options = {
                "rotation": {
                    "roll": deg_to_rad(img["r"]),
                    "yaw": -deg_to_rad(img["y"]),
                    "pitch": -deg_to_rad(img["p"]),
                },
                "width": int(img["w"]),
                "height": int(img["h"]),
                "hfov": deg_to_rad(img["v"]) / (img.get('fov_ratio', 0.5) * 2),
                "center_dx": float(img["d"]),
                "center_dy": float(img["e"]),
                "radial": [float(img["a"]), float(img["b"]), float(img["c"])],
            }
            for key in ('circular_crop', 'exclude_masks', ):
                if key in img:
                    options[key] = img[key]
            yield {
                "type": "fullframe_fisheye",
                "options": options
            }

if __name__ == '__main__':
    parser = PTXParser()
    with open(sys.argv[1]) as f:
        for line in f:
            parser.process_line(line)
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
            "inputs": list(parser.dump()),
          }, indent=4))
