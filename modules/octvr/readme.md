## libmap

### `Camera`类

定义了不同类型的“Camera”，比如针孔相机(pinhole)、鱼眼相机(fisheye)、2:1投影(equirectangular)，用来表示一个图像和球面的对应关系。每个Camera可以支持以下两个操作（或者其中一个）：

- `obj_to_image`：把单位圆上的点投影到图像平面上
- `image_to_bf`：把图像平面的点投影到单位圆上

单位圆上的坐标用经纬度表示，图像上的点用xy表示，xy均在0到1之间。

### `libmap.h`

提供了从一个Camera到另一个Camera的变换操作。

## FFmpeg集成

### How-to build

``` bash
# 在octVR下
./configure --prefix=/home/xxx/xxx
make install
# 在ffmpeg/下(ocrVR分支)
git checkout octVR
./configure --extra-cflags="-I/home/xxx/xxx/include" --extra-libs="/home/xxx/xxx/lib/libmap.a"
make
```

### How-to run

运行的时候接受一个json文件作为输入输出模型的参数，比如有`map.json`：

``` json
{
    "output": {
        "type": "equirectangular",
        "options": {}
    },
    "inputs": [
        {
            "options": {
                "rotate": [0, 0, 0.15],
                "cx": 966.004663835948,
                "cy": 552.562410967409,
                "dist_coeffs": [
                    -0.320414285265493,
                    0.104694493578686,
                    -0.000104514827602454,
                    0.000658872153318541,
                ],
                "fx": 1095.80624712818,
                "fy": 1095.39665386569,
                "height": 1080,
                "width": 1920
            },
            "type": "pinhole"
        },
         {
            "options": {
                "rotate": [3.14, 0, 0.15],
                "cx": 966.004663835948,
                "cy": 552.562410967409,
                "dist_coeffs": [
                    -0.320414285265493,
                    0.104694493578686,
                    -0.000104514827602454,
                    0.000658872153318541,
                ],
                "fx": 1095.80624712818,
                "fy": 1095.39665386569,
                "height": 1080,
                "width": 1920
            },
            "type": "pinhole"
        },
    ]
}
```

其中`inputs`和`output`分别为输入输出模型的参数，inputs可以有多个。

运行`./ffmpeg -i ../octVR/data/1.jpg -i ../octVR/data/2.jpg -filter_complex '[0][1]vr_map=inputs=2:options=map.json:out_width=3840' -y output%d.bmp`即可输出`output1.bmp`。