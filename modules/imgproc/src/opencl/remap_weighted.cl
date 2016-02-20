// by blahgeek

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

enum
{
    INTER_BITS = 5,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

#define OUTSIDE(gx, gy) (gx >= src_cols || gy >= src_rows || gx < 0 || gy < 0)

__kernel void remap_weighted(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                             __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                             __global const uchar * map1ptr, int map1_step, int map1_offset,
                             __global const uchar * map2ptr, int map2_step, int map2_offset,
                             __global const uchar * weight_map_ptr, int weight_map_step, int weight_map_offset
                             ) {
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols) {
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(DST_T), dst_offset));
        int map1_index = mad24(y, map1_step, mad24(x, (int)sizeof(short2), map1_offset));
        int map2_index = mad24(y, map2_step, mad24(x, (int)sizeof(ushort), map2_offset));
        int weight_map_index = mad24(y, weight_map_step, mad24(x, (int)sizeof(uchar), weight_map_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map1_index += map1_step, map2_index += map2_step, 
            dst_index += dst_step, weight_map_index += weight_map_step)
            if (y < dst_rows)
            {
                __global const short2 * map1 = (__global const short2 *)(map1ptr + map1_index);
                __global const ushort * map2 = (__global const ushort *)(map2ptr + map2_index);
                __global const uchar * weight_map = (__global const uchar *)(weight_map_ptr + weight_map_index);
                __global DST_T * dst = (__global DST_T *)(dstptr + dst_index);

                uchar weight = weight_map[0];

                int2 map_dataA = convert_int2(map1[0]);
                int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
                int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
                int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);

                ushort map2Value = (ushort)(map2[0] & (INTER_TAB_SIZE2 - 1));
                WT2 u = (WT2)(map2Value & (INTER_TAB_SIZE - 1), map2Value >> INTER_BITS) / (WT2)(INTER_TAB_SIZE);

                WT a = 0, b = 0, c = 0, d = 0;

                #define loadpix(addr) (*(__global const SRC_T *)(addr))

                if (!OUTSIDE(map_dataA.x, map_dataA.y))
                    a = convertToWT(loadpix(srcptr + mad24(map_dataA.y, src_step, map_dataA.x * (int)sizeof(SRC_T) + src_offset)));
                if (!OUTSIDE(map_dataB.x, map_dataB.y))
                    b = convertToWT(loadpix(srcptr + mad24(map_dataB.y, src_step, map_dataB.x * (int)sizeof(SRC_T) + src_offset)));
                if (!OUTSIDE(map_dataC.x, map_dataC.y))
                    c = convertToWT(loadpix(srcptr + mad24(map_dataC.y, src_step, map_dataC.x * (int)sizeof(SRC_T) + src_offset)));
                if (!OUTSIDE(map_dataD.x, map_dataD.y))
                    d = convertToWT(loadpix(srcptr + mad24(map_dataD.y, src_step, map_dataD.x * (int)sizeof(SRC_T) + src_offset)));

                WT dst_data = a * (1 - u.x) * (1 - u.y) +
                              b * (u.x)     * (1 - u.y) +
                              c * (1 - u.x) * (u.y) +
                              d * (u.x)     * (u.y);
                dst_data *= convertToWT(weight);

                dst[0] += convertToDstT(dst_data);
            }
    }
}
