#include "lut.h"


// ======================================================================== //
//                               MuLUT
// ======================================================================== //

void MuLUT_init(MuLUT *mdl, char* dir, char* modes, int nmode, int nstage, int bitwidth, int interval, int scale) {
    if(!mdl || !dir || !modes) {
        return;
    }

    char lut_path[54];
    char mode;
    int stage, midx, lidx;

    mdl->dir = strdup(dir);
    mdl->modes = strdup(modes);
    mdl->nmode = nmode;
    mdl->nstage = nstage;
    mdl->bitwidth = bitwidth;
    mdl->interval = interval;
    mdl->scale = scale;
    mdl->free_mid = 1;

    for(stage = 1; stage <= nstage; stage++) {
        for(midx = 0; midx < nmode; midx++) {
            mode = modes[midx];
            lidx = (stage - 1) * nmode + midx;
            sprintf(lut_path, "%s/x%d_%db_i%d_s%d_%c.bin", dir, scale, interval, bitwidth, stage, mode);
            mdl->lut[lidx] = TensorData_load(lut_path);
            printf("Load lut[%d]<%s>(%d, %d): %s\n", lidx, TensorType_str(mdl->lut[lidx]->dtype), mdl->lut[lidx]->shape[0], mdl->lut[lidx]->shape[1], lut_path);
        }
    }

    for(int i = nstage * nmode; i < LUT_PTR_MAX; i++) {
        mdl->lut[i] = NULL;
    }
}

void MuLUT_free(MuLUT *mdl) {
    if(!mdl) {
        return;
    }

    if(mdl->dir) {
        free(mdl->dir);
        mdl->dir = NULL;
    }
    if(mdl->modes) {
        free(mdl->modes);
        mdl->modes = NULL;
    }

    for(int i = 0; i < LUT_PTR_MAX; i++) {
        TensorData_free(mdl->lut[i]);
        mdl->lut[i] = NULL;
        printf("Free lut[%d]\n", i);
    }
}

void MuLUT_forward(MuLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C) {
    if(!mdl || !O || !I) {
        return;
    }
    
    int H2 = H * mdl->scale;
    int W2 = W * mdl->scale;

    char mode;
    int stage, midx, pad, rot, norm, bias, scale;
    int HP, WP;
    int32_t *trg, **mid = mdl->mid;

    // 1. Alloc middle tensor
    mid[0] = DAlc_s32( H *  W * C);         // in   [ H,  W, C]  --- stage 1
    mid[1] = DAlc_s32( H *  W * C);         // rot  [ H,  W, C]
    mid[3] = DAlc_s32( H *  W * C);         // intp [ H,  W, C]  (scale = 1)
    mid[4] = DAlc_s32( H *  W * C);         // out  [ H,  W, C]  --- stage 2
    mid[5] = DAlc_s32(H2 * W2 * C);         // intp [H2, W2, C]  (scale = scale)
    mid[6] = DAlc_s32(H2 * W2 * C);         // rot  [H2, W2, C]
    mid[7] = DAlc_s32(H2 * W2 * C);         // out  [H2, W2, C]

    // 2. Forward
    Zero_s32(mid[7], H2 * W2 * C);
    for(stage = 1; stage <= mdl->nstage; stage++) {
        // 2.1 stage head
        if(stage != mdl->nstage) {
            norm = mdl->nmode * 4;
            bias = 0;
            scale = 1;
            trg = mid[3];
        } else {
            norm = mdl->nmode;
            bias = 0;
            scale = mdl->scale;
            trg = mid[5];
        }
        if(stage == 1) {
            Cast_s32_u8(mid[0], I, H * W * C);
        } else {
            Copy_s32(mid[0], mid[4], H * W * C);
        }
        Zero_s32(mid[4], H * W * C);
        // 2.2 stage body
        for(midx = 0; midx < mdl->nmode; midx++) {
            mode = mdl->modes[midx];
            pad = Lut_pad_dict(mode);
            HP = H + pad;
            WP = W + pad;
            mid[2] = DAlc_s32(HP * WP * C);     // pad  [HP, WP, C]
            for(rot = 0; rot < 4; rot++) {
                Rot_s32_hwc(mid[1], mid[0], C, H, W, rot);
                Pad_s32_hwc(mid[2], mid[1], C, H, W, pad, 1, 0);
                Intp_simplex4d_s32_hwc(
                    trg,
                    mid[2],
                    mdl->lut[(stage - 1) * mdl->nmode + midx]->data,
                    C,
                    H + pad,
                    W + pad,
                    mdl->bitwidth,
                    mdl->interval,
                    scale,
                    norm,
                    bias,
                    mode
                );
                if(stage != mdl->nstage) {  // --- stage 1
                    Rot_s32_hwc(mid[1], trg, C, H, W, (4 - rot));
                    Add_s32(mid[4], mid[4], mid[1], H * W * C);
                    printf("stage %d, mode %c\n", stage, mode);
                    Disp_u32_chw(mid[4], H, W, C);
                } else {                    // --- stage 2
                    Rot_s32_hwc(mid[6], trg, C, H2, W2, (4 - rot));
                    Add_s32(mid[7], mid[7], mid[6], H2 * W2 * C);
                }
            }
            DRlz_s32(mid[2]);
        }
        // 2.3 stage tail
        if(stage != mdl->nstage) {  // --- stage 1
            Addi_s32(mid[4], mid[4], 127, H * W * C);
            Clamp_s32(mid[4], mid[4], H * W * C, 8, 0);
        } else {                    // --- stage 2
            Clamp_s32(mid[7], mid[7], H2 * W2 * C, 8, 0);
        }
    }
    // 3. Output
    Cast_u8_s32(O, mid[7], H2 * W2 * C);
    // 4. Free
    if(mdl->free_mid) {
        for(int i = 0; i < 8; i++) {
            DRlz_s32(mid[i]);
        }
    }
}


// ======================================================================== //
//                               TinyLUT
// ======================================================================== //

void TinyLUT_init(TinyLUT *mdl, char* dir, char* modes, int nmode, int nstage, int bitwidth, int interval, int scale) {
    if(!mdl || !dir || !modes) {
        return;
    }

    char lut_path[54];
    char mode;
    int stage, lidx;

    mdl->dir = strdup(dir);
    mdl->modes = strdup(modes);
    mdl->nmode = nmode;
    mdl->nstage = nstage;
    mdl->bitwidth = bitwidth;
    mdl->interval = interval;
    mdl->scale = scale;
#ifdef LUT_MEMPOOL
    mdl->free_mid = 0;  // if you use MemPool, we sugguest you set this to 0
#else
    mdl->free_mid = 1; // if you don't use MemPool, we sugguest you set this to 1
#endif // LUT_MEMPOOL
    
#ifdef LUT_PERF
    mdl->perf_time = 0;
    mdl->perf_cnt = 0;
    mdl->perf_fps = 0;
    mdl->perf_mem = 0;
#endif // LUT_PERF

    for(stage = 1; stage <= nstage; stage++) {
        mode = modes[stage - 1];
        lidx = (stage - 1) * 2;
        sprintf(lut_path, "%s/x%d_%db_i%d_s%d_%c_L2.bin", dir, scale, interval, bitwidth, stage, mode);
        mdl->lut[lidx] = TensorData_load(lut_path);
        printf("Load lut[%d]<%s>(%d, %d): %s\n", lidx, TensorType_str(mdl->lut[lidx]->dtype), mdl->lut[lidx]->shape[0], mdl->lut[lidx]->shape[1], lut_path);
        sprintf(lut_path, "%s/x%d_%db_i%d_s%d_%c_H6.bin", dir, scale, interval, bitwidth, stage, mode);
        mdl->lut[lidx + 1] = TensorData_load(lut_path);
        printf("Load lut[%d]<%s>(%d, %d): %s\n", lidx + 1, TensorType_str(mdl->lut[lidx + 1]->dtype), mdl->lut[lidx + 1]->shape[0], mdl->lut[lidx + 1]->shape[1], lut_path);
#ifdef LUT_PERF
        // <<< Add LUT Size to perf_mem >>>
        mdl->perf_mem += TensorData_size(mdl->lut[lidx]) + TensorData_size(mdl->lut[lidx + 1]);
#endif // LUT_PERF
    }

    for(int i = 2 * nstage; i < LUT_PTR_MAX; i++) {
        mdl->lut[i] = NULL;
    }

#ifdef LUT_MEMPOOL
    MemPool_init(&mdl->mem_pool, 10, NULL);
#endif // LUT_MEMPOOL
    
}

void TinyLUT_forward(TinyLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C) {
    if(!mdl || !O || !I) {
        return;
    }

    // 0. Prepare
    int H2 = H * mdl->scale;
    int W2 = W * mdl->scale;
    char mode;
    int stage, rot, pad = 2, norm, bias, scale = mdl->scale;
    int HP = H + pad;
    int WP = W + pad;
    int C2 = C * mdl->scale * mdl->scale;

#define DTYPEN 1

#if DTYPEN == 1
#   define DTYPE int8_t
#   define DTYPEC s8
#elif DTYPEN == 2
#   define DTYPE int16_t
#   define DTYPEC s16
#elif DTYPEN == 4
#   define DTYPE int32_t
#   define DTYPEC s32
#elif DTYPEN == 8
#   define DTYPE int64_t
#   define DTYPEC s64
#endif // DTYPEN

    typedef DTYPE dtype_t;    // Setup dtype
    dtype_t *trg, **mid = (dtype_t**)mdl->mid;

    // 1. Static Alloc middle tensor
#ifdef LUT_MEMPOOL
    // Use MemPool instead
    MemInfo mem_info[10] = {
        { .start =  0, .end = 11, .size =  H *  W *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size =  H *  W *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = HP * WP *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = HP * WP *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = HP * WP *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(dtype_t) },
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(dtype_t) },
    };
    mdl->mem_pool.info = mem_info;
    MemPool_solve(&mdl->mem_pool, (void**)mid, 0);
#else
    // Static Alloc by hand
    mid[0] = LUT_OPR(DAlc, DTYPEC)( H *  W * C);             // in   [ H,  W,  C]    --- prepare
    mid[1] = LUT_OPR(DAlc, DTYPEC)(HP * WP * C);             // rot  [ H,  W,  C]
    mid[2] = LUT_OPR(DAlc, DTYPEC)(HP * WP * C);             // pad  [HP, WP,  C]
    mid[3] = LUT_OPR(DAlc, DTYPEC)(HP * WP * C);             // low  [HP, WP,  C]    --- segment
    mid[4] = LUT_OPR(DAlc, DTYPEC)(HP * WP * C);             // high [HP, WP,  C]
    mid[5] = LUT_OPR(DAlc, DTYPEC)(H2 * W2 * C);             // hhh1 [H2, W2,  C]    --- MSB
    mid[6] = LUT_OPR(DAlc, DTYPEC)(H2 * W2 * C);             // hhh2 [H2, W2,  C]
    mid[7] = LUT_OPR(DAlc, DTYPEC)(H2 * W2 * C);             // lll1 [H2, W2,  C]    --- LSB
    mid[8] = LUT_OPR(DAlc, DTYPEC)(H2 * W2 * C);             // lll2 [H2, W2,  C] 
    mid[9] = LUT_OPR(DAlc, DTYPEC)(H2 * W2 * C);             // out  [H2, W2,  C]    --- output
#endif // LUT_MEMPOOL


#ifdef LUT_PERF
    if(mdl->perf_cnt == 0) {
#ifdef LUT_MEMPOOL
        MemPool_disp(&mdl->mem_pool);
        mdl->perf_mem += mdl->mem_pool.size;    // <<< Add Middle Tensor Size to perf_mem >>>
#else 
        mdl->perf_mem += (1 * (H *  W *  C) + 4 * (HP * WP * C) + 5 * (H2 * W2 * C)) * sizeof(dtype_t);    // <<< Add Middle Tensor Size to perf_mem >>>
#endif // LUT_MEMPOOL
    }
    double tstart, tend, tval;
    tstart = get_time_ms();
#endif // LUT_PERF

    // 2. Forward
#if DTYPEN == 1 // s8
    Cast_s8_u8(mid[0], I, H * W * C);
    Zero_s8(mid[9], H2 * W2 * C);
    for(rot = 0; rot < 4; rot++) {
        // ---- Prepare
        Rot_s8_hwc(mid[1], mid[0], C, H, W, rot);
        Pad_s8_hwc(mid[2], mid[1], C, H, W, pad, 0, 1);
        Seg_s8_hwc(mid[3], mid[4], mid[2], HP * WP * C, 2);

        // ---- LSB
        Intp_depthwise_s8_hwc(mid[7], mid[3], mdl->lut[0]->data, C, HP, WP, 3, scale, 1, 0);
        MulQ_tile_s16_s8_hwc(mid[7], mid[7], TinyLUT_hl1, TinyLUT_hl1_scale, TinyLUT_hl1_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s8_hwc(mid[8], mid[7], mdl->lut[2]->data, TinyLUT_hl1, C, H2, W2, TinyLUT_hl1_scale, TinyLUT_hl1_offset, mdl->scale, 1, 0, 0);
        MulQ_tile_s16_s8_hwc(mid[8], mid[8], TinyLUT_hl2, TinyLUT_hl2_scale, TinyLUT_hl2_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s8_hwc(mid[7], mid[8], mdl->lut[4]->data, TinyLUT_hl2, C, H2, W2, TinyLUT_hl2_scale, TinyLUT_hl2_offset, mdl->scale, 1, 1, 0);

        // ---- MSB
        Intp_depthwise_s8_hwc(mid[5], mid[4], mdl->lut[1]->data, C, HP, WP, 3, scale, 1, 1);
        MulQ_tile_s16_s8_hwc(mid[5], mid[5], TinyLUT_hh1, TinyLUT_hh1_scale, TinyLUT_hh1_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s8_hwc(mid[6], mid[5], mdl->lut[3]->data, TinyLUT_hh1, C, H2, W2, TinyLUT_hh1_scale, TinyLUT_hh1_offset, mdl->scale, 1, 0, 1);
        MulQ_tile_s16_s8_hwc(mid[6], mid[6], TinyLUT_hh2, TinyLUT_hh2_scale, TinyLUT_hh2_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s8_hwc(mid[5], mid[6], mdl->lut[5]->data, TinyLUT_hh2, C, H2, W2, TinyLUT_hh2_scale, TinyLUT_hh2_offset, mdl->scale, 1, 1, 1);

        // ---- Merge
        Add_s8(mid[7], mid[7], mid[5], H2 * W2 * C);
        Rot_s8_hwc(mid[8], mid[7], C, H2, W2, 4 - rot);
        Add_s8(mid[9], mid[9], mid[8], H2 * W2 * C);
    }
    // Disp_s8_chw(mid[7], H2, W2, C);
    // Disp_s8_chw(mid[5], H2, W2, C);
    // ---- Output
    Cast_u8_s8(O, mid[9], H2 * W2 * C);

#elif DTYPEN == 4 // s32
    Cast_s32_u8(mid[0], I, H * W * C);
    Addi_s32(mid[0], mid[0], -128, H * W * C);
    Zero_s32(mid[9], H2 * W2 * C);
    for(rot = 0; rot < 4; rot++) {
        // ---- Prepare
        Rot_s32_hwc(mid[1], mid[0], C, H, W, rot);
        Pad_s32_hwc(mid[2], mid[1], C, H, W, pad, 0, 1);
        Seg_s32_hwc(mid[3], mid[4], mid[2], HP * WP * C, 2);

        // ---- LSB
        Intp_depthwise_s32_hwc(mid[7], mid[3], mdl->lut[0]->data, C, HP, WP, 3, scale, 1, 0);
        MulQ_tile_s16_s32_hwc(mid[7], mid[7], TinyLUT_hl1, TinyLUT_hl1_scale, TinyLUT_hl1_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s32_hwc(mid[8], mid[7], mdl->lut[2]->data, TinyLUT_hl1, C, H2, W2, TinyLUT_hl1_scale, TinyLUT_hl1_offset, mdl->scale, 1, 0, 0);
        MulQ_tile_s16_s32_hwc(mid[8], mid[8], TinyLUT_hl2, TinyLUT_hl2_scale, TinyLUT_hl2_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s32_hwc(mid[7], mid[8], mdl->lut[4]->data, TinyLUT_hl2, C, H2, W2, TinyLUT_hl2_scale, TinyLUT_hl2_offset, mdl->scale, 1, 1, 0);

        // ---- MSB
        Intp_depthwise_s32_hwc(mid[5], mid[4], mdl->lut[1]->data, C, HP, WP, 3, scale, 1, 1);
        MulQ_tile_s16_s32_hwc(mid[5], mid[5], TinyLUT_hh1, TinyLUT_hh1_scale, TinyLUT_hh1_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s32_hwc(mid[6], mid[5], mdl->lut[3]->data, TinyLUT_hh1, C, H2, W2, TinyLUT_hh1_scale, TinyLUT_hh1_offset, mdl->scale, 1, 0, 1);
        MulQ_tile_s16_s32_hwc(mid[6], mid[6], TinyLUT_hh2, TinyLUT_hh2_scale, TinyLUT_hh2_offset, C, H2, W2, mdl->scale);
        Intp_pointwise_s32_hwc(mid[5], mid[6], mdl->lut[5]->data, TinyLUT_hh2, C, H2, W2, TinyLUT_hh2_scale, TinyLUT_hh2_offset, mdl->scale, 1, 1, 1);

        // ---- Merge
        Add_s32(mid[7], mid[7], mid[5], H2 * W2 * C);
        Clamp_s32(mid[7], mid[7], H2 * W2 * C, 8, 1);
        Rot_s32_hwc(mid[8], mid[7], C, H2, W2, 4 - rot);
        Add_s32(mid[9], mid[9], mid[8], H2 * W2 * C);
    }
    //  ---- Output
    Clamp_s32(mid[9], mid[9], H2 * W2 * C, 8, 1);
    Addi_s32(mid[9], mid[9], 128, H2 * W2 * C);
    Cast_u8_s32(O, mid[9], H2 * W2 * C);

#else
#   error "Unsupported DTYPE"
#endif

#ifdef LUT_PERF
    tend = get_time_ms();
    tval = tend - tstart;
    mdl->perf_cnt++;
    mdl->perf_time += (float)tval / 1000.0;
    // total cnt / total time
    mdl->perf_fps = mdl->perf_cnt / mdl->perf_time;
    printf("TinyLUT forward [%d,%d,%d] %2d: %.3f ms, %.3f fps, %ld KB\n", H, W, C, mdl->perf_cnt, tval, mdl->perf_fps, mdl->perf_mem / (1024));
#endif // LUT_PERF

    // 3. Free
#ifdef LUT_MEMPOOL
    // Use MemPool instead
    if(mdl->free_mid) {
        MemPool_free(&mdl->mem_pool);
    }
#else
    if(mdl->free_mid) {
        for(int i = 0; i < 10; i++) {
            DRlz_s32(mid[i]);
        }
    }
#endif // LUT_MEMPOOL

}

#ifdef LUT_MEMPOOL
void TinyLUT_forward_opt(TinyLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C) {
    if(!mdl || !O || !I) {
        return;
    }
    // 0. Prepare
    int H2 = H * mdl->scale;
    int W2 = W * mdl->scale;
    char mode;
    int stage, rot, pad = 2, norm, bias, scale = mdl->scale;
    int HP = H + pad;
    int WP = W + pad;
    int C2 = C * mdl->scale * mdl->scale;
    int8_t *trg, **mid = (int8_t**)mdl->mid;

    // 1. Static Alloc middle tensor
    MemInfo mem_info[10] = {
        { .start =  0, .end = 11, .size =  H *  W *  C * sizeof(int8_t) },  // in
        { .start =  0, .end = 11, .size =  H *  W *  C * sizeof(int8_t) },  // rot
        { .start =  0, .end = 11, .size = HP * WP *  C * sizeof(int8_t) },  // pad
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(int8_t) },  // Depthwise
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(int8_t) },  // Pointwise1
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(int8_t) },  // Pointwise2
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(int8_t) },  // rot
        { .start =  0, .end = 11, .size = H2 * W2 *  C * sizeof(int8_t) },  // out
    };
    mdl->mem_pool.info = mem_info;
    MemPool_solve(&mdl->mem_pool, (void**)mid, 0);

#ifdef LUT_PERF
    if(mdl->perf_cnt == 0) {
        MemPool_disp(&mdl->mem_pool);
        mdl->perf_mem += mdl->mem_pool.size;    // <<< Add Middle Tensor Size to perf_mem >>>
    }
    double tstart, tend, tval;
    tstart = get_time_ms();
#endif // LUT_PERF

    // 2. Forward
    Cast_s8_u8(mid[0], I, H * W * C);
    Zero_s8(mid[7], H2 * W2 * C);
    for(rot = 0; rot < 4; rot++) {
        // ---- Prepare
        Rot_s8_hwc(mid[1], mid[0], C, H, W, rot);
        Pad_s8_hwc(mid[2], mid[1], C, H, W, pad, 0, 1);

        // ---- Depthwise
        Intp_fuse_s8_hwc(
            mid[3], mid[2], 
            mdl->lut[0]->data, 
            mdl->lut[1]->data, 
            NULL, NULL, C, HP, WP, 
            0, 0, 0, 0, 
            4, 3, 1, 0, 1);

        // ---- Pointwise
        Intp_fuse_s8_hwc(
            mid[4], mid[3], 
            mdl->lut[2]->data, 
            mdl->lut[3]->data, 
            TinyLUT_hl1, TinyLUT_hh1, C, H2, W2, 
            TinyLUT_hl1_scale, TinyLUT_hl1_offset, 
            TinyLUT_hh1_scale, TinyLUT_hh1_offset, 
            4, 0, 1, 0, 0);

        // ---- Pointwise
        Intp_fuse_s8_hwc(
            mid[5], mid[4], 
            mdl->lut[4]->data, 
            mdl->lut[5]->data, 
            TinyLUT_hl2, TinyLUT_hh2, C, H2, W2, 
            TinyLUT_hl2_scale, TinyLUT_hl2_offset, 
            TinyLUT_hh2_scale, TinyLUT_hh2_offset, 
            4, 0, 1, 1, 0);

        // ---- Merge
        Rot_s8_hwc(mid[6], mid[5], C, H2, W2, 4 - rot);
        Add_s8(mid[7], mid[7], mid[6], H2 * W2 * C);
    }

    // Disp_s8_chw(mid[4], H2, W2, C);
    // ---- Output
    Cast_u8_s8(O, mid[7], H2 * W2 * C);


#ifdef LUT_PERF
    tend = get_time_ms();
    tval = tend - tstart;
    mdl->perf_cnt++;
    mdl->perf_time += (float)tval / 1000.0;
    // total cnt / total time
    mdl->perf_fps = mdl->perf_cnt / mdl->perf_time;
    printf("TinyLUT forward [%d,%d,%d] %2d: %.3f ms, %.3f fps, %ld KB\n", H, W, C, mdl->perf_cnt, tval, mdl->perf_fps, mdl->perf_mem / (1024));
#endif // LUT_PERF

    // 3. Free
    if(mdl->free_mid) {
        MemPool_free(&mdl->mem_pool);
    }
}
#endif // LUT_MEMPOOL


void TinyLUT_free(TinyLUT *mdl) {
    if(!mdl) {
        return;
    }

    if(mdl->dir) {
        free(mdl->dir);
        mdl->dir = NULL;
    }
    if(mdl->modes) {
        free(mdl->modes);
        mdl->modes = NULL;
    }
#ifdef LUT_MEMPOOL
    MemPool_free(&mdl->mem_pool);
#endif // LUT_MEMPOOL
    for(int i = 0; i < LUT_PTR_MAX; i++) {
        TensorData_free(mdl->lut[i]);
        mdl->lut[i] = NULL;
        printf("Free lut[%d]\n", i);
    }
}