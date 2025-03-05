/**
 * Module       : LUTTable
 * Input        : x[31:0] - Signed/Unsigned 8-bit interger
 * Output       : y[31:0] - Unsigned/Signed 8-bit interger
 * Description  : LUT Table for read and write.
 * Author       : lancerstadium
 * Date         : Tue Mar  4 16:41:09 CST 2025
 * License      : MIT
 */

module LUTTable_x4_4b_i8_s1_D_L2 (
    input  logic        en_write,                           // 脉冲启动批量写入
    input  logic        en_read,                            // 脉冲启动批量读取
    input  logic        [5:0] base,    // 非展平基地址
    input  logic        [31:0] wdata [0:3],    // 并行输入
    output logic        [31:0] rdata [0:3]     // 并行输出
);

    // Original Shape <int8x4>: (36, 4), Total Entries: 36
    logic [31:0] lut_mem [0:143]
    =
    '{
    /* (0, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (1, 0) */	    32'h01FE0000, 32'h02FFFFFB, 32'hFFFB0001, 32'hFF00FF00, 
    /* (2, 0) */	    32'h02FD0000, 32'h05FEFFF6, 32'hFEF7FF02, 32'hFE00FEFF, 
    /* (3, 0) */	    32'h04FB0000, 32'h07FDFEF1, 32'hFDF2FF03, 32'hFDFFFEFF, 
    /* (4, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (5, 0) */	    32'h00FF01FD, 32'h01FE01FB, 32'hFEFB0003, 32'hFE00FF00, 
    /* (6, 0) */	    32'h00FD01FB, 32'h02FD01F6, 32'hFCF5FF07, 32'hFB00FE01, 
    /* (7, 0) */	    32'h00FC02F8, 32'h03FB02F1, 32'hFAF0FF0A, 32'hF9FFFD01, 
    /* (8, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (9, 0) */	    32'h00FCFDFE, 32'h04FEFFFC, 32'h02FB0405, 32'h01000000, 
    /* (10, 0) */	    32'h00F8FAFC, 32'h07FDFFF7, 32'h04F60809, 32'h010000FF, 
    /* (11, 0) */	    32'h00F5F7F9, 32'h0BFBFEF3, 32'h05F10B0E, 32'h020100FF, 
    /* (12, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (13, 0) */	    32'h050001FE, 32'h040000FB, 32'hFEFBFF03, 32'hFD00FE00, 
    /* (14, 0) */	    32'h09FF02FB, 32'h07FFFFF7, 32'hFDF7FD07, 32'hFAFFFC01, 
    /* (15, 0) */	    32'h0EFF03F9, 32'h0BFFFFF2, 32'hFBF2FC0A, 32'hF7FFFA01, 
    /* (16, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (17, 0) */	    32'hF6FAFF09, 32'h01FEFEFE, 32'h00FE02FB, 32'h0202FFF9, 
    /* (18, 0) */	    32'hEBF4FF12, 32'h01FDFDFB, 32'h00FB05F5, 32'h0403FEF3, 
    /* (19, 0) */	    32'hE1EEFE1A, 32'h02FBFBF9, 32'h00F907F0, 32'h0505FCEC, 
    /* (20, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (21, 0) */	    32'h0C030B03, 32'h04FD01FA, 32'hF9FDF5FD, 32'hFAFEF901, 
    /* (22, 0) */	    32'h18051607, 32'h08FA02F5, 32'hF2FAEAFA, 32'hF4FCF201, 
    /* (23, 0) */	    32'h2408200A, 32'h0CF702EF, 32'hEBF7DFF7, 32'hEEFBEA02, 
    /* (24, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (25, 0) */	    32'h04FF00FD, 32'h0601FDF9, 32'h02FEFFFB, 32'h0101FF00, 
    /* (26, 0) */	    32'h08FE00FB, 32'h0C01FAF3, 32'h03FBFFF6, 32'h0203FF01, 
    /* (27, 0) */	    32'h0CFE01F8, 32'h1202F6EC, 32'h05F9FEF1, 32'h0304FE01, 
    /* (28, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (29, 0) */	    32'hFAFCFF04, 32'hF7F80C03, 32'hF7F5020A, 32'hF9FEFF00, 
    /* (30, 0) */	    32'hF3F7FE09, 32'hEFF01805, 32'hEDEA0514, 32'hF2FCFE00, 
    /* (31, 0) */	    32'hEDF3FD0D, 32'hE6E82508, 32'hE4DF071F, 32'hEAFBFD00, 
    /* (32, 0) */	    32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000, 
    /* (33, 0) */	    32'h00F5F7F8, 32'h09FBF4F7, 32'hFDFF07FC, 32'hFC01F803, 
    /* (34, 0) */	    32'h00E9EFF0, 32'h11F6E8ED, 32'hFBFE0DF9, 32'hF903F006, 
    /* (35, 0) */	    32'hFFDEE6E7, 32'h1AF1DCE4, 32'hF8FD14F5, 32'hF504E708
};

    // 写入数据
    always_latch begin
        if (en_write) begin
            lut_mem[base * 4 + 0] = wdata[0];
            lut_mem[base * 4 + 1] = wdata[1];
            lut_mem[base * 4 + 2] = wdata[2];
            lut_mem[base * 4 + 3] = wdata[3];
        end else begin
            ;
        end
    end

    // 读取数据
    always_latch begin
        if (en_read) begin
            rdata[0] = lut_mem[base * 4 + 0];
            rdata[1] = lut_mem[base * 4 + 1];
            rdata[2] = lut_mem[base * 4 + 2];
            rdata[3] = lut_mem[base * 4 + 3];
        end else begin
            ;
        end
    end

endmodule
