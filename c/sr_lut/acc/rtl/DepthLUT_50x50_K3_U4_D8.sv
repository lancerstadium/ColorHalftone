/**
 * Module       : DepthLUT
 * Input        : I [H][W] 
 * Output       : O [H_OUT][W_OUT]
 * Description  : Depthwise Interpolation Lookup Table
 * Author       : lancerstadium
 * Date         : Tue Mar  4 16:41:09 CST 2025
 * License      : MIT
 */

module DepthLUT_50x50_K3_U4_D8 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         [7:0] I [0:49][0:49],
    output logic         [7:0] O [0:47][0:47][0:3],
    output logic         lut_mode
);

    //============= LUT模块信号定义 =============//
    logic [9:0] lut_addr_msb_base;
    logic [5:0] lut_addr_lsb_base;
    logic [31:0] lut_data_msb [0:3];
    logic [31:0] lut_data_lsb [0:3];
    logic msb_done, lsb_done;

    //============= LUT模块实例化 =============//
    LUTTable_x4_4b_i8_s1_D_H6 u_lut_msb (
        .en_write(lut_mode[1]),
        .en_read(lut_mode[0]),
        .base(lut_addr_msb_base),
        .rdata(lut_data_msb)
    );

    LUTTable_x4_4b_i8_s1_D_L2 u_lut_lsb (
        .en_write(lut_mode[1]),
        .en_read(lut_mode[0]),
        .base(lut_addr_lsb_base),
        .rdata(lut_data_lsb)
    );


    //============= 核心计算逻辑优化 =============//
    generate
        for (genvar h = 0; h < 48; h++) begin : H_ROW
            for (genvar w = 0; w < 48; w++) begin : W_COL
                // 窗口缓存寄存器
                logic signed [7:0] kernel_reg [0:8];
                logic kernel_valid;
                logic [3:0] phase_cnt;
                
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        kernel_valid <= 0;
                        phase_cnt <= 0;
                        lut_mode <= 0;
                        lut_addr_msb_base <= 0;
                        lut_addr_lsb_base <= 0;
                    end else begin
                        // 装载新kernel
                        kernel_reg[0] <= I[h*4+0][w*4+0];
                        kernel_reg[1] <= I[h*4+0][w*4+1];
                        kernel_reg[2] <= I[h*4+0][w*4+2];
                        kernel_reg[3] <= I[h*4+1][w*4+0];
                        kernel_reg[4] <= I[h*4+1][w*4+1];
                        kernel_reg[5] <= I[h*4+1][w*4+2];
                        kernel_reg[6] <= I[h*4+2][w*4+0];
                        kernel_reg[7] <= I[h*4+2][w*4+1];
                        kernel_reg[8] <= I[h*4+2][w*4+2];
                        kernel_valid <= 1;

                        // 地址生成逻辑
                        if (kernel_valid) begin
                            lut_addr_msb_base[0*6 +: 6] <= kernel_reg[0][7:2];
                            lut_addr_lsb_base[0*2 +: 2] <= kernel_reg[0][1:0];
                            lut_addr_msb_base[1*6 +: 6] <= kernel_reg[1][7:2];
                            lut_addr_lsb_base[1*2 +: 2] <= kernel_reg[1][1:0];
                            lut_addr_msb_base[2*6 +: 6] <= kernel_reg[2][7:2];
                            lut_addr_lsb_base[2*2 +: 2] <= kernel_reg[2][1:0];
                            lut_addr_msb_base[3*6 +: 6] <= kernel_reg[3][7:2];
                            lut_addr_lsb_base[3*2 +: 2] <= kernel_reg[3][1:0];
                            lut_addr_msb_base[4*6 +: 6] <= kernel_reg[4][7:2];
                            lut_addr_lsb_base[4*2 +: 2] <= kernel_reg[4][1:0];
                            lut_addr_msb_base[5*6 +: 6] <= kernel_reg[5][7:2];
                            lut_addr_lsb_base[5*2 +: 2] <= kernel_reg[5][1:0];
                            lut_addr_msb_base[6*6 +: 6] <= kernel_reg[6][7:2];
                            lut_addr_lsb_base[6*2 +: 2] <= kernel_reg[6][1:0];
                            lut_addr_msb_base[7*6 +: 6] <= kernel_reg[7][7:2];
                            lut_addr_lsb_base[7*2 +: 2] <= kernel_reg[7][1:0];
                            lut_addr_msb_base[8*6 +: 6] <= kernel_reg[8][7:2];
                            lut_addr_lsb_base[8*2 +: 2] <= kernel_reg[8][1:0];
                            lut_mode <= 1;
                        end else begin
                            lut_mode <= 0;
                        end
                    end
                end

                // 恢复原始后处理模块
                for (genvar p = 0; p < 16; p++) begin : PHASE
                    logic signed [31:0] acc_msb, acc_lsb;
                    logic signed [31:0] div_msb, div_lsb;
                    logic signed [5:0] clamp_msb;
                    logic signed [1:0] clamp_lsb;

                    // 累加逻辑
                    always_ff @(posedge clk) begin
                        if (lut_mode) begin
                            acc_msb <= 
                            + $signed(lut_data_msb[0][7:0]) + $signed(lut_data_msb[0][15:8]) + $signed(lut_data_msb[0][23:16]) + $signed(lut_data_msb[0][31:24])
                            + $signed(lut_data_msb[1][7:0]) + $signed(lut_data_msb[1][15:8]) + $signed(lut_data_msb[1][23:16]) + $signed(lut_data_msb[1][31:24])
                            + $signed(lut_data_msb[2][7:0]) + $signed(lut_data_msb[2][15:8]) + $signed(lut_data_msb[2][23:16]) + $signed(lut_data_msb[2][31:24])
                            + $signed(lut_data_msb[3][7:0]) + $signed(lut_data_msb[3][15:8]) + $signed(lut_data_msb[3][23:16]) + $signed(lut_data_msb[3][31:24])
                            + $signed(lut_data_msb[4][7:0]) + $signed(lut_data_msb[4][15:8]) + $signed(lut_data_msb[4][23:16]) + $signed(lut_data_msb[4][31:24])
                            + $signed(lut_data_msb[5][7:0]) + $signed(lut_data_msb[5][15:8]) + $signed(lut_data_msb[5][23:16]) + $signed(lut_data_msb[5][31:24])
                            + $signed(lut_data_msb[6][7:0]) + $signed(lut_data_msb[6][15:8]) + $signed(lut_data_msb[6][23:16]) + $signed(lut_data_msb[6][31:24])
                            + $signed(lut_data_msb[7][7:0]) + $signed(lut_data_msb[7][15:8]) + $signed(lut_data_msb[7][23:16]) + $signed(lut_data_msb[7][31:24])
                            + $signed(lut_data_msb[8][7:0]) + $signed(lut_data_msb[8][15:8]) + $signed(lut_data_msb[8][23:16]) + $signed(lut_data_msb[8][31:24])
                            ;
                            acc_lsb <= 
                            + $signed(lut_data_lsb[0][7:0]) + $signed(lut_data_lsb[0][15:8]) + $signed(lut_data_lsb[0][23:16]) + $signed(lut_data_lsb[0][31:24])
                            + $signed(lut_data_lsb[1][7:0]) + $signed(lut_data_lsb[1][15:8]) + $signed(lut_data_lsb[1][23:16]) + $signed(lut_data_lsb[1][31:24])
                            + $signed(lut_data_lsb[2][7:0]) + $signed(lut_data_lsb[2][15:8]) + $signed(lut_data_lsb[2][23:16]) + $signed(lut_data_lsb[2][31:24])
                            + $signed(lut_data_lsb[3][7:0]) + $signed(lut_data_lsb[3][15:8]) + $signed(lut_data_lsb[3][23:16]) + $signed(lut_data_lsb[3][31:24])
                            + $signed(lut_data_lsb[4][7:0]) + $signed(lut_data_lsb[4][15:8]) + $signed(lut_data_lsb[4][23:16]) + $signed(lut_data_lsb[4][31:24])
                            + $signed(lut_data_lsb[5][7:0]) + $signed(lut_data_lsb[5][15:8]) + $signed(lut_data_lsb[5][23:16]) + $signed(lut_data_lsb[5][31:24])
                            + $signed(lut_data_lsb[6][7:0]) + $signed(lut_data_lsb[6][15:8]) + $signed(lut_data_lsb[6][23:16]) + $signed(lut_data_lsb[6][31:24])
                            + $signed(lut_data_lsb[7][7:0]) + $signed(lut_data_lsb[7][15:8]) + $signed(lut_data_lsb[7][23:16]) + $signed(lut_data_lsb[7][31:24])
                            + $signed(lut_data_lsb[8][7:0]) + $signed(lut_data_lsb[8][15:8]) + $signed(lut_data_lsb[8][23:16]) + $signed(lut_data_lsb[8][31:24])
                            ;
                        end
                    end

                    // 恢复原始除法模块
                    RoundDivS32_P9 u_div_msb (
                        .x(acc_msb),
                        .result(div_msb)
                    );
                    
                    RoundDivS32_P9 u_div_lsb (
                        .x(acc_lsb),
                        .result(div_lsb)
                    );

                    // 恢复原始钳位模块
                    ClampS32_S6 u_clamp_msb (
                        .x(div_msb),
                        .y(clamp_msb)
                    );
                    
                    ClampS32_U2 u_clamp_lsb (
                        .x(div_lsb),
                        .y(clamp_lsb)
                    );

                    // 输出寄存器
                    always_ff @(posedge clk) begin
                        O[h][w][p] <= {clamp_msb, clamp_lsb};
                    end
                end
            end
        end
    endgenerate

endmodule
