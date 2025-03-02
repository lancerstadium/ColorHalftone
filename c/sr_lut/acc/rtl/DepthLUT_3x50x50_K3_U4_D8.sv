/**
 * Module       : DepthLUT
 * Input        : I [C][H][W] 
 * Output       : O [C][H_OUT][W_OUT]
 * Description  : Depthwise Interpolation Lookup Table
 * Author       : lancerstadium
 * Date         : Sun Mar  2 18:12:13 CST 2025
 * License      : MIT
 */

module DepthLUT_3x50x50_K3_U4_D8 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic signed [7:0] I [0:2][0:49][0:49],
    output logic signed [7:0] O [0:2][0:47][0:47][0:15],
    // LUT模块接口
    output logic        lut_req,      // 同时触发两个LUT
    input  logic        lut_done     // 两个LUT都完成
);

    //============= LUT模块信号定义 =============//
    logic [9:0] lut_addr_msb_base;
    logic [5:0] lut_addr_lsb_base;
    logic signed [7:0] lut_data_msb [0:8];
    logic signed [7:0] lut_data_lsb [0:8];
    logic msb_done, lsb_done;

    //============= LUT模块实例化 =============//
    LUTTable_x4_4b_i8_s1_D_H6 u_lut_msb (
        .clk(clk),
        .rst_n(rst_n),
        .burst_start(lut_req),
        .burst_done(msb_done),
        .base_addr(lut_addr_msb_base),
        .data(lut_data_msb)
    );

    LUTTable_x4_4b_i8_s1_D_L2 u_lut_lsb (
        .clk(clk),
        .rst_n(rst_n),
        .burst_start(lut_req),
        .burst_done(lsb_done),
        .base_addr(lut_addr_lsb_base),
        .data(lut_data_lsb)
    );

    assign lut_done = msb_done & lsb_done;

    //============= 核心计算逻辑优化 =============//
    generate
        for (genvar c = 0; c < 3; c++) begin : CHAN
            for (genvar h = 0; h < 48; h++) begin : H_ROW
                for (genvar w = 0; w < 48; w++) begin : W_COL
                    // 窗口缓存寄存器
                    logic signed [7:0] kernel_reg [0:8];
                    logic kernel_valid;
                    logic [3:0] phase_cnt;
                    
                    // 修正后的地址生成逻辑
                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            kernel_valid <= 0;
                            phase_cnt <= 0;
                            lut_req <= 0;
                            // 初始化地址寄存器
                            lut_addr_msb_base <= 0;
                            lut_addr_lsb_base <= 0;
                        end else begin
                            // 装载新kernel
                            kernel_reg[0] <= I[c][h*4+0][w*4+0];
                            kernel_reg[1] <= I[c][h*4+0][w*4+1];
                            kernel_reg[2] <= I[c][h*4+0][w*4+2];
                            kernel_reg[3] <= I[c][h*4+1][w*4+0];
                            kernel_reg[4] <= I[c][h*4+1][w*4+1];
                            kernel_reg[5] <= I[c][h*4+1][w*4+2];
                            kernel_reg[6] <= I[c][h*4+2][w*4+0];
                            kernel_reg[7] <= I[c][h*4+2][w*4+1];
                            kernel_reg[8] <= I[c][h*4+2][w*4+2];
                            kernel_valid <= 1;

                            // 地址生成逻辑（修正拼接顺序）
                            if (kernel_valid) begin
                                lut_addr_msb_base[0*6 +: 6] <= 
                                    kernel_reg[0][7:2];
                                lut_addr_lsb_base[0*2 +: 2] <= 
                                    kernel_reg[0][1:0];
                                lut_addr_msb_base[1*6 +: 6] <= 
                                    kernel_reg[1][7:2];
                                lut_addr_lsb_base[1*2 +: 2] <= 
                                    kernel_reg[1][1:0];
                                lut_addr_msb_base[2*6 +: 6] <= 
                                    kernel_reg[2][7:2];
                                lut_addr_lsb_base[2*2 +: 2] <= 
                                    kernel_reg[2][1:0];
                                lut_addr_msb_base[3*6 +: 6] <= 
                                    kernel_reg[3][7:2];
                                lut_addr_lsb_base[3*2 +: 2] <= 
                                    kernel_reg[3][1:0];
                                lut_addr_msb_base[4*6 +: 6] <= 
                                    kernel_reg[4][7:2];
                                lut_addr_lsb_base[4*2 +: 2] <= 
                                    kernel_reg[4][1:0];
                                lut_addr_msb_base[5*6 +: 6] <= 
                                    kernel_reg[5][7:2];
                                lut_addr_lsb_base[5*2 +: 2] <= 
                                    kernel_reg[5][1:0];
                                lut_addr_msb_base[6*6 +: 6] <= 
                                    kernel_reg[6][7:2];
                                lut_addr_lsb_base[6*2 +: 2] <= 
                                    kernel_reg[6][1:0];
                                lut_addr_msb_base[7*6 +: 6] <= 
                                    kernel_reg[7][7:2];
                                lut_addr_lsb_base[7*2 +: 2] <= 
                                    kernel_reg[7][1:0];
                                lut_addr_msb_base[8*6 +: 6] <= 
                                    kernel_reg[8][7:2];
                                lut_addr_lsb_base[8*2 +: 2] <= 
                                    kernel_reg[8][1:0];
                                lut_req <= 1;
                            end else begin
                                lut_req <= 0;
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
                            if (lut_done) begin
                                acc_msb <= 0;
                                acc_lsb <= 0;
                                acc_msb <= acc_msb + $signed(lut_data_msb[0]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[0]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[1]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[1]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[2]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[2]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[3]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[3]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[4]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[4]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[5]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[5]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[6]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[6]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[7]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[7]);
                                acc_msb <= acc_msb + $signed(lut_data_msb[8]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[8]);
                            end
                        end

                        // 恢复原始除法模块
                        RoundDivS32_P9 u_div_msb (
                            .clk(clk),
                            .x(acc_msb),
                            .result(div_msb)
                        );
                        
                        RoundDivS32_P9 u_div_lsb (
                            .clk(clk),
                            .x(acc_lsb),
                            .result(div_lsb)
                        );

                        // 恢复原始钳位模块
                        ClampS32_S6 u_clamp_msb (
                            .clk(clk),
                            .x(div_msb),
                            .y(clamp_msb)
                        );
                        
                        ClampS32_U2 u_clamp_lsb (
                            .clk(clk),
                            .x(div_lsb),
                            .y(clamp_lsb)
                        );

                        // 输出寄存器
                        always_ff @(posedge clk) begin
                            O[c][h][w][p] <= {clamp_msb, clamp_lsb};
                        end
                    end
                end
            end
        end
    endgenerate

endmodule
