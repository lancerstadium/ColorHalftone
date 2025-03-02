
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */
module AccTop (
    input   logic         clk,      // 全局时钟
    input   logic         rst_n,    // 异步复位
    input   logic         a,
    output  logic         b,
    output  logic signed [7:0] data [0:15]
);
    // CastD8_U2S cast1(a[7:0], b[7:0]);    // OK
    // CastD8_S2U cast2(a[7:0], b[7:0]);    // OK
    // ClampS32_U2 clamp1(a, b[1:0]);       // OK
    // ClampS32_S6 clamp2(a, b[5:0]);       // OK
    // RoundDivS32_P9 div9(a, b);           // OK
    // RoundDivS32_P16 div16(a, b);         // OK

    logic [9:0] addr = 10'b0;

    LUTTable_x4_4b_i8_s1_D_H6 lut1(
        .clk(clk),
        .rst_n(rst_n),
        .burst_start(a),
        .burst_done(b),
        .base_addr(addr),
        .data(data)
    );
endmodule
/* verilator lint_on UNDRIVEN */
/* verilator lint_on UNUSEDSIGNAL */
