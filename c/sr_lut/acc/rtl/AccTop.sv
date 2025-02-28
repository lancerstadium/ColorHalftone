
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */
module AccTop (
    input  wire [31:0]  a,
    output  wire [31:0]  b
);
    // CastD8_U2S cast1(a[7:0], b[7:0]);    // OK
    // CastD8_S2U cast2(a[7:0], b[7:0]);    // OK
    // ClampS32_U2 clamp1(a, b[1:0]);       // OK
    // ClampS32_S6 clamp2(a, b[5:0]);       // OK
    // RoundDivS32_P9 div9(a, b);           // OK
    // RoundDivS32_P16 div16(a, b);         // OK
    DepthLUT_3x50x50_K3_U4_D8 depthlut1(a, b);
endmodule
/* verilator lint_on UNDRIVEN */
/* verilator lint_on UNUSEDSIGNAL */
