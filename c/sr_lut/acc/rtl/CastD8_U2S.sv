/**
 * Module       : CastD8
 * Input        : x[7:0] - Signed/Unsigned 8-bit interger
 * Output       : y[7:0] - Unsigned/Signed 8-bit interger
 * Description  : Signed 8-bit clamp to the range of a sub-bit signed integer or reverse.
 * Author       : lancerstadium
 * Date         : Tue Mar  4 16:41:09 CST 2025
 * License      : MIT
 */

/* verilator lint_off WIDTHEXPAND */
module CastD8_U2S (
    input [7:0] x,
    output [7:0] y
);
    // Core conversion logic using bitwise XOR for efficient conversion
    // S2U: -128(0x80)~127(0x7F) -> 0~255 (XOR 0x80)
    // U2S: 0~255 -> -128~127 (XOR 0x80)
    assign y = x ^ 8'h80;
endmodule
/* verilator lint_on WIDTHEXPAND */
