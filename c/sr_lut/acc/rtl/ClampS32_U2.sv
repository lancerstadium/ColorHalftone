/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : Sun Mar  2 18:12:13 CST 2025
 * License      : MIT
 */

module ClampS32_U2 (
    input  logic signed [31:0] x,
    output logic  [1:0] y
);
    // Unsigned clamping parameters
    localparam MAX_VAL = (1 << 2) - 1;
    always_comb begin
        logic sign_bit = x[31];
        logic [29:0] upper_bits = x[31:2];
        
        y = sign_bit       ? 2'b0 :
            (|upper_bits) ? MAX_VAL[1:0] :
            x[1:0];
    end
endmodule
