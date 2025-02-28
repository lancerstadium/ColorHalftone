/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : Fri Feb 28 15:51:36 CST 2025
 * License      : MIT
 */

module ClampS32_U2 (
    input  logic signed [31:0] x,
    output logic  [1:0] y
);

    localparam MIN_VAL = 
        0;

    localparam MAX_VAL = 
        (1 << 2) - 1;

    always_comb begin
            if (x < 0) begin
                y = 0;
            end else if (x > MAX_VAL) begin
                y = MAX_VAL[1:0];
            end else begin
                y = x[1:0];
            end
    end
endmodule
