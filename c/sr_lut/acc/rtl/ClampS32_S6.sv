/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : Fri Feb 28 15:51:36 CST 2025
 * License      : MIT
 */

module ClampS32_S6 (
    input  logic signed [31:0] x,
    output logic signed [5:0] y
);

    localparam MIN_VAL = 
        -(1 << (6 - 1));

    localparam MAX_VAL = 
        (1 << (6 - 1)) - 1;

    always_comb begin
            if (x < MIN_VAL) begin
                y = MIN_VAL[5:0];
            end else if (x > MAX_VAL) begin
                y = MAX_VAL[5:0];
            end else begin
                y = x[5:0];
            end
    end
endmodule
