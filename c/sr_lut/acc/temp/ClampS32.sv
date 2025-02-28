/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    sign = context.get('sign', False)
    bitwidth = context.get('bitwidth', 8)
    module_name = f"ClampS32_{'S' if sign else 'U'}{bitwidth}"
%>
module ${module_name} (
    input  logic signed [31:0] x,
    output logic ${'signed' if sign else ''} [${bitwidth-1}:0] y
);

    localparam MIN_VAL = 
    % if sign:
        -(1 << (${bitwidth} - 1));
    % else:
        0;
    % endif

    localparam MAX_VAL = 
    % if sign:
        (1 << (${bitwidth} - 1)) - 1;
    % else:
        (1 << ${bitwidth}) - 1;
    % endif

    always_comb begin
        % if sign:
            if (x < MIN_VAL) begin
                y = MIN_VAL[${bitwidth-1}:0];
            end else if (x > MAX_VAL) begin
                y = MAX_VAL[${bitwidth-1}:0];
            end else begin
                y = x[${bitwidth-1}:0];
            end
        % else:
            if (x < 0) begin
                y = 0;
            end else if (x > MAX_VAL) begin
                y = MAX_VAL[${bitwidth-1}:0];
            end else begin
                y = x[${bitwidth-1}:0];
            end
        % endif
    end
endmodule
