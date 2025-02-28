
module AccTop (
    input  wire [31:0]  a,
    input  wire [31:0]  b
);

    RoundDivS32_P9 div1(a, b);
    
endmodule
