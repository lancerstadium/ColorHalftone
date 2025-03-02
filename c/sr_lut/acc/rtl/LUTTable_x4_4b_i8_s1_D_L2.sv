/**
 * Module       : LUTTable
 * Input        : x[7:0] - Signed/Unsigned 8-bit interger
 * Output       : y[7:0] - Unsigned/Signed 8-bit interger
 * Description  : LUT Table for read and write.
 * Author       : lancerstadium
 * Date         : Sun Mar  2 18:12:13 CST 2025
 * License      : MIT
 */

/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off CASEINCOMPLETE */
module LUTTable_x4_4b_i8_s1_D_L2 (
    // 系统接口
    input  logic         clk,      // 全局时钟
    input  logic         rst_n,    // 异步复位
    // 批量读取控制
    input  logic         burst_start,  // 脉冲启动批量读取
    output logic         burst_done,   // 批量读取完成标志
    input  logic [5:0]    base_addr,  // 非展平基地址
    // 数据输出
    output logic signed [7:0] data [0:15]  // 并行输出
);

    (* rom_style = "block" *) 
    logic signed [7:0] lut_mem [0:575]
    =
    // Original Shape: (36, 16), Total Entries: 576
'{
    /* (0, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (1, 0) */	    8'sh00, 8'sh00, 8'shfe, 8'sh01, 8'shfb, 8'shff, 8'shff, 8'sh02, 8'sh01, 8'sh00, 8'shfb, 8'shff, 8'sh00, 8'shff, 8'sh00, 8'shff, 
    /* (2, 0) */	    8'sh00, 8'sh00, 8'shfd, 8'sh02, 8'shf6, 8'shff, 8'shfe, 8'sh05, 8'sh02, 8'shff, 8'shf7, 8'shfe, 8'shff, 8'shfe, 8'sh00, 8'shfe, 
    /* (3, 0) */	    8'sh00, 8'sh00, 8'shfb, 8'sh04, 8'shf1, 8'shfe, 8'shfd, 8'sh07, 8'sh03, 8'shff, 8'shf2, 8'shfd, 8'shff, 8'shfe, 8'shff, 8'shfd, 
    /* (4, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (5, 0) */	    8'shfd, 8'sh01, 8'shff, 8'sh00, 8'shfb, 8'sh01, 8'shfe, 8'sh01, 8'sh03, 8'sh00, 8'shfb, 8'shfe, 8'sh00, 8'shff, 8'sh00, 8'shfe, 
    /* (6, 0) */	    8'shfb, 8'sh01, 8'shfd, 8'sh00, 8'shf6, 8'sh01, 8'shfd, 8'sh02, 8'sh07, 8'shff, 8'shf5, 8'shfc, 8'sh01, 8'shfe, 8'sh00, 8'shfb, 
    /* (7, 0) */	    8'shf8, 8'sh02, 8'shfc, 8'sh00, 8'shf1, 8'sh02, 8'shfb, 8'sh03, 8'sh0a, 8'shff, 8'shf0, 8'shfa, 8'sh01, 8'shfd, 8'shff, 8'shf9, 
    /* (8, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (9, 0) */	    8'shfe, 8'shfd, 8'shfc, 8'sh00, 8'shfc, 8'shff, 8'shfe, 8'sh04, 8'sh05, 8'sh04, 8'shfb, 8'sh02, 8'sh00, 8'sh00, 8'sh00, 8'sh01, 
    /* (10, 0) */	    8'shfc, 8'shfa, 8'shf8, 8'sh00, 8'shf7, 8'shff, 8'shfd, 8'sh07, 8'sh09, 8'sh08, 8'shf6, 8'sh04, 8'shff, 8'sh00, 8'sh00, 8'sh01, 
    /* (11, 0) */	    8'shf9, 8'shf7, 8'shf5, 8'sh00, 8'shf3, 8'shfe, 8'shfb, 8'sh0b, 8'sh0e, 8'sh0b, 8'shf1, 8'sh05, 8'shff, 8'sh00, 8'sh01, 8'sh02, 
    /* (12, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (13, 0) */	    8'shfe, 8'sh01, 8'sh00, 8'sh05, 8'shfb, 8'sh00, 8'sh00, 8'sh04, 8'sh03, 8'shff, 8'shfb, 8'shfe, 8'sh00, 8'shfe, 8'sh00, 8'shfd, 
    /* (14, 0) */	    8'shfb, 8'sh02, 8'shff, 8'sh09, 8'shf7, 8'shff, 8'shff, 8'sh07, 8'sh07, 8'shfd, 8'shf7, 8'shfd, 8'sh01, 8'shfc, 8'shff, 8'shfa, 
    /* (15, 0) */	    8'shf9, 8'sh03, 8'shff, 8'sh0e, 8'shf2, 8'shff, 8'shff, 8'sh0b, 8'sh0a, 8'shfc, 8'shf2, 8'shfb, 8'sh01, 8'shfa, 8'shff, 8'shf7, 
    /* (16, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (17, 0) */	    8'sh09, 8'shff, 8'shfa, 8'shf6, 8'shfe, 8'shfe, 8'shfe, 8'sh01, 8'shfb, 8'sh02, 8'shfe, 8'sh00, 8'shf9, 8'shff, 8'sh02, 8'sh02, 
    /* (18, 0) */	    8'sh12, 8'shff, 8'shf4, 8'sheb, 8'shfb, 8'shfd, 8'shfd, 8'sh01, 8'shf5, 8'sh05, 8'shfb, 8'sh00, 8'shf3, 8'shfe, 8'sh03, 8'sh04, 
    /* (19, 0) */	    8'sh1a, 8'shfe, 8'shee, 8'she1, 8'shf9, 8'shfb, 8'shfb, 8'sh02, 8'shf0, 8'sh07, 8'shf9, 8'sh00, 8'shec, 8'shfc, 8'sh05, 8'sh05, 
    /* (20, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (21, 0) */	    8'sh03, 8'sh0b, 8'sh03, 8'sh0c, 8'shfa, 8'sh01, 8'shfd, 8'sh04, 8'shfd, 8'shf5, 8'shfd, 8'shf9, 8'sh01, 8'shf9, 8'shfe, 8'shfa, 
    /* (22, 0) */	    8'sh07, 8'sh16, 8'sh05, 8'sh18, 8'shf5, 8'sh02, 8'shfa, 8'sh08, 8'shfa, 8'shea, 8'shfa, 8'shf2, 8'sh01, 8'shf2, 8'shfc, 8'shf4, 
    /* (23, 0) */	    8'sh0a, 8'sh20, 8'sh08, 8'sh24, 8'shef, 8'sh02, 8'shf7, 8'sh0c, 8'shf7, 8'shdf, 8'shf7, 8'sheb, 8'sh02, 8'shea, 8'shfb, 8'shee, 
    /* (24, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (25, 0) */	    8'shfd, 8'sh00, 8'shff, 8'sh04, 8'shf9, 8'shfd, 8'sh01, 8'sh06, 8'shfb, 8'shff, 8'shfe, 8'sh02, 8'sh00, 8'shff, 8'sh01, 8'sh01, 
    /* (26, 0) */	    8'shfb, 8'sh00, 8'shfe, 8'sh08, 8'shf3, 8'shfa, 8'sh01, 8'sh0c, 8'shf6, 8'shff, 8'shfb, 8'sh03, 8'sh01, 8'shff, 8'sh03, 8'sh02, 
    /* (27, 0) */	    8'shf8, 8'sh01, 8'shfe, 8'sh0c, 8'shec, 8'shf6, 8'sh02, 8'sh12, 8'shf1, 8'shfe, 8'shf9, 8'sh05, 8'sh01, 8'shfe, 8'sh04, 8'sh03, 
    /* (28, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (29, 0) */	    8'sh04, 8'shff, 8'shfc, 8'shfa, 8'sh03, 8'sh0c, 8'shf8, 8'shf7, 8'sh0a, 8'sh02, 8'shf5, 8'shf7, 8'sh00, 8'shff, 8'shfe, 8'shf9, 
    /* (30, 0) */	    8'sh09, 8'shfe, 8'shf7, 8'shf3, 8'sh05, 8'sh18, 8'shf0, 8'shef, 8'sh14, 8'sh05, 8'shea, 8'shed, 8'sh00, 8'shfe, 8'shfc, 8'shf2, 
    /* (31, 0) */	    8'sh0d, 8'shfd, 8'shf3, 8'shed, 8'sh08, 8'sh25, 8'she8, 8'she6, 8'sh1f, 8'sh07, 8'shdf, 8'she4, 8'sh00, 8'shfd, 8'shfb, 8'shea, 
    /* (32, 0) */	    8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 8'sh00, 
    /* (33, 0) */	    8'shf8, 8'shf7, 8'shf5, 8'sh00, 8'shf7, 8'shf4, 8'shfb, 8'sh09, 8'shfc, 8'sh07, 8'shff, 8'shfd, 8'sh03, 8'shf8, 8'sh01, 8'shfc, 
    /* (34, 0) */	    8'shf0, 8'shef, 8'she9, 8'sh00, 8'shed, 8'she8, 8'shf6, 8'sh11, 8'shf9, 8'sh0d, 8'shfe, 8'shfb, 8'sh06, 8'shf0, 8'sh03, 8'shf9, 
    /* (35, 0) */	    8'she7, 8'she6, 8'shde, 8'shff, 8'she4, 8'shdc, 8'shf1, 8'sh1a, 8'shf5, 8'sh14, 8'shfd, 8'shf8, 8'sh08, 8'she7, 8'sh04, 8'shf5
};

    // 状态机增强版（防止锁死）
    typedef enum logic [1:0] {
        IDLE,
        ACTIVE,
        ERROR_STATE
    } burst_state_t;
    
    burst_state_t state;
    logic [5:0] counter;  // 增加安全位宽
    logic [9:0] addr_reg [0:15];

    // 地址有效性检查
    logic addr_valid;
    assign addr_valid = (base_addr * 16 + 16) <= 576;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            counter <= 0;
            burst_done <= 0;
            foreach(addr_reg[i]) addr_reg[i] <= 0;
        end else case (state)
            IDLE: begin
                burst_done <= 0;
                if (burst_start && addr_valid) begin
                    state <= ACTIVE;
                    counter <= 16;
                    foreach(addr_reg[i]) 
                        addr_reg[i] <= base_addr * 16 + i;
                end else if (burst_start) begin
                    state <= ERROR_STATE;  // 处理非法地址
                end
            end
            
            ACTIVE: begin
                counter <= counter - 1;
                foreach(addr_reg[i])
                    addr_reg[i] <= addr_reg[i] + 16;
                
                if (counter == 1) begin
                    state <= IDLE;
                    burst_done <= 1;
                end
            end
            
            ERROR_STATE: begin
                // 可扩展错误恢复逻辑
                state <= IDLE;
            end
        endcase
    end

    generate
        for (genvar i = 0; i < 16; i++) begin : pipeline
            always_ff @(posedge clk) begin
                if (state == ACTIVE) begin
                    data[i] <= lut_mem[addr_reg[i]];
                end else begin
                    data[i] <= '0;  // 非活动状态清零
                end
            end
        end
    endgenerate

endmodule
/* verilator lint_on CASEINCOMPLETE */
/* verilator lint_on WIDTHTRUNC */
