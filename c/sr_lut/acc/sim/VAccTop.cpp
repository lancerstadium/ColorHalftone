#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "verilated_vcd_c.h" // 生成vcd文件使用
#include "VAccTop.h"
#include "verilated.h"

VerilatedContext* contextp = NULL;
VAccTop* top = NULL;
VerilatedVcdC* ftp = NULL;

// 1个时间步长
void single_cycle() {
    top->clk = 0;
    top->eval();  // 更新状态信号
    contextp->timeInc(1); // 时间+1，推动仿真时间
    ftp->dump(contextp->time()); // dump wave
    top->clk = 1;
    top->eval();  // 更新状态信号
    contextp->timeInc(1); // 时间+1，推动仿真时间
    ftp->dump(contextp->time()); // dump wave
}
//
static void reset(int n) {
    top->rst_n = 0;
    while (n-- > 0) single_cycle();
    top->rst_n = 1;
}

int main (int argc, char **argv) {
    if (false && argc && argv) {}
    contextp = new VerilatedContext;    //建立仿真对象
    contextp->commandArgs(argc, argv);  //传递命令行参数
    top = new VAccTop{contextp};           //创建Verilator实例
    contextp->traceEverOn(true);        // 生成波形文件使用，打开追踪功能
    ftp = new VerilatedVcdC; // vcd对象指针
    top->trace(ftp, 5); // 2层
    ftp->open("wave.vcd"); //设置输出的文件wave.vcd
    int flag = 0;

    reset(10); // 复位信号
    
    
    while (!contextp->gotFinish() && ++flag < 200) {

        // if(flag == 5) { // write data
        //     top->a = 1;
        //     ((int32_t*)(top->wdata))[0] = 1;
        //     ((int32_t*)(top->wdata))[1] = 2;
        //     ((int32_t*)(top->wdata))[2] = 3;
        //     ((int32_t*)(top->wdata))[3] = 4;
        // }

        if (flag == 10 || flag == 20) { // read data
            top->b = 1;
        }

        // if(flag == 15) { // write data
        //     top->a = 1;
        //     ((int32_t*)(top->wdata))[0] = 7;
        //     ((int32_t*)(top->wdata))[1] = 8;
        //     ((int32_t*)(top->wdata))[2] = 9;
        //     ((int32_t*)(top->wdata))[3] = 10;
        // }

        
        single_cycle();

        // printf("a = %d, b = %d\n", a, top->b);
        // printf("a = %d, b = %d\n", a, ((int8_t)((u_int8_t)top->b << 2) >> 2));        // Clamp s6
        // printf("a = %d, b = %d\n", (int8_t)(a & 0xFF), top->b - 128);    // Cast s2u
        // printf("a = %d, b = %d\n", a, (int8_t)(top->b & 0xFF) + 128);    // Cast u2s
        // assert(top->f == (a ^ b));
        
        // if(flag == 5 || flag == 15) {
        //     top->a = 0;
        // }

        if (flag == 10 || flag == 20) { // read data
            printf("burst done\n");
            // read data array
            int32_t *rdata = (int32_t *)top->rdata;
            for(int i = 0; i < 4; i++) {
                printf("%d,", rdata[i]);
            }
            top->b = 0;
            // break;
        }
    }

    top->final();
    ftp->close(); // 必须有
    return 0;
}