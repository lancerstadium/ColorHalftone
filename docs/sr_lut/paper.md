

<div align="center">
<h2>
<i>WLLUT</i>: Weightless Lookup Table for Efficient Image Super-resolution on RISC-V-based Edge Device
</h2>
</div>

<div align="center"> 
<a href="">TianShuo Lu</a><sup>1</sup>, XXX<sup>1*</sup>, XXX<sup>1*†</sup>
</div>

<p align="center"> <sup>1</sup>JiangNan University, <sup>2</sup>XXX </p>

<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/lancerstadium/evo/tree/ml"  alt="github">
    <img src="https://img.shields.io/badge/github-WLLab-blue"/></a>
<a href="https://github.com/lancerstadium/evo/blob/ml/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-MIT-%23B7A800" /></a>
</p>

---

<b>Overview:</b> 
<div style="text-align: justify;">
...
</div>

<b>Keywords:</b> <i>look-up table, RISC-V</i>


---

### 0 Abstract

现有的基于深度神经网络（DNN）的超分辨率（SR）方法密集的使用了卷积。其计算与存储资源的大量需求与边缘推理（Edge Inference）的机制（regime）相矛盾。最近的研究将查找表（LUT）与DNN相结合，以减少边缘推理的资源占用。然而，现有的XXXXXX。为了解决这些问题，我们通过XXX（称为WLLUT）来XXX。（细节...）。与现有的LUT方案相比，WLLUT具有优秀的推理精度和高效的硬件性能。其在五个流行的基准数据集上取得了优异的性能。代码见：。

---

### 1 Introduction



---

### 2 Related Work

#### 2.1 Development of Super-Resolution

近年来，人们对边缘设备上高质量 SR 的追求激发了对高效 SISR 方案的兴趣和研究，这些方案大致分为传统方法、深度学习方法和基于 LUT 的方法。

传统的基于插值的超分辨率方法，如：Nearest, Bilinear 和 Bicubic，虽然效率很高，但由于忽略了图像细节，结果往往比较模糊。基于稀疏编码的方法通过学习到稀疏字典将低分辨率图像复原成高分辨率图像。经过运行成本大幅增加，但其结果优于基于插值的方法。

基于深度学习的 SR 方法取得了巨大进步，SRCNN、VDSR、EDSR 和 RCAN 等架构都以提高性能为目标。然而，这些方法需要大量的计算资源。为了缓解这一问题，ESPCN 和 FSRCNN 采用了更小的网络。其他策略包括减少参数、稀疏化或量化，在保持性能的同时减少计算需求。不过，对于计算资源有限的设备（如移动设备）来说，这些模型仍然过于繁重。

随着边缘推理的需求日益增加，越来越多的研究关注基于 LUT 的方法。Jo 和 Kim 提出了一种在 SR 中使用 LUT 的方法（SR-LUT）[[1]](#ref_01)。他们训练了一个具有有限接收场（RF）的简单深度 SR 网络，并将输入（作为索引）和网络输出（作为索引的位置和作为值的像素强度）传输到 LUT。LUT 用于在扫描过程中生成最终结果。由于不需要额外的计算，SR-LUT 在移动设备上的运行速度与基于插值法的 SR 方法一样快，即几十毫秒。由于 LUT 的大小与索引容量成指数增长，因此 SR-LUT 中的 RF 限制为 3 × 3。然而，文献证明 RF 的大小是至关重要的，SR LUT 自然会获得较差的性能。Li 等人提出了 MuLUT，通过引入手工制作的索引模式和级联 LUT 来增加射频。通过这两种方案，MuLUT 与 SR-LUT 相比取得了显著的改进，但代价是 LUT 总大小的线性增长。同样，Ma 等人也采用级联 LUT 的思路来扩大射频。他们提出了串并联 LUT (SPLUT) 框架，该框架引入了通道级 LUT，并并行处理从原始 8 位输入中分离出来的两个分量。这两种方法将 SR-LUT 的单 LUT 转换为多 LUT，从而提高了 SR 模型的射频。


|   Model   |   
|:---------:|
|   SP-LUT  |
|   SR-LUT  |
|   MuLUT   |
|   RC-LUT  |

#### 2.2 




---

### 3 Method1



---

### 4 Method2



---

### 5 Evaluation


---

### 6 Conclusion


---

<b>Reference:</b> 
<div style="text-align: justify;">
<a id="ref_01">[1]</a> Practical Single-Image Super-Resolution Using Look-Up Table
</div>