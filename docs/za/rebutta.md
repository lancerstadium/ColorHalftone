

### Comment 1

Thank you for your insightful question. We understand your concerns regarding the integration of Section III.B with the rest of the framework. Below is a detailed explanation.
1. **The purpose of Section III.B and its relation to the framework**: Section III.B primarily focuses on the optimization strategies for the low-bit convolution operation. This section discusses how to execute "QConv" operators efficiently, which is a key part of the framework shown in Figure 1. In this part of the framework, we have implemented the C version of operators to allow easier hardware integration, and we customize the computation flow through inline assembly instructions, as demonstrated in Figure 7.
2. **Compile-time operations**: During compilation, the toolchain (as shown in Figure 8) identifies the low-bit convolution operators and integrates them with the weights and the forward inference process. These are then compiled into a new program entry file. The RISC-V toolchain cross-compiles these files into binary instructions suitable for hardware execution. The optimization during this phase takes into account the specific hardware platform to ensure optimal performance.
3. **Runtime operations**: At runtime, coarse-grained data operations, such as tiling, Im2col, and data packing, are dynamically executed on the software side. Specifically, the Im2col operation occurs before data packing, where the convolution input is transformed into a packed format for efficient SIMD dot product with low-bitwidth hardware. In contrast, finer-grained operations, such as multiplication and accumulation of the packed data, are handled by dedicated hardware execution units to ensure efficiency.
4. **Im2col’s role**: The Im2col operation is performed at runtime after tiling phase. Traditional Im2col flattens all input data for computation. Our approach applies dynamic flattening to local computation data, as opposed to flattening all data at once. This requires designing different computational flows and data buffers for different convolution types. By doing so, we reduce redundant data movement and improve computation efficiency, especially for low-bit computations.

---

### Comment 2

We appreciate your suggestion on the usability of the code and datasets used in our research. We place great importance on transparency and reproducibility in our research. If the paper is accepted, we will make relative data and source code available on GitHub, ensuring that the community can independently verify and reproduce our results.


---

### Comment 3


Thank you very much for your insightful question. We are pleased to provide a detailed explanation regarding the models used in the experiments presented in Table II.
1. **Selection of the experimental models**: The results shown in Table II come from several neural network models, particularly those with computationally intensive layers and low-bit weightss. We specifically selected these models to highlight the impact of low-bit quantization on various layers in a neural network, especially in convolutional and fully connected layers.
2. **Focus on computationally intensive layers**: The primary focus of our evaluation is on computationally intensive layers, such as Convolution and General Matrix Multiply (GEMM). These layers typically require a significant amount of computational resources, and low-bit quantization significantly accelerates their computation. The impact of low-bit quantization is most pronounced in these layers, where it reduces both computation and memory overhead.
3. **Comprehensive evaluation of quantization effects**: Our evaluation covers a range of different layers and models to ensure a comprehensive assessment of low-bit quantization. We evaluated the performance across various architectures and types of layers, with a particular emphasis on how low-bit quantization affects computationally intensive layers. This broad evaluation ensures that the observed benefits of low-bit quantization are not limited to specific layers but can be generalized to different network structures.

---

### Comment 4


Thank you for your question about framework's performance. Your understanding is generally correct, but we would like to provide further clarifications and additional details.
1. **Performance improvement and calculation basis**: Indeed, the overall optimizations lead to about a 1.5x improvement in execution time. However, it is important to clarify that this performance improvement is based on cycle counts, which may vary depending on the synthesis and hardware configurations. Therefore, the actual performance gains might differ when deployed on different hardware platforms.
2. **Information of cv32e40p**: The original cv32e40p core does not have dedicated hardware units for low-bit data acceleration, resulting in lower energy efficiency for low-bit computations. However, the original core already performs well in terms of power efficiency, especially given its focus on lightweight and efficient operation. Internet research and various benchmarks confirm that the base power efficiency of cv32e40p is among the best in class, especially for general-purpose processors targeting embedded applications. XPULPNN, on the other hand, improves energy efficiency by using a cluster of 8 cv32e40p cores combined with specialized low-bit-width execution units. This architecture enables XPULPNN to achieve an energy efficiency of 1111 GOPS/W, significantly improving performance for low-bit data computations.a
3. **Our framework’s optimization strategy**: Unlike XPULPNN, which relies on a large core cluster, our framework does not depend on such clusters. Instead, we achieve energy efficiency improvements through a co-design of hardware asnd software. Specifically, on the software side, we optimize data flow and packing operations to improve data reuse and movement. On the hardware side, we enhance the processing of low-bit data using custom SIMD instruction extensions and adaptive computation units. This approach allows us to achieve similar energy efficiency gains without relying on a large core cluster, while also enhancing hardware flexibility and scalability.

---
