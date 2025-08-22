# CUDA向量加法示例详解

## 核心概念解析

本示例展示了CUDA编程的基本模式和核心概念，这是GPU编程的基础。下面详细解释代码中的关键部分：

### 1. CUDA核函数

```cuda
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
```

![vectorAdd核函数解析](images/vector_add_kernel.svg)

- `__global__` 关键字：表示这是一个可以从CPU调用并在GPU上执行的函数
- 线程层次结构：通过`blockIdx.x`（块索引）和`threadIdx.x`（线程索引）来确定每个线程处理的数据位置
- `blockDim.x`：每个线程块中的线程数
- 边界检查：确保线程不会访问超出数组范围的内存

### 2. 内存管理

CUDA编程中，内存管理是一个关键环节，通常遵循以下步骤：

![CUDA内存管理流程](images/cuda_memory_management.svg)

1. **分配主机内存**：使用标准C++容器或内存分配函数
2. **分配设备内存**：使用`cudaMalloc`函数
3. **数据传输**：
   - 主机到设备：`cudaMemcpyHostToDevice`
   - 设备到主机：`cudaMemcpyDeviceToHost`
4. **释放内存**：使用`cudaFree`释放设备内存

### 3. 线程配置

![CUDA线程层次结构](images/cuda_thread_hierarchy.svg)

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

- **线程块大小**：通常选择256、512等，取决于GPU架构和计算需求
- **网格大小计算**：确保所有元素都能被处理，向上取整
- **执行配置语法**：`<<<blocksPerGrid, threadsPerBlock>>>`指定了核函数的执行配置

### 4. 同步与错误检查

![CUDA向量加法执行流程](images/vector_add_execution.svg)

```cuda
cudaDeviceSynchronize();
cudaError_t error = cudaGetLastError();
```

- `cudaDeviceSynchronize()`：确保GPU上的所有操作都已完成，用于精确计时
- 错误检查：使用`cudaGetLastError()`检查核函数启动时可能发生的错误

## 代码优化建议

1. **添加异步数据传输**：使用CUDA流重叠计算和数据传输

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步数据传输
cudaMemcpyAsync(d_A, h_A.data(), size, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_B, h_B.data(), size, cudaMemcpyHostToDevice, stream);

// 在流中启动核函数
vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements);

// 异步复制结果
cudaMemcpyAsync(h_C.data(), d_C, size, cudaMemcpyDeviceToHost, stream);

// 等待流完成
cudaStreamSynchronize(stream);
```

2. **使用统一内存**：简化内存管理

```cuda
float* u_A = NULL;
float* u_B = NULL;
float* u_C = NULL;
cudaMallocManaged(&u_A, size);
cudaMallocManaged(&u_B, size);
cudaMallocManaged(&u_C, size);

// 直接在主机上初始化数据
for (int i = 0; i < numElements; ++i)
{
    u_A[i] = rand() / (float)RAND_MAX;
    u_B[i] = rand() / (float)RAND_MAX;
}

// 核函数执行
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(u_A, u_B, u_C, numElements);

// 确保结果对主机可见
cudaDeviceSynchronize();

// 释放统一内存
cudaFree(u_A);
cudaFree(u_B);
cudaFree(u_C);
```

3. **添加性能测量**：对比不同优化策略的效果

通过这些优化，可以显著提高GPU程序的性能。