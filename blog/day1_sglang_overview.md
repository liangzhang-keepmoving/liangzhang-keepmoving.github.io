# SGLang（一）：整体认知

## 发展时间线

| 时间       | 里程碑                                                                 |
|------------|------------------------------------------------------------------------|
| 2024/01    | SGLang 发布，引入 RadixAttention 实现最高 5 倍推理速度提升             |
| 2024/02    | 推出压缩有限状态机，实现 3 倍更快的 JSON 解码                          |
| 2024/07    | v0.2 发布：针对 Llama3 的优化，性能优于 TensorRT-LLM 和 vLLM          |
| 2024/09    | v0.3 发布：DeepSeek MLA 速度提升 7 倍，torch.compile 速度提升 1.5 倍  |
| 2024/10    | 首届 SGLang 在线技术交流会                                              |
| 2024/12    | v0.4 发布：零开销批处理调度器、缓存感知负载均衡器、更快的结构化输出    |
| 2025/01    | 为 DeepSeek V3/R1 模型提供首日支持                                     |
| 2025/05    | 在 96 个 H100 GPU 上部署带有 PD 分解和大规模专家并行的 DeepSeek        |
| 2025/08    | 为 OpenAI gpt-oss 模型提供首日支持                                     |
| 2025/10    | SGLang-Jax 后端发布，原生支持 TPU                                      |
| 2025/11    | SGLang Diffusion 发布，加速视频和图像生成                              |
| 2025/12    | 为最新开源模型（MiMo-V2-Flash、Nemotron 3 Nano 等）提供首日支持         |

## 核心功能

### 高效运行时

SGLang 的核心优势在于其高效的运行时设计，通过多种优化技术显著提升模型推理性能：

- **RadixAttention**：基于基数树的前缀缓存机制，避免重复计算相同前缀的注意力
- **零开销批处理调度器**：优化批处理调度算法，减少调度开销
- **预填充-解码分解**：将模型推理分为预填充和解码两个阶段，分别优化
- **推测解码**：通过预测可能的输出加速生成过程
- **连续批处理**：动态合并请求，提高硬件利用率
- **分页注意力**：高效管理 KV 缓存，减少内存占用
- **张量/流水线/专家/数据并行**：支持多种并行计算策略
- **结构化输出**：优化 JSON 等结构化数据的生成
- **分块预填充**：处理长输入的高效策略
- **量化**：支持 FP4/FP8/INT4/AWQ/GPTQ 等多种量化方式
- **多 LoRA 批处理**：同时处理多个 LoRA 适配器

### 广泛的模型支持

SGLang 支持多种类型的模型，为不同应用场景提供灵活选择：

- **语言模型**：Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等
- **嵌入模型**：e5-mistral、gte、mcdse 等
- **奖励模型**：Skywork 等
- **扩散模型**：WAN、Qwen-Image 等
- **兼容性**：支持大多数 Hugging Face 模型和 OpenAI API

### 丰富的硬件支持

SGLang 为多种硬件平台提供优化支持，确保在不同环境下都能发挥最佳性能：

- **NVIDIA GPU**：GB200/B300/H100/A100/Spark 等
- **AMD GPU**：MI355/MI300 等
- **Intel Xeon CPU**
- **Google TPU**
- **Ascend NPU**
- **其他硬件平台**

## 整体架构

SGLang 采用分层架构设计，各层职责明确，协同工作：

### 1. 前端

前端提供多种接口方式，方便用户与系统交互：

- **OpenAI 兼容 API**：提供与 OpenAI API 兼容的接口，便于迁移现有应用
- **命令行界面（CLI）**：支持通过命令行快速启动和配置服务
- **Python SDK**：提供编程接口，方便集成到 Python 应用中

### 2. 中间层

中间层负责请求处理和资源管理：

- **SGLang Model Gateway**：模型路由网关，管理工作节点生命周期，平衡流量
- **负载均衡器**：智能分配请求，优化资源利用
- **缓存管理器**：管理 KV 缓存，提高推理效率

### 3. 后端

后端是 SGLang 的核心，负责模型推理和优化：

- **运行时核心（SRT）**：实现核心推理逻辑和优化策略
- **硬件后端适配层**：为不同硬件平台提供适配和优化
- **模型优化器**：对模型进行编译和优化

### 4. 底层

底层提供基础功能支持：

- **sgl-kernel**：优化的内核实现，提供高性能算子
- **内存管理**：高效管理内存资源，减少内存开销
- **并行计算框架**：支持多种并行计算策略

## 行业影响力

SGLang 自发布以来，已在行业内产生广泛影响：

- **广泛采用**：被 xAI、AMD、NVIDIA、Intel、LinkedIn、Cursor 等公司部署使用，运行在超过 400,000 个 GPU 上
- **学术支持**：由非营利组织 LMSYS 托管，得到学术社区的支持
- **生态系统**：成为行业标准的 LLM 推理引擎，拥有活跃的开发者社区
- **企业级应用**：为众多科技公司提供技术支持，处理每天数万亿 tokens 的推理请求

## 代码示例

### 1. 基本服务启动

以下是使用 SGLang 启动基本服务的示例代码：

```python
# 使用 Python SDK 启动服务
from sglang import Sglang

# 初始化服务
server = Sglang(
    model="meta-llama/Llama-2-7b-chat-hf",
    port=8000,
    host="0.0.0.0"
)

# 启动服务
server.start()
```

### 2. 使用命令行启动服务

```bash
# 使用 CLI 启动服务
python -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --host 0.0.0.0
```

### 3. 发送请求示例

```python
# 使用 Python SDK 发送请求
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

print(response.json())
```

### 4. 使用 OpenAI 兼容 API

```python
# 使用 OpenAI SDK 发送请求
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy_key"  # SGLang 不需要真实 API 密钥
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 关键文件和目录结构

SGLang 项目的主要文件和目录结构如下：

```
sglang/
├── python/              # Python 代码
│   ├── sglang/          # 主源码
│   ├── pyproject.toml   # 项目配置
│   └── setup.py         # 安装脚本
├── sgl-kernel/          # 优化内核
├── sgl-model-gateway/   # 模型网关
├── docs/                # 文档
├── test/                # 测试代码
├── benchmark/           # 基准测试
└── scripts/             # 工具脚本
```

### 核心模块

- **sglang/srt/**：SGLang 运行时核心（SRT - SGLang Runtime）
- **sglang/lang/**：前端语言和 API
- **sglang/jit_kernel/**：即时编译内核
- **sglang/multimodal_gen/**：多模态生成

## 总结

SGLang 是一个高性能的 LLM 和多模态模型服务框架，通过创新的优化技术和灵活的架构设计，实现了低延迟、高吞吐量的模型推理。其广泛的模型支持和硬件适配能力，使其成为从单 GPU 到大规模分布式集群的理想选择。

SGLang 的发展速度非常快，持续引入新特性和优化，已成为行业标准的 LLM 推理引擎之一。通过本课程的学习，我们将深入了解 SGLang 的技术细节和应用方法，掌握这一强大工具的使用技巧。
