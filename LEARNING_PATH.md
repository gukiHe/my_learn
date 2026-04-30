# MiniMind 学习路线图

从零开始，通过阅读和运行 MiniMind 的代码，系统性理解 LLM 的架构、训练全流程与强化学习算法。

---

## 阶段一：理解 Tokenizer 与 Embedding（第 1-2 天）

### 学习目标
- 理解 Tokenizer 如何把自然语言映射成 token ID
- 理解 Embedding 层的角色
- 理解为什么 MiniMind 只用了 6400 词表

### 需要阅读的文件

| 文件 | 行数 | 重点 |
|------|------|------|
| `model/tokenizer.json` | 大文件 | 浏览结构，了解 BPE merge 规则 |
| `model/tokenizer_config.json` | 小配置 | vocab_size=6400, BPE+ByteLevel |

### 核心概念

```
文本 "hello world"
     ↓ tokenizer.encode()
tokens [1523, 892]
     ↓ model.embed_tokens()
vectors [768-dim, 768-dim]
```

### 思考题
1. 为什么小模型要用小词表？（提示：Embedding 层占总参数的比例）
回答：词表越大，embedding层占用的参数量越大，训练/推理的时候占用的显存越大。6400 应该是实践中找到的平衡点.
Embedding 占用的内容不过多，Transformer的结构占用的会多。
2. BPE 和 WordPiece 的区别是什么？
回答：BPE是找到最频繁的字符对，仅关注频率。WordPice找的是最大似然对数，可能保存更多的语义？实际的应用中差距不大

### 动手练习
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./model')
print(tok.encode("你好世界"))          # 看 token ID
print(tok.decode([1523, 892]))        # 看回译效果
print(len(tok.vocab))                  # 确认 6400
```
已完成

### 延伸阅读
- 《Attention Is All You Need》Section 5.1 (Subword Tokenization)
- [Byte-Pair Encoding 论文](https://www.aclweb.org/anthology/P16-1162/)

---

## 阶段二：手撕 Transformer 架构（第 3-7 天）

### 学习目标
- 逐行理解 `model_minimind.py` 里的 280 行代码
- 理解 RMSNorm、RoPE、GQA、SwiGLU 四大组件
- 能画出完整的 forward pass 数据流图

### 需要阅读的文件

| 文件 | 行数 | 重点 |
|------|------|------|
| `model/model_minimind.py` | 280 | 核心，反复读 3 遍 |
| `model/model_lora.py` | 65 | LoRA 的矩阵分解思路 |

### 逐行拆解指南

#### 2.1 RMSNorm (L49-59)

```python
# 对比 LayerNorm: RMSNorm 不减均值，直接除均方根
# 公式: x / sqrt(mean(x^2) + eps) * weight
# 为什么用 RMSNorm 而不是 LayerNorm? → 训练更稳定，计算量更小
```

**阅读顺序：** L49 → L51-L59

#### 2.2 RoPE 旋转位置编码 (L61-83)

```python
# precompute_freqs_cis  (L61-77)
#   → 生成 cos/sin 频率表，提前算好存 buffer
# YaRN 外推逻辑        (L63-72)
#   → 对超出训练长度的位置做线性插值衰减
# apply_rotary_pos_emb (L79-83)
#   → q 和 k 各自 * cos + rotate_half(q/k) * sin
```

**阅读顺序：** L61 → L62 理解 freqs 公式 → L63-72 理解 YaRN ramp → L79-83

**关键理解：**
- `rotate_half` 是怎么把向量切成两半然后交换的？
- 为什么 cos 和 sin 在 L75-76 要 `[...cat(...), ...cat(...)]` 拼接两次？

#### 2.3 GQA 与 Attention (L85-133)

```python
# repeat_kv           (L85-88)     → KV heads=4 扩到 Q heads=8
# Attention.__init__  (L91-108)    → q_proj, k_proj, v_proj, o_proj
# Attention.forward   (L110-133)   → Q/K/V 投影 → q/k norm → RoPE → SDPA/softmax
```

**阅读顺序：** L91-92 (构造函数) → L99-102 (GQA: q_heads=8, kv_heads=4) → L103-104 (q_norm, k_norm) → L110-133 (forward 完整流程)

**关键理解：**
- 为什么 `q_heads=8, kv_heads=4`？这叫 Grouped Query Attention (GQA)，节省显存同时保留大部分效果
- Flash Attention 的 fallback 路径 (L124-130)：什么时候用 `scaled_dot_product_attention`，什么时候退回到手动 softmax？
- 因果掩码 (causal mask) 在 L128 是怎么用 `triu(1)` 实现的？

#### 2.4 SwiGLU FeedForward (L135-145)

```python
# gate_proj → x * W_gate  →  SwiGLU(act)
# up_proj   → x * W_up    →  直接线性
# 两者 element-wise 相乘 → down_proj 投影回 hidden_size
```

**阅读顺序：** L136-145

**关键理解：**
- 为什么 intermediate_size = `ceil(hidden_size * π / 64) * 64`？（这是 SwiGLU 的最优维度公式）

#### 2.5 MoE 路由 (L147-175)

```python
# gate 线性层 → softmax → topk 选专家 → 加权求和
# aux_loss 用于负载均衡，防止所有 token 都走同一个专家
```

**阅读顺序：** L148-152 (结构) → L155-167 (forward) → L169-174 (aux_loss)

**关键理解：**
- L168-169 那个 `y[0, 0] += 0 * ...` 在训练时是做什么的？
- 为什么 MiniMind 去掉了 shared expert？

#### 2.6 完整模型组装 (L177-246)

```python
# MiniMindBlock (L177-193)
#   → Pre-Norm: input_layernorm → Attention → residual → post_attention_layernorm → MLP
# MiniMindModel (L195-227)
#   → embed → 8 layers → norm → return hidden_states + past_kvs + aux_loss
# MiniMindForCausalLM (L229-246)
#   → lm_head 投影到 vocab_size → cross_entropy loss
```

**阅读顺序：** L177 → L195 → L229

### 必须画出的架构图

```
input_ids (B, seq_len)
    ↓
embed_tokens (B, seq_len, 768)
    ↓
┌─ Layer 0 ──────────────────────────────────┐
│  RMSNorm → Attention → +residual           │
│  RMSNorm → SwiGLU/MoE → +residual          │
└─────────────────────────────────────────────┘
    ↓ ... (重复 8 层)
RMSNorm
    ↓
lm_head (B, seq_len, 6400)
    ↓
CrossEntropyLoss
```

### 思考题
1. Pre-Norm 和 Post-Norm 的区别是什么？为什么选 Pre-Norm？
2. 为什么 `generate()` 要继承 `GenerationMixin` 而不是自己写推理循环？
3. LoRA 的 `A * B` 为什么要这样初始化：A 正态分布，B 全零？

### 延伸阅读
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202)

---

## 阶段三：理解训练基础设施（第 8-10 天）

### 学习目标
- 理解 DDP 多卡训练、混合精度、梯度累积、Cosine LR 调度、Checkpoint 恢复

### 需要阅读的文件

| 文件 | 行数 | 重点 |
|------|------|------|
| `trainer/trainer_utils.py` | 177 | 工具函数全集，必须逐行读 |
| `trainer/train_pretrain.py` | 170 | 最干净的训练脚本，是其他脚本的模板 |
| `dataset/lm_dataset.py` | 看内容 | PretrainDataset 的数据加载逻辑 |

### 逐行拆解

#### 3.1 DDP 初始化 (trainer_utils.py L44-51)

```python
# init_distributed_mode()
#   → 检测 RANK 环境变量 → nccl backend → set_device
#   → 返回 local_rank
```

#### 3.2 Cosine LR 调度 (trainer_utils.py L40-41)

```python
lr = lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
# warmup + cosine decay
```

#### 3.3 Checkpoint 原子写入 (trainer_utils.py L63-116)

```python
# 先写 .tmp → os.replace() 原子替换 → 防止中断产生损坏文件
# 跨 GPU 数量恢复 (L110-114): step * saved_ws // current_ws
```

#### 3.4 混合精度训练 (train_pretrain.py L34, L39)

```python
with autocast_ctx:          # forward 用 bfloat16
    res = model(input_ids, labels=labels)
scaler.scale(loss).backward()  # backward 自动 unscale 回 fp32
```

#### 3.5 梯度累积 (train_pretrain.py L37, L41-48)

```python
loss = loss / args.accumulation_steps   # 先除 accumulation_steps
if step % accumulation_steps == 0:      # 每 N 步才真正 optimizer.step()
```

#### 3.6 保存逻辑 (train_pretrain.py L60-69)

```python
raw_model = model.module if DDP else model    # 解壳
raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 解 compile
state_dict = {k: v.half().cpu() for ...}     # 转 half + CPU 存储
```

### 思考题
1. 为什么保存权重时用 `half().cpu()`，但 resume 时需要完整优化器状态？
2. `SkipBatchSampler` 是用来干什么的？
3. 为什么 DDP 要 ignore `freqs_cos` 和 `freqs_sin`？

---

## 阶段四：预训练 + SFT（第 11-15 天）

### 学习目标
- 跑通完整训练流程
- 理解 Pretrain（next token prediction）和 SFT（指令跟随）的区别
- 学会看 loss 曲线判断训练健康度

### 需要阅读的文件

| 文件 | 重点 |
|------|------|
| `trainer/train_pretrain.py` | 已读过，现在关注数据流 |
| `trainer/train_full_sft.py` | 对比 pretrain，找差异 |

### 对比 Pretrain vs SFT

| 维度 | Pretrain | SFT |
|------|----------|-----|
| 损失函数 | CrossEntropy (next token) | CrossEntropy (assistant response only) |
| 数据格式 | `{"text": "..."}` | `{"conversations": [...]}` |
| Label mask | 全部 token 都学 | 只学 assistant 部分 (role != user) |
| 目标 | 学会语言基础 | 学会跟随指令 |

### 动手实践

```bash
# Step 1: 预训练
cd trainer && python train_pretrain.py

# Step 2: 测试预训练效果
python eval_llm.py --weight pretrain
# → 观察：模型会"续写"但不一定能回答问题

# Step 3: SFT 指令微调
cd trainer && python train_full_sft.py

# Step 4: 测试 SFT 效果
python eval_llm.py --weight full_sft
# → 观察：现在能理解和回答问题了
```

### Loss 曲线判读

| 现象 | 含义 |
|------|------|
| Loss 稳定下降 | 训练健康 |
| Loss 突然暴涨 | 学习率过大、梯度爆炸 |
| Loss 平台不动 | 学习率太小、数据问题 |
| Train Loss 远 < Val Loss | 过拟合 |

### 延伸阅读
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)（MiniMind 的灵感来源）

---

## 阶段五：LoRA 低秩适配（第 16-18 天）

### 学习目标
- 理解 LoRA 的数学直觉
- 能手写一个 LoRA Layer
- 理解为什么 LoRA 权重可以合并回基座

### 需要阅读的文件

| 文件 | 行数 | 重点 |
|------|------|------|
| `model/model_lora.py` | 65 | 全文精读 |
| `trainer/train_lora.py` | 看内容 | LoRA 训练脚本 |

### 核心数学

```
原权重:     W ∈ R(d×d)
LoRA 更新:  ΔW = B × A,  A ∈ R(r×d), B ∈ R(d×r), r << d
新输出:     y = W·x + B·(A·x)

初始化: A ~ N(0, 0.02), B = 0  → 初始时 ΔW = 0，保证从 0 开始微调
```

### 代码对应

```python
# L6-18: LoRA class
A = Linear(in, rank, bias=False)    # 矩阵 A
B = Linear(rank, out, bias=False)   # 矩阵 B
forward: return B(A(x))             # B * A * x

# L21-32: apply_lora
#   → 遍历所有 nn.Linear (方形权重)，挂载 LoRA
#   → forward_with_lora: original(x) + lora(x)

# L56-65: merge_lora
#   → W_merged = W_original + B @ A
#   → 这样推理时就不需要额外的 LoRA 分支了
```

### 动手实践

```bash
# 训练一个医疗 LoRA
cd trainer && python train_lora.py

# 加载基座 + LoRA 做推理
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### 思考题
1. 为什么只对 `nn.Linear`（方形权重）挂 LoRA？embedding 和 norm 呢？
2. rank=16 时，一个 768×768 层的参数量减少了多少倍？
3. merge_lora 之后模型还能再做 LoRA 微调吗？

### 延伸阅读
- [LoRA 原论文](https://arxiv.org/abs/2106.09685)

---

## 阶段六：RLHF — DPO（第 19-21 天）

### 学习目标
- 理解偏好对齐的直觉
- 从代码理解 DPO loss 的简化推导
- 对比 DPO vs PPO：为什么 DPO 更稳定但能力上限更低？

### 需要阅读的文件

| 文件 | 重点 |
|------|------|
| `trainer/train_dpo.py` | DPO 损失实现 |

### DPO Loss 公式

```
L_DPO = -log(σ(β * [log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))]))

chosen vs rejected: 最大化 chosen 的概率，最小化 rejected 的概率
```

### 代码中找对应实现
- Chosen/Rejected 的 log probability 怎么计算的？
- β 系数在哪里？
- ref model 是怎么冻结的？

### 动手实践

```bash
# 训练 DPO
cd trainer && python train_dpo.py
```

### 思考题
1. DPO 不需要 Reward Model，为什么？
2. DPO 是 on-policy 还是 off-policy？

### 延伸阅读
- [DPO 原论文](https://arxiv.org/abs/2305.18290)

---

## 阶段七：RLAIF — PPO, GRPO, CISPO（第 22-30 天）

### 学习目标（最核心阶段）
- 理解强化学习在 LLM 中的统一视角
- 逐个对比 PPO → GRPO → CISPO 的三个组件差异
- 亲手跑一个 RLAIF 训练

### 需要阅读的文件

| 文件 | 重点 |
|------|------|
| `trainer/train_ppo.py` | Actor + Critic, GAE |
| `trainer/train_grpo.py` | Group Relative Policy Optimization |
| `trainer/train_agent.py` | Agentic RL, 多轮 Tool-Use |
| `trainer/rollout_engine.py` | 训推分离架构 |
| `trainer/trainer_utils.py L160-177` | Reward Model 封装 |

### 统一框架：所有 PO 算法对比

| 组件 | DPO | PPO | GRPO | CISPO |
|------|-----|-----|------|-------|
| **策略项** | log(r_w) - log(r_l) | min(r, clip(r)) | min(r, clip(r)) | min(r, ε_max) · log(π) |
| **优势项** | 无 | R - V(s) | (R-μ)/σ | (R-μ)/σ |
| **正则项** | 隐含 β | β·E[KL] | β·KL | β·KL |
| **网络数** | 1(+ref) | 2 (Actor+Critic) | 1 | 1 |
| **Policy** | Off-policy | On-policy | On-policy | On-policy |

### 逐文件拆解

#### 7.1 PPO (train_ppo.py)

**关键概念：**
- Actor 网络：生成回答的策略 π(a|s)
- Critic 网络：估计状态价值 V(s)
- GAE (Generalized Advantage Estimation)：优势 A = R - V
- Clip: min(r·A, clip(r, 1-ε, 1+ε)·A) 防止更新过激

**阅读顺序：**
1. 先找 Actor 和 Critic 分别怎么定义的
2. 找 GAE 的计算逻辑
3. 找 PPO loss 的实现
4. 理解为什么 Critic 需要单独的网络

#### 7.2 GRPO (train_grpo.py)

**关键概念：**
- 取消 Critic 网络
- 同一个问题生成 N 个回答，算平均作为 baseline
- 优势 A = (reward - group_mean) / group_std
- `loss_type="cispo"` 切换为 CISPO 变体

**阅读顺序：**
1. N 个回答是怎么采样的？
2. group_mean 和 group_std 怎么算的？
3. 和 PPO 的 loss 相比，改了哪几行？
4. `loss_type=cispo` 时 loss 公式变成什么？

#### 7.3 CISPO (train_grpo.py, loss_type="cispo")

PPO/GRPO 的 ratio 被 clip 之后梯度流就断了。CISPO 把策略项改写成：
```
min(r, ε_max) · A · log(π_θ)
```
这样即使 ratio 被截断，梯度还在。

#### 7.4 Agentic RL (train_agent.py)

**关键概念：**
- 多轮 rollout：generate → tool_call → exec → observation → 继续 generate
- 延迟奖励：整条轨迹跑完才结算一次 reward
- Reward = R_answer + R_tool + R_format + R_rm - R_unfinished

**阅读顺序：**
1. rollout 的循环结构在哪？
2. tool_call 怎么解析和执行的？
3. reward 怎么计算的？
4. policy update 用的是哪种 PO 算法？

### 动手实践

```bash
# 需要准备 reward model（internlm2-1_8b-reward）放在同级目录

# GRPO 训练
cd trainer && python train_grpo.py

# CISPO 训练
cd trainer && python train_grpo.py  # loss_type 改配置

# Agent RL
cd trainer && python train_agent.py

# 测试效果
python eval_toolcall.py --weight agent
```

### 延伸阅读
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [CISPO 论文](https://huggingface.co/papers/2506.13585)

---

## 阶段八：部署与服务（第 31-33 天）

### 学习目标
- 理解 OpenAI API 兼容服务器
- 理解模型转换（torch ↔ transformers ↔ GGUF）
- 能部署自己的模型服务

### 需要阅读的文件

| 文件 | 重点 |
|------|------|
| `scripts/serve_openai_api.py` | REST API 实现 |
| `scripts/web_demo.py` | Streamlit WebUI |
| `scripts/convert_model.py` | 格式转换 + LoRA merge |

### 动手实践

```bash
# 启动 API 服务
cd scripts && python serve_openai_api.py

# 测试
cd scripts && python chat_api.py

# 启动 WebUI
cd scripts && streamlit run web_demo.py
```

---

## 阶段九：蒸馏与进阶（第 34-35 天）

### 学习目标
- 理解黑盒蒸馏 vs 白盒蒸馏
- 理解 CE + KL 混合损失

### 需要阅读的文件

| 文件 | 重点 |
|------|------|
| `trainer/train_distillation.py` | teacher/student + CE+KL loss |

### 核心公式

```
白盒蒸馏 Loss = α · CE(y, p_student) + (1-α) · T² · KL(softmax(p_teacher/T) || softmax(p_student/T))
```

---

## 总览：学习路径图

```
Day 1-2    Tokenizer & Embedding
              ↓
Day 3-7    Transformer 架构（核心阶段，精读 model_minimind.py）
              ↓
Day 8-10   训练基础设施（DDP, 混合精度, LR调度, Checkpoint）
              ↓
Day 11-15  Pretrain + SFT（跑通训练，看 loss 曲线）
              ↓
Day 16-18  LoRA 低秩适配
              ↓
Day 19-21  DPO 偏好优化
              ↓
Day 22-30  PPO → GRPO → CISPO → Agent RL（最核心阶段）
              ↓
Day 31-33  部署与服务（API, WebUI, GGUF）
              ↓
Day 34-35  知识蒸馏
```

---

## 学习策略

1. **先跑代码，再读源码**：跑通训练流程、看到效果后再读代码，理解更深
2. **画图**：每读完一个模块，自己画出数据流图
3. **做思考题**：每个阶段的思考题都是面试级别的
4. **改写代码**：理解后再尝试改参数、加模块，验证理解是否正确
5. **阅读论文原文**：每个算法都有对应在延伸阅读里的原论文
6. **看 README**：这个项目的主 README 写得极好，几乎所有算法的理论背景都讲到了
