# 自动驾驶路口交互意图联合估计 C++ Demo 技术方案

## 1. 目标与边界

### 1.1 目标

给定自车和单个目标车在路口冲突场景下的时序观测，在线输出：

1. 目标车三个隐状态概率：
   - `yield`
   - `go`
   - `hesitate`
2. 目标车连续隐参数：
   - `accepted_gap_s`：最小接受间隙
   - `aggressiveness`：激进度
   - `yield_deceleration_mps2`：目标车选择让行时的平顺减速度

### 1.2 明确不做的事情

本 demo **不做**以下内容：

- 不生成自车和目标车预测轨迹
- 不引入神经网络
- 不做离线监督训练
- 不追求学术最优，而追求工程上能快速跑通、易解释、好调参

---

## 2. 总体方案

整体采用“**离散隐状态 + 连续参数 + 粒子滤波**”联合估计框架。

### 2.1 离散隐状态

定义目标车行为模式：

- `yield`：目标车倾向让行
- `go`：目标车倾向抢行/先行
- `hesitate`：目标车处于观望或摇摆

### 2.2 连续隐参数

定义慢变量风格参数：

- `accepted_gap_s`：目标车愿意接受的最小时距阈值
- `aggressiveness`：目标车在冲突临近时保持推进的倾向
- `yield_deceleration_mps2`：目标车决定让行后通常采用的平顺减速度

### 2.3 在线估计思路

每一帧都做如下操作：

1. 从观测中构造交互特征
2. 根据上一帧模式、当前场景和连续参数，更新模式转移概率
3. 对连续参数做随机游走传播
4. 用“模式条件期望响应”与真实观测做似然匹配
5. 更新粒子权重，输出模式概率和连续参数加权均值

---

## 3. 输入输出定义

### 3.1 输入观测

每帧输入字段如下：

- `timestamp_s`
- `dt_s`
- `ego_distance_to_conflict_m`
- `ego_speed_mps`
- `ego_acc_mps2`
- `obj_distance_to_conflict_m`
- `obj_speed_mps`
- `obj_acc_mps2`
- `object_has_priority`
- `object_has_yield_sign`

### 3.2 输出结果

每帧输出：

- `p_yield`
- `p_go`
- `p_hesitate`
- `accepted_gap_s`
- `aggressiveness`
- `yield_deceleration_mps2`
- 若干便于调试的派生特征：
   - `required_yield_deceleration_mps2`
   - `yield_feasibility`
   - `yield_deceleration_excess_mps2`
  - `delta_ttc_s`
  - `delta_ttc_rate`
  - `ego_commit_score`
  - `obj_stop_proximity_score`

---

## 4. 核心数学模型

### 4.1 状态定义

离散状态：

\[
m_t \in \{Y, G, H\}
\]

连续参数：

\[
	heta_t = [g_t, \alpha_t, a^y_t]^\top
\]

其中：

- \(g_t\)：accepted gap
- \(\alpha_t\)：aggressiveness
- \(a^y_t\)：yield deceleration

### 4.2 派生交互特征

由当前观测得到：

\[
T^{ego}_t = \frac{d^{ego}_t}{\max(v^{ego}_t, \epsilon)}
\]

\[
T^{obj}_t = \frac{d^{obj}_t}{\max(v^{obj}_t, \epsilon)}
\]

\[
\Delta T_t = T^{obj}_t - T^{ego}_t
\]

\[
\dot{\Delta T}_t \approx \frac{\Delta T_t - \Delta T_{t-1}}{dt}
\]

附加构造两个工程特征：

- `ego_commit_score`：表示自车是否已经接近冲突承诺区
- `obj_stop_proximity_score`：表示目标车是否接近让行/停止位置

### 4.3 条件化隐状态转移

模式转移不是固定矩阵，而是场景条件化 softmax：

\[
P(m_t = k | m_{t-1}, \theta_{t-1}, s_t)
= \mathrm{softmax}(\text{score}_k)
\]

其中 `score_k` 由以下因素组成：

- 上一模式保持偏置
- 当前 \(\Delta T_t - g_t\) 的裕度
- 激进度 \(\alpha_t\)
- 让行减速度强度
- 目标车当前加/减速证据
- 自车承诺程度
- 路权和让行标志

直觉上：

- gap 裕度大 + 目标减速 + 让行减速度更强 + 自车承诺高 → `yield` 概率升高
- gap 裕度小 + 目标激进 + 有优先权 → `go` 概率升高
- gap 处于临界带且纵向动作不明确 → `hesitate` 概率升高

### 4.4 连续参数传播

连续参数采用简单随机游走：

\[
\theta_t = \theta_{t-1} + w_t
\]

\[
w_t \sim \mathcal{N}(0, Q)
\]

参数边界采用硬截断：

- `accepted_gap_s` \(\in [0.6, 3.0]\)
- `aggressiveness` \(\in [0, 1]\)
- `yield_deceleration_mps2` \(\in [-3.0, -0.5]\)

### 4.5 模式条件观测似然

为了不引入轨迹生成，本 demo 只对两个关键量建立似然：

1. 目标纵向加速度 `obj_acc_mps2`
2. 时距差变化率 `delta_ttc_rate`

即：

\[
p(o_t | m_t, \theta_t)
= p(a^{obj}_t | m_t, \theta_t) \cdot p(\dot{\Delta T}_t | m_t, \theta_t)
\]

#### 4.5.1 `yield` 模式的期望响应

- 目标车应表现出减速
- 更贴近“先匀速观察、后进入平顺匀减速”的工程风格
- `delta_ttc` 应逐步变大
- 平顺让行减速度越大，预测减速越明显
- 自车越接近承诺区、目标越靠近停止线，减速越明显

当前工程实现中，`yield` 模式不是一进入就立即强制制动，而是增加了基于目标车 `obj_time_to_conflict_s` 的减速启动门控：

- 离冲突点还较远时，允许 `yield` 模式下保持近似匀速观望
- 接近设定的 braking onset TTC 后，期望加速度逐步收敛到 `yield_deceleration_mps2`
- 因此该模式更适合表达“已经决定让行，但会以平顺方式开始减速”的行为

此外，工程实现里还引入了一个轻量的让行可行性约束：

- 根据当前目标车距离、速度以及 `accepted_gap_s`，近似反推“若要让 ego 先过，目标车当前至少需要提供多大常值减速度”
- 将这个近似所需减速度与粒子的 `yield_deceleration_mps2` 做比较
- 当所需减速度超过该粒子可接受的平顺让行减速度时，`yield` 的转移分数和观测似然都会被连续压低
- 且超出的幅度越大，`yield` 越不容易成立

为便于调试，该约束相关量也会随帧输出：

- `required_yield_deceleration_mps2`：当前近似所需让行减速度
- `yield_feasibility`：由所需减速度与隐参数比较得到的平滑可行性分数
- `yield_deceleration_excess_mps2`：所需减速度超过隐参数能力的超出量

#### 4.5.2 `go` 模式的期望响应

- 目标车应表现出保速或轻加速
- `delta_ttc` 应减小或维持较小负向趋势
- 激进度越高、有路权越明显，推进倾向越强

#### 4.5.3 `hesitate` 模式的期望响应

- 目标纵向动作较弱
- `delta_ttc_rate` 更接近 0
- 在 gap 临界带内更容易出现

工程实现里使用高斯似然：

\[
p(x) = \mathcal{N}(x; \mu, \sigma^2)
\]

---

## 5. 为什么这种实现适合快速工程落地

### 5.1 不依赖训练数据

参数、规则、权重都可人工初始化和手工标定，适合快速起一版。

### 5.2 解释性强

每个输出都能追溯到：

- 当前时距差
- 目标加减速
- 路权/让行规则
- 已估计的 gap/aggressiveness/yield deceleration

### 5.3 复杂度低

单目标仅做粒子传播与简单高斯似然更新，不涉及图搜索、神经网络推理、联合轨迹优化。

### 5.4 易于后续升级

后续可以逐步替换：

- 用更好的观测模型替换手工似然
- 用 IMM-UKF 或 RBPF 强化推断
- 用数据驱动校准状态转移权重
- 增加多目标管理和目标生命周期管理

---

## 6. Demo 工程结构

```text
.
├── CMakeLists.txt
├── README.md
├── data/
│   └── sample_observations.csv
├── docs/
│   └── technical_solution.md
├── include/
│   └── intent_demo/
│       ├── csv_io.h
│       ├── intent_estimator.h
│       └── types.h
└── src/
    ├── csv_io.cpp
    ├── intent_estimator.cpp
    └── main.cpp
```

---

## 7. 关键模块说明

### 7.1 `types.h`

定义：

- 输入观测 `Observation`
- 派生特征 `DerivedFeatures`
- 连续参数 `ContinuousParameters`
- 输出结构 `EstimatorOutput`

### 7.2 `intent_estimator.h / .cpp`

实现核心联合估计器：

- 初始化粒子集合
- 计算派生特征
- 计算模式转移概率
- 连续参数传播
- 计算观测似然
- 权重归一化与重采样
- 输出加权结果

### 7.3 `csv_io.h / .cpp`

负责：

- 读取输入 CSV
- 写出每帧结果 CSV

### 7.4 `main.cpp`

负责：

- 加载输入文件
- 按帧调用估计器
- 打印摘要结果
- 保存输出 CSV

---

## 8. 当前 demo 的参数含义与建议

### 8.1 粒子数

默认 `300`。

建议：

- 原型验证：200~300
- 更稳定：500 左右
- 单机性能较弱：100~200

### 8.2 过程噪声

- `process_noise_gap = 0.05`
- `process_noise_aggressiveness = 0.03`
- `process_noise_yield_deceleration = 0.08`

噪声太小：参数僵化，不容易自适应。
噪声太大：参数抖动，后验不稳定。

### 8.3 观测方差

- `sigma_acc = 0.60`
- `sigma_delta_rate = 0.25`

如果目标加速度观测噪声大，应适当放宽 `sigma_acc`。

### 8.4 让行减速启动参数

- `yield_brake_onset_ttc_s = 2.8`
- `yield_brake_ramp_ttc_s = 1.2`

它们控制 `yield` 模式从“匀速观察”过渡到“平顺匀减速”的启动时机与过渡宽度。

### 8.5 让行可行性平滑参数

- `yield_feasibility_softness_mps2 = 0.25`

该参数越小，`required_yield_deceleration_mps2` 一旦超过 `yield_deceleration_mps2`，`yield` 概率下降越快；越大则下降越平滑。

---

## 9. 输出如何消费

本 demo 本身只输出估计结果，但在实际系统中，下游典型消费方包括：

1. **行为决策器**
   - 根据 `p_yield / p_go / p_hesitate` 决定是否保留 through/yield 候选
2. **规划器**
   - 根据 `accepted_gap_s`、`aggressiveness` 调整通行风险和代价
3. **commit gate**
   - 根据对象风格和模式置信度决定是否允许真正吃让行窗口
4. **风险监督层**
   - 对高激进度、小 gap 对象提高保守权重
5. **离线诊断模块**
   - 分析误判是模式层问题还是参数层问题

---

## 10. 已知局限

### 10.1 单目标 demo

当前只针对单个目标车。真实量产系统需要：

- 多目标并行估计
- 目标生命周期管理
- 目标关联与消失重建

### 10.2 只看纵向交互

本 demo 核心证据来自：

- 纵向加速度
- 冲突点时距差

没有显式建模横向轨迹和地图细节。

### 10.3 规则参数需要人工标定

由于不使用数据驱动，权重主要靠人工经验初始化，因此不同场景需要重新调参。

---

## 11. 后续建议升级路线

### 阶段 1：把 demo 接进真实日志回放

- 从自车规划日志导出双方观测
- 对不同路口场景回放
- 检查输出是否符合直觉

### 阶段 2：增强输入特征

增加：

- 目标航向与转向状态
- 停止线距离
- 信号灯状态
- 地图优先权
- 更稳健的冲突点计算

### 阶段 3：强化推断结构

- 粒子中加入模式依赖过程噪声
- 引入模式翻转快速检测
- 用 IMM-UKF/RBPF 提高稳定性

### 阶段 4：再考虑数据驱动校准

等工程版基本跑顺以后，再考虑：

- 用数据校准转移权重
- 用学习模型替代手工似然的一部分
- 引入更细的风格参数

---

## 12. 运行说明

### 构建

优先使用 CMake：

```bash
cmake -S . -B build
cmake --build build
```

如果开发环境还没装 `cmake`，也可以直接用系统编译器快速验证：

```bash
c++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude src/main.cpp src/intent_estimator.cpp src/csv_io.cpp -o intent_demo
```

### 运行默认样例

```bash
./build/intent_demo
```

### 自定义输入输出

```bash
./build/intent_demo data/sample_observations.csv output/sample_outputs.csv
```

---

## 13. 交付物清单

本次交付包括：

1. 可编译的 C++ demo 工程
2. 样例观测输入 CSV
3. 结果输出 CSV 接口
4. 完整技术方案文档
5. README 使用说明

---

## 14. 结论

这版 demo 刻意保持克制：

- 不上神经网络
- 不做花哨预测
- 不做复杂联合轨迹推断

而是只解决一个核心问题：

> 给定双方观测，在线输出目标车三个隐状态概率和三个连续风格参数。

对于快速工程验证，这种方案是合适的：

- 实现成本低
- 解释性强
- 便于和现有规划/决策系统对接
- 后续也有清晰升级路径
