# autonomous_intent_estimator_demo

一个面向自动驾驶路口交互场景的 C++ demo。输入自车和障碍物的时序观测，在线输出：

- 障碍物 `yield / go / hesitate` 三个隐状态概率
- 连续隐参数：`accepted_gap_s / aggressiveness / response_delay_s`

本工程强调：

- 只做工程可快速落地的一版
- 不引入神经网络和数据驱动训练
- 不生成自车或障碍物轨迹
- 用规则化特征 + 条件化状态转移 + 粒子滤波实现在线联合估计

## 目录

- `include/intent_demo/`：头文件
- `src/`：核心实现
- `data/sample_observations.csv`：样例观测输入
- `output/`：运行后生成结果
- `docs/technical_solution.md`：完整技术方案文档

## 构建

```bash
cmake -S . -B build
cmake --build build
```

## 运行

默认读取 `data/sample_observations.csv`，输出到 `output/sample_outputs.csv`：

```bash
./build/intent_demo
```

如果本机暂时没有 `cmake`，也可以直接用系统编译器快速编译：

```bash
c++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude src/main.cpp src/intent_estimator.cpp src/csv_io.cpp -o intent_demo
./intent_demo
```

也可以指定输入输出路径：

```bash
./build/intent_demo data/sample_observations.csv output/custom_outputs.csv
```

## 输入 CSV 格式

```text
timestamp_s,dt_s,ego_distance_to_conflict_m,ego_speed_mps,ego_acc_mps2,obj_distance_to_conflict_m,obj_speed_mps,obj_acc_mps2,object_has_priority,object_has_yield_sign
```

## 输出 CSV 格式

```text
timestamp_s,p_yield,p_go,p_hesitate,accepted_gap_s,aggressiveness,response_delay_s,delta_ttc_s,delta_ttc_rate,ego_commit_score,obj_stop_proximity_score
```

## 注意

这只是一个工程化原型 demo，用于快速验证联合估计框架，不直接等价于量产参数。
