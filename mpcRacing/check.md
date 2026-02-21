# Hybrid RL-MPC Defensive Racing – Revision Checklist

---

# Phase 0 – 不依赖新实验的修改（优先完成）

---

## A. Abstract 重写（即使实验未补也必须改）

☐ 明确问题是 “defensive racing under adversarial interaction”  
☐ 明确指出 pure RL 的约束与安全问题  
☐ 明确指出 pure MPC 缺乏战略层  
☐ 不再强调“架构新颖”，改为强调“defensive modeling”  
☐ 强调：RL 生成 defensive spatial waypoints  
☐ 强调：MPC enforce safety & constraints  
☐ 删除过度实验性表述（避免“significant improvement”之类）  
☐ 用“demonstrate effectiveness in representative scenarios”代替夸张语气  

---

## B. Introduction 重构

☐ 删除主观性语言  
☐ 增加 defensive racing 相关文献  
☐ 增加 pure RL racing 文献  
☐ 增加 game-theoretic racing 文献  
☐ 明确指出现有方法缺口（缺 defensive blocking 建模）  
☐ 重写 contribution 列表（聚焦战略建模，而不是架构）  

---

## C. Related Work 重排逻辑

☐ Section A – End-to-End RL Racing  
☐ Section B – MPC/Game-Theoretic Racing  
☐ Section C – Hierarchical RL-MPC  
☐ 每节最后写 “limitation summary”  
☐ 加对比表（Strategy / Safety / Constraints / Real-time）  

---

## D. 公式与模型修正

### 动力学

☐ 解释 Eq.(2) 控制输入选择  
☐ 说明输入与 MPC 兼容性  
☐ 明确 tire model 来源  
☐ 写清 Eq.(3) 与 Eq.(2) 的关系  

### Frenet

☐ 解释 Eq.(6) 的用途  
☐ 明确 Frenet 仅用于战略层  
☐ 明确 MPC 在 world frame 运行  

---

## E. Reward Function 明确化

☐ 写出完整 reward 公式  
☐ 明确每一项物理意义  
☐ 解释 collision fault 区分逻辑  
☐ 修改 Fig.4 使其与 reward 对应  
☐ 修改 Fig.8 标签说明  

---

## F. Baseline 叙事修正

☐ 将 “Spatial Envelope MPC” 改名为 “Aggressive MPC”  
☐ 加 opponent driving mode 表格  
☐ 解释对手具备 overtaking intent  
☐ 调整文字，避免“弱对比”质疑  

---

# Phase 1 – 不新增算法，仅增强叙事

---

## G. 实验逻辑重构（即使未补新实验）

☐ 明确说明 straight defense 价值有限  
☐ 强调 corner 才是 defense 关键  
☐ 将现有实验描述为 “representative corner case”  
☐ 加 discussion 段落解释局限性  

---

## H. Pure RL 对比（不做新实验的情况下）

☐ 查找 pure RL racing 文献  
☐ 做对比表  
☐ 分析 safety / constraint / stability 差异  
☐ 写 computational complexity 对比  

---

# Phase 2 – 需要补实验

---

## I. 增加复杂弯道

☐ Hairpin  
☐ Large-radius curve  
☐ 连续弯  

---

## J. 泛化能力测试

☐ 不同初始距离  
☐ 不同对手 aggressiveness  
☐ 不同初始速度  

---

## K. Sim-to-Real 讨论

☐ 加 vehicle model validation 说明  
☐ 加 sim-to-real gap discussion  
☐ 若有真实数据，加对比  

---

# 当前优先执行顺序

1️⃣ A – Abstract  
2️⃣ B – Introduction  
3️⃣ D – 公式修正  
4️⃣ E – Reward 明确化  
5️⃣ F – Baseline 叙事  
6️⃣ C – Related Work 重排  

---


