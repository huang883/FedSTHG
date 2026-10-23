# 时间投影对照与训练/验证曲线诊断：冻结方案

日期：2026-10-22。原始成果已上传至 GitHub main，基线提交 e6d0ad475827742a820d1bd8d2484e7af15b8752。实验在独立分支 experiment/temporal-projection 进行。

本方案在运行新候选前写定。此前已经查看过该队列的基线测试结果，因此这是探索性复现实验，不声称是全新的盲测或证明统计显著提升。

## 唯一结构变量

- baseline：原有 Linear(6670,64) → ReLU → LayerNorm(64)，630,348 个推理参数。
- temporal256：Linear(6670,256) → ReLU → Linear(256,64) → LayerNorm(64)，1,927,628 个推理参数。
- temporal256 不是作者确认结构，是根据表6共享参数量和表8参数张量数提出的 B03 候选。没有追加输出 ReLU 或 dropout，以便精确固定已讨论的结构。
- 其余模块、数据、损失、batch16、每阶段1 epoch、50通信轮、学习率等均保持原样。架构不同会改变初始化随机数消耗；同种子不意味着其余权重逐个相同。

## 实验与选择规则

同一484人数据和缓存，站点内五折，seed42，初始化42+fold；baseline和temporal256均重新训练。使用原有固定划分，逐项验证与原始结果一致。每5轮及初始化时，在不更新参数、不启用dropout的模式下分别评估训练和验证集，使用各自固定的划分顺序和batch16。没有训练/验证混批，没有测试集参与曲线或选择。

记录 pooled/各客户端 ACC、AUC、SEN、SPE、F1、从logits计算的分类CE、融合门控均值和时空一致性。训练曲线的CE与验证曲线同口径，不把含近端项的优化loss当分类CE。保存并恢复模式和随机数状态，测试诊断不会改变训练轨迹。两阶段训练loss仍单独保留。

不依曲线早停，不依测试结果调参，全部固定第50轮评估。预先指定以第50轮五折平均验证AUC作为候选选择依据，并同时报告ACC和CE；不把某一个最佳轮次事后当主结果。测试只在固定最终轮评估。候选效果未知，不保证193万参数优于63万参数。

单个基础种子下五折结果只能描述该设置，不做5种子统计或显著性结论。若结果提示改进，后续另行冻结多种子验证。若训练CE降低而验证CE上升、差距扩大，描述为泛化差距/过拟合迹象，不能排除站点差异、样本量等其他原因。

## 验收

旧37项测试+新增投影参数量/梯度、诊断不改变RNG和优化轨迹、CE公式、防划分重叠、候选保存加载测试。先跑单折，再完成两组五折；从保存的最终检查点重算预测。baseline与历史逐人预测核对，确认诊断插入没有改变原始基线。输出原始曲线JSON、图、逐折对照、时间/显存和局限。

## 命令（在项目根目录）

```powershell
$py = '../../work/venv/Scripts/python.exe'
& $py -m pytest -q --junitxml=results/projection_tests.xml
& $py -m fedsthg.experiment --config configs/projection_baseline.json --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --out results/projection_study/baseline --device cuda
& $py -m fedsthg.experiment --config configs/projection_256.json --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --out results/projection_study/temporal256 --device cuda
```

基线归档哈希保留在 main 的 delivery_manifest.json；本实验分支完成后生成新的同名交付清单。历史训练的 source_hashes.json 保持原样，不能把历史源码哈希当作实验分支当前源码的哈希。
