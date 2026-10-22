# FedSTHG：实现前研究与可行性评估

研究日期：2026-10-20。本文先于模型代码编写。结论不是“已完成严格复现”。

## 1. 依据与来源核对

主依据为用户提供的《FedSTHG_论文中文全文译稿_附原文.pdf》，共 47 页：中文译稿 1–22 页，附录说明 23 页，英文原文 24–47 页。以下页码均指**英文原文印刷页码**，在合并 PDF 中加 23。已阅读全部正文、图表说明、参考文献，并渲染核对 Fig.3、Eqs.17–37、Table 2。同目录另有 24 页英文 PDF，提取全文用于交叉核对。译稿有公式转义损坏（例如 concat 被显示为 norm），以英文公式为准。

论文标注作者 Xinyue Yu 等，期刊 Expert Systems With Applications，DOI `10.1016/j.eswa.2026.134766`，2026-10-14 接收。Crossref 已独立确认题名、六位作者、文章编号及出版月，证据见下文。提供稿件页眉卷页仍是占位符，致谢含修订标记；尚未取得正式版全文逐项核对修订差异。

实际检索了完整题名、FedSTHG、DOI、GitHub/Gitee 限域搜索。未找到可确认对应的公开官方仓库。英文 p.23 Data availability 明确说明代码和预训练模型需向通讯作者合理申请。补充材料指向 DOI，未获取，补充表 S1 未核实。不把其他同名项目当作官方实现。

公开来源：

- 论文所载 DOI：https://doi.org/10.1016/j.eswa.2026.134766
- ABIDE 原始数据入口：https://fcon_1000.projects.nitrc.org/indi/abide/databases.html
- ABIDE PCP 下载：https://preprocessed-connectomes-project.org/abide/download.html
- ABIDE PCP 预处理/图谱：https://preprocessed-connectomes-project.org/abide/Pipelines.html
- PCP 元数据仓库：https://github.com/preprocessed-connectomes-project/abide
- ADHD-200：https://neurobureau.projects.nitrc.org/ADHD200/Introduction.html
- ADHD 预处理下载目录：https://neurobureau.projects.nitrc.org/ADHD200/Data.html
- REST-meta-MDD：https://rfmri.org/REST-meta-MDD
- MDD 官方页面指向 ScienceDB DOI：`10.57760/sciencedb.o00115.00013`；需签署协议获取密码及表型资料，不能绕过。
- GIN 原作者仓库：https://github.com/weihua916/powerful-gnns
- HGNN 原作者仓库：https://github.com/iMoonLab/HGNN
- Ditto 原作者仓库：https://github.com/litian96/ditto
- PyTorch CUDA wheel：https://pytorch.org/get-started/previous-versions/

后续联网核验成功：Crossref API HTTP200 确认题名、六位作者、文章编号134766、出版月2026-10；完整响应保存为 `crossref.json`。DOI HTTP200 重定向到 `https://linkinghub.elsevier.com/retrieve/pii/S0957417426036705`，正式文章页为 `https://www.sciencedirect.com/science/article/pii/S0957417426036705`。GitHub 搜索 API HTTP200 返回 total_count=0。这更新了上述初次访问失败的状态，但补充材料仍未取得。

GIN、HGNN、Ditto 是相关算法来源，**不是 FedSTHG 官方实现**。本项目独立实现论文公式，没有复制这些仓库的代码。特别是 HGNN 的对称度归一化不能替换论文 HGIN 的 `H De^-1 H^T`。目前没有第三方源码复用，因此不存在未确认许可证下的复制。数据文件保留其原有使用条件，不纳入源代码许可证。

## 2. 研究问题与数据流

目标：利用 rs-fMRI 的静态/动态互补信息、成对/群组关系，在不汇集原始脑影像的多站点场景中诊断疾病，同时适应站点差异。

三个问题：普通图只表达成对关系；静动态信息简单融合不充分；异质站点下单一全局模型不足。设计对应为四路并行编码、自适应融合、持久个性化两阶段训练。

输入每名受试者 `X[T,N]`，论文 `N=116`。锚点对齐为 `[116,116]`，按 ROI 时间轴归一化。sFC `[N,N]`；dFC `[W,N,N]`；`W=floor((M-Lwin)/Lstep)+1`。空间节点是 ROI，时间节点是**整张 dFC 的窗口**。输出二分类概率 `[B,2]`，患者标记 1。

```
BOLD -> 对齐/归一化 -> sFC -> signed row projection -> Spatial GIN  ┐
                                            └-----> Spatial HGIN ┴ concat -> mean ┐
                   -> dFC -> upper triangle -> projection + pos -> Temporal GIN   │
                                                   └-----------> Temporal HGIN   │
                                                   concat -> attention ---------┤
                           softmax gate -> weighted sum -> projection -> z_st    │
                           本地批次受试者图 -> adapter -> head -> CE              │
                           cosine(z_s,z_t) -> consistency loss -----------------┘
```

图与超图仅共享**初始特征**，传播参数彼此独立。负 FC 保留在输入中；空间构图用 `max(C,0)` 排名，不按绝对值。零分并列时可选到非正相关节点，不能误实现为“全部负相关节点永不入图”。

## 3. 公式—代码—验证映射

以下是开发契约，测试通过情况以最终测试日志为准。B 表示描述不完整，C 表示未给出的工程决策。

| 模块/级别 | 原文位置 | 输入 → 输出 | 实现方案 | 验收测试 |
|---|---|---|---|---|
| 锚点对齐 B | §3.1 Eqs.1–2, p.5 | `[T,N]→[M,N]` | 均匀锚点邻域平均，范围单独配置 | 恒值、等长、压缩、扩展、非法数据 |
| 归一化 A/C | Eq.3 | `[M,N]→[M,N]` | 单受试者逐 ROI z-score；epsilon 为假设 | 均值、常数 ROI、无跨受试者拟合 |
| sFC A/C | Eq.4 | `[M,N]→[N,N]` | Pearson 后 clip/atanh，零对角 | 与 NumPy 参照、完美正负相关、有限性 |
| dFC A/B | Eqs.5–7 | `[M,N]→[W,N,N]` | 固定滑窗，长度/步长未给出 | 窗口数、最后窗口、边界、对称 |
| 空间图 A/C | Eqs.8–9 | `[B,N,N]→A[B,N,N]` | 正相关优先 kNN，OR 对称，无自环 | 手算邻居、并列规则、负值保留 |
| 空间超图 A | Eq.10 | `[B,N,N]→H[B,N,N]` | 列为锚点超边，包含自己+独立 k_sh 邻居 | 每列 k_sh+1、重叠、方向 |
| 时间节点 A/B | Eqs.11–12 | `[B,W,N,N]→[B,W,64]` | 上三角 6670 维，投影+可学习位置；投影内部待确认 | 位置梯度、非原始 FC 构图、W 语义 |
| 时间图/超图 A | Eqs.13–14 | `[B,W,64]→A,H[B,W,W]` | latent cosine kNN，各自 k_tg/k_th | 邻接和超边成员；模型更新后重新构图 |
| GIN A | Eqs.15–16 | `[B,m,64]→[B,m,64]` | LN(MLP((1+eps)X+AX)) 两层 | 显式循环参照、epsilon 梯度 |
| HGIN A | Eqs.17–21 | 同上 | LN(MLP((1+eps)X+H De^-1 H^T X)) 两层 | 手算两阶段、矩阵/循环等价、无节点度归一化 |
| 分支整合 A | Eq.22 | 两个 `[B,m,64]→[B,m,128]` | 末端拼接，绝不串行 | 独立参数、共同投影、梯度 |
| 池化 A | Eqs.23–25 | 空间/时间→两个 `[B,128]` | 空间均值；时间 learnable query softmax | 注意力和为1、零 query 对应均值 |
| 门控融合 A | Eqs.26–27 | `[B,128]` 两路→`[B,128]` | concat→2门控，128→128→ReLU→LN→dropout | 权重和为1、数值、梯度 |
| 一致性 A/C | Eq.28 | 两个 `[B,128]→标量` | 1-cos，batch mean，数值 epsilon | 同向0、反向2、零向量稳定 |
| 受试者图 A/B | Eqs.29–30 | `[B,128]→[B,128]` | 本地当前批次 cosine；D^-1/2 A D^-1/2；无 bias Wp | B=1 全层绕过；标签无关；无跨划分图 |
| adapter/head B | Eqs.31–33 | `[B,128]→[B,2]` | 可替换本地 adapter 和两层 head；详细形状为假设 | 维度、梯度、CE vs log-softmax |
| 阶段 I A/C | Eq.35 | θ,φ_k→临时θ_sh,φ_k^I | AdamW 一轮本地 epoch，θ_k 不变 | 参数快照、持久模型不被覆盖 |
| 聚合 A | Eq.36 | 临时编码器列表→θ_new | 仅编码器，按训练人数加权 | 手算权重、拒绝混入 local keys |
| 阶段 II A/C | Eq.37 | θ_k,φ_k^I→新个性化状态 | 固定 detach θ_new；CE+λST+μ/2 sum squared | 固定参考无梯度、持久路径、近端梯度 |
| 完整训练 A/C | Eqs.38–39 | 多站点→各自模型 | 50轮、每阶段1 epoch、最终个性化推理 | 保存/加载、续训等价、无泄漏 |
| 解释性 B | Eqs.40–41 §4.8 | heldout→ROI/超边排序 | 原文遮挡定义/消息强度细节不足 | 本次不声称复现25模型解释性实验 |

## 4. 已确定超参数（A）

`N=116, M=116, d=64, L=2, dropout=.3, batch=16`；每层 MLP `64→128→64`，中间 ReLU/dropout，末端 LN；每层 epsilon 初始化 0。AdamW `lr=.0003, weight_decay=.0001`，clip norm 1，固定 50 轮，无学习率衰减/早停。每轮共享、个性化阶段各 1 epoch。

| 数据集 | k_sg | k_tg | k_sh | k_th | μ | λ |
|---|---:|---:|---:|---:|---:|---:|
| ABIDE | 8 | 5 | 10 | 7 | 1 | .001 |
| ADHD-200 | 15 | 2 | 6 | 5 | 1 | .001 |
| REST-meta-MDD | 15 | 3 | 3 | 3 | .01 | .005 |
| ID | 8 | 5 | 3 | 8 | 2 | .01 |

## 5. 不能伪装成论文事实的缺口

| 编号/级别 | 缺失项 | 暂定决策 / 风险 |
|---|---|---|
| B01 | Lwin/Lstep 未给 | 实验配置暂用 30/2（W=44）；必须显式标记，不是论文默认值 |
| B02 | 锚点平均片段边界未给 | 均匀锚点覆盖 0…T-1，半径 (T-1)/(2(M-1))，空段取最近点；等长恒等；扩展重复取样 |
| B03 | 时间投影内部未给 | Linear(6670,64)-ReLU-LN，然后加位置向量；可能解释参数量差异 |
| B04 | k_p 未给 | 暂用5，batch内 min(5,B-1)；影响共批次推理 |
| B05 | adapter/head 内部维度未给 | adapter 128→128 ReLU LN dropout；head 128→64→2 ReLU；bias 为工程假设 |
| B06 | 原始预处理软件、滤波频段、去除帧、QC未给 | 下载 PCP cpac/filt_global AAL 作为独立公开数据实验；**不能声称等于作者统一管线** |
| B07 | 受试者名单、排除标准、UM/UCLA子站点合并未给 | 提供名单及计数；不为匹配529人数而按结果删人 |
| B08 | 随机种子、分层与否、固定测试顺序未给 | 暂用42，站点内分层5折，训练折再分层20%验证，测试按ID排序；保留全部split |
| B09 | 优化器状态跨阶段/轮次是否延续未给 | 每次 LocalOptimize 新建 AdamW；模型参数持久；断点保留轮次边界 RNG |
| B10 | 指标跨站点如何汇总、SD ddof 未给 | 同时报 pooled 和 macro-client；折SD用ddof=1；避免宣称精确对应原表 |
| C01 | 并列kNN、clip/epsilon、初始化未给 | 稳定按索引破同分；clip 1e-6；std ddof=0；pos/query normal std .02 |
| B11 | 消融具体维度调整/简单融合未给 | graph-only/hyper-only删除相应路径并调整宽度；无融合为等权平均；无个性化另行说明 |
| B12 | Table6参数量与正文简单投影不匹配的可能性 | 实测统计，不通过堆无关参数凑1.914M |

严谨的“同配置结果复现”仍需要作者代码、预处理配置、名单、补充材料和随机种子。允许的工程假设不等于已经解决这些缺口。

## 6. 数据与结果目标

原文 Table1：ABIDE NYU184/UM145/USM101/UCLA99，共529；ADHD NYU216/Peking194/KKI83/OHSU79，共572；MDD S20 533/S21 156/S25 152/S8 150，共991；ID94（47患者47对照）分为47/47两个虚拟客户端。ID按患者ISI降序33/14、对照年龄降序14/33分配；缺失ISI放最后、并列按ID，不伪造这些临床数据。

Table2 的 FedSTHG 结果（%，均值±折SD；不是本项目结果）：

| 数据集 | ACC | AUC | SEN | SPE | F1 |
|---|---|---|---|---|---|
| ABIDE | 72.81±2.22 | 77.93±3.25 | 70.89±2.70 | 72.96±4.50 | 71.04±2.42 |
| ADHD-200 | 68.59±3.98 | 64.29±3.69 | 68.83±5.73 | 67.82±7.45 | 62.64±2.58 |
| MDD | 64.68±2.93 | 66.28±3.41 | 72.30±7.84 | 55.81±10.12 | 68.58±3.12 |
| ID | 78.65±7.67 | 77.68±6.62 | 74.67±14.24 | 82.29±5.43 | 77.29±9.12 |

五折：每客户端80%开发数据，其中20%验证，最终约64/16/20。测试batch16固定顺序，同批测试受试者之间构图是论文设定，不可与训练/验证拼批。不能按测试结果调参或选择轮次。最终模型用第50轮，验证用于配置选择。统计实验为5 seeds ×5 folds，先折均值再进行种子级配对t检验与BH-FDR，不能把25折当独立样本。

六项消融：去时间、去空间、仅图、仅超图、去个性化、等权融合。原文Table5全模型ACC/AUC均最高，但并不证明每项差异显著。Table4 ABIDE对FedAvg+GCN/GIN未显著；Table7 ID attention–mean ACC80.67高于默认78.65，不能称默认池化普遍最优。解释性结果也不是因果/临床生物标志物。隐私压力测试不能证明抗重建能力。

## 7. 环境与可行性（初检）

- Windows 11，10.0.26200；Intel Core Ultra 9 275HX，24核/24线程。
- 物理内存约31.4 GiB，初检可用约15.3 GiB。
- NVIDIA RTX 5070 Laptop 系列，8151 MiB 显存；驱动592.01；nvidia-smi 的 CUDA13.1 是驱动能力，**不是已安装 PyTorch runtime**。
- 系统 Python3.13.9，初检无torch；将在 workspace/work/venv 隔离安装 torch2.9.1+cu128。与论文Python3.11不同，最终锁定实测依赖。
- C盘约47 GiB可用，E盘约831 GiB；优先下载小体积ROI衍生物，不批量抓取全原始影像。
- 项目初始为空，没有可复用模型。网络默认代理连接被拒绝；仅对任务进程绕过失效代理后 PyPI HTTP200，不修改系统设置。

判断：公式模块、CPU小规模测试、单GPU全维训练可行；8GB显存是否足够通过实测决定。四数据集完整严格复现当前不具备条件，主要障碍是方法细节/数据访问而非必然GPU不足。原始影像预处理还需要完整扫描元数据和影像工具链；ROI输入不是从原始NIfTI复现的证明。

## 8. 分阶段计划与验收门槛

1. 研究：全文、来源、A/B/C、形状/公式对应、资源评估落盘（本文）；不填假结果。
2. 数据/构图：对齐、FC、四种结构分别通过解析参照和边界测试后进入模型阶段。
3. 四路编码、融合、个性化：独立公式测试、两层与共享投影检查、全路径反传、B=1绕过后进入联邦阶段。
4. 联邦：验证共享/持久/本地三个参数流，近端参考冻结、加权聚合、断点续训等价。
5. 合成集：仅用于功能验证，小规模训练/推理与消融、GPU全尺寸batch测试，结果明确synthetic。
6. 公共数据：从官方地址下载ABIDE并hash核验，记录每个排除原因；先小规模真实数据流程，再运行资源允许的五折/50轮。差异完整保留。
7. 交付：源码、精确命令、配置、测试日志、逐受试者预测、split、checkpoint、环境锁、结果对照与未完成项。不把一次成功运行等同于方法/结果全面验证。
