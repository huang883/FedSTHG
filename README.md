# FedSTHG 可审计复现项目

本工作区位于 `experiment/temporal-projection` 分支：新增“两层时间投影对照＋训练/验证曲线诊断”。原始成果保留在 GitHub `main` 的 `e6d0ad4`。新实验的冻结方案见 [projection_protocol.md](docs/projection_protocol.md)，结果见 [projection_report.md](docs/projection_report.md)。新增配置默认不启用两层投影，旧检查点仍可加载。下面原始复现说明和历史结果继续保留。

这是依据用户提供论文独立实现的科研项目，包含公式测试、两阶段个性化联邦训练、ABIDE真实数据实验和明确标记的合成验证。**不声称已精确复现作者全部实验。** 原始影像管线、若干网络/训练细节和作者受试者名单未公开，见研究报告。最终实测结果见 `docs/experiment_report.md`（由运行记录生成）。

## 先看这些文档

- `docs/research.md`：完整研究解析、A/B/C不确定性、公式—模块—测试表、原文结果与分阶段验收。
- `docs/data_preparation.md`：数据来源、484人与论文529人的差异、数据接入和防泄漏规则。
- `docs/learning_guide.md`：面向本科生的实现讲解。
- `docs/open_questions.md`：严格复现仍需作者确认的事项。
- `docs/abide_manifest.json`：真实下载文件的URL、哈希、形状和排除记录。

## 环境

实际运行：Windows11、Python3.13.9、PyTorch2.9.1+cu128、RTX5070 Laptop 8GB。论文使用Python3.11和RTX A6000；本项目用纯PyTorch矩阵运算表达论文GIN/HGIN，不依赖PyG，自定义公式与原始HGNN不同。

以下PowerShell命令从本项目目录执行，所有模块/配置均为实际文件。创建新环境：

```powershell
py -3.13 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
```

无NVIDIA GPU时将第一条torch安装的索引替换为 `https://download.pytorch.org/whl/cpu`，训练使用`--device cpu`。`requirements-lock.txt`记录本次GPU运行的实际软件版本，不是跨平台通用安装清单。

在当前交付工作区，无需重装，可使用已经测试的环境：

```powershell
$py = "..\..\work\venv\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $py -m pytest -q --junitxml=results/tests.xml
```

CUDA确定性设置由训练/推理入口自动设置；直接调用库时先设置`$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"`。确定性保证针对已验证软件/设备和轮次边界，不承诺跨硬件逐位一致。

## 先跑合成功能验证

```powershell
& $py -m fedsthg.experiment --config configs/synthetic.json --synthetic-dir data/synthetic --cache cache/synthetic --out results_new/synthetic --device cpu
& $py run_ablations.py --config configs/synthetic.json --manifest data/synthetic/manifest.json --cache cache/synthetic --out results_new/synthetic_ablations --device cpu
```

合成集60人、2个虚拟客户端、12个ROI、隐藏16维、5轮、5折；只测试功能。其高AUC不能解释成脑疾病诊断效果，消融次序也不应当与论文比较。

## 下载与训练真实 ABIDE

```powershell
& $py -m fedsthg.download --out data/abide --workers 6
& $py -m fedsthg.experiment --config configs/abide_smoke.json --manifest data/abide/manifest.json --cache cache/abide --out results_new/abide_smoke --device cuda --folds 0
& $py -m fedsthg.experiment --config configs/abide.json --manifest data/abide/manifest.json --cache cache/abide --out results_new/abide_full --device cuda
& $py run_ablations.py --config configs/abide.json --manifest data/abide/manifest.json --cache cache/abide --out results_new/abide_ablations --device cuda
```

仅当环境代理失效、且网络允许直接访问时，下载加`--direct`。它只忽略本进程代理，不改系统设置。当前已下载数据位于项目外`../../work/data/abide`，FC缓存位于`../../work/cache/abide`，可将上述manifest/cache路径替换成这些路径复用。

`configs/abide.json`采用正文确认的116ROI、M116、64隐藏维、两层、batch16、dropout0.3、AdamW、50轮、lambda/mu及四个独立k；**window30/step2、subject-k5、adapter、种子42等是已记录假设。** `adhd.json/mdd.json/id.json`仅提供方法配置，不表示数据已取得或实验已执行。

六个消融：去时间、去空间、仅图、仅超图、等权融合、去个性化。单路径消融宽度由128改64；无个性化消融移除受试者细化/adapter，聚合全局编码器和分类头，仅阶段I。后者只用于显式消融，完整FedSTHG仍严格只上传共享编码器并执行阶段II。这些细节为B11假设，不能声称等同于作者消融代码。

## 结果、续训与审计

每个实验保存：配置、环境、源码SHA256、cohort、split、summary；每折保存每轮loss/验证日志、最终指标、逐受试者预测和包含所有客户端状态的checkpoint。只在固定最后一轮评估测试集，不用测试集选轮次。验证集没有加入训练。服务器聚合权重使用该折**训练人数**。

中断后按原命令添加`--resume`。从最近完整通信轮恢复；加载时核对配置、受试者划分、manifest和源码哈希，不允许静默换数据或模型。默认不覆盖已有checkpoint。每个LocalOptimize按已记录假设重建AdamW，所以轮次边界无需保存跨轮动量；本项目不支持恢复到半个本地epoch。

审计交付结果并从检查点重新推理：

```powershell
& $py audit_run.py --run results/abide_full --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --device cuda
```

指标单位在JSON中为0–1，报告中为百分数。默认同时保存跨站点pooled和macro-client；正文未明确其汇总方式，二者不能混为同一数值。五折SD用ddof=1。单类AUC/SEN/SPE无法定义时写null，不伪造0.5。

## 单受试者推理

```powershell
& $py predict.py --checkpoint results/abide_full/fold_0/checkpoint.pt --site NYU --roi ../../work/data/abide/roi/NYU_0050952_rois_aal.1D --batch-size 1 --device cuda --out results/inference_example.json
```

这是使用已参与训练站点的本地模型进行研究性分类。默认B=1完整绕过受试者细化；使用多个测试受试者和batch16时，预测会受同批其他受试者影响，这是论文的设定。不能把训练和测试样本放进同一推理批次。未知站点没有本地模型，入口会拒绝；本项目没有声称实现新站点零样本迁移。

## 工程结构

```text
fedsthg/
  config.py          论文设置和显式假设
  preprocessing.py   ROI对齐、归一化、sFC/dFC
  graphs.py          四种图/超图和HGIN消息算子
  layers.py          并行GIN/HGIN
  model.py           编码、池化、融合、本地预测
  federated.py       两阶段优化、聚合、断点
  data.py            manifest、缓存、划分
  download.py        官方ABIDE下载与哈希
  metrics.py         ACC/AUC/SEN/SPE/F1
  experiment.py      训练/评估入口
tests/ configs/ docs/ results/
predict.py run_ablations.py audit_run.py
```

原始影像预处理尚未重建；MDD授权、私有ID、作者代码/补充材料仍未取得；本项目没有实现或声称完成所有13种外部基线、5随机种子的统计检验、25模型可解释性及隐私攻击实验。联邦实现是单机顺序模拟，没有部署跨机构服务、差分隐私或安全聚合。以上限制不影响检查现有公式与运行证据，但限制了“完整论文复现”的结论。

## 结果报告与交付校验

完整模型和六项真实数据消融均完成后，从保存的 JSON 重新生成报告和对照 CSV：

```powershell
& $py make_report.py
# 可选的科研结果图
& $py -m pip install -r requirements-report.txt
& $py make_report.py --plot
```

交付压缩包包含源代码、文档、配置、测试日志、逐人预测与最终模型检查点，不含 Python 环境和原始数据缓存。工作区现有数据保存在 `../../work/data/`，缓存位于 `../../work/cache/`；异地使用请按数据准备说明重新下载并通过命令行指定路径。`delivery_manifest.json` 记录交付文件的 SHA-256，可用 `verify_delivery.py` 校验。结果文件中的本机绝对路径属于运行溯源，不影响显式指定新路径重跑。

新实验结果保存在 `results/projection_study/`。实验结束后运行以下命令审计并生成曲线报告（绘图需要 `requirements-report.txt`）：

```powershell
& $py audit_run.py --run results/projection_study/baseline --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --device cuda
& $py audit_run.py --run results/projection_study/temporal256 --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --device cuda
& $py audit_projection_curves.py --manifest ../../work/data/abide/manifest.json --cache ../../work/cache/abide --device cuda
& $py report_projection.py
```

注意：Git clone 到其他目录后，`../../work/` 不一定存在。应自行创建环境、下载数据，并把上述命令中的路径替换为实际位置。`main` 和实验分支各自的文件校验清单对应各自交付时的内容。
