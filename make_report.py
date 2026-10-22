"""Regenerate the report from completed runs; never insert unmeasured results."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ['no_temporal', 'no_spatial', 'graph_only', 'hypergraph_only',
            'equal_fusion', 'no_personalization']
LABELS = {'full': '完整模型', 'no_temporal': '去时间分支', 'no_spatial': '去空间分支',
          'graph_only': '仅图', 'hypergraph_only': '仅超图',
          'equal_fusion': '等权融合', 'no_personalization': '去个性化'}


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def formatted(metric):
    return f"{100 * metric['mean']:.2f} ± {100 * metric['sd']:.2f}"


def main(plot=False):
    paper = read('docs/paper_results.json')
    runs = {'full': read('results/abide_full/summary.json')}
    for variant in VARIANTS:
        runs[variant] = read(f'results/abide_ablations/{variant}/summary.json')
    for name, run in runs.items():
        assert run['evaluated_folds'] == list(range(5)), name
        assert run['rounds'] == 50 and run['seed'] == 42 and not run['synthetic']
        prefix = 'results/abide_full' if name == 'full' else f'results/abide_ablations/{name}'
        checked = read(prefix + '/audit.json')
        assert checked['all_checks_passed'] and checked['complete_five_fold_run'], name
        assert all(f['replay_max_abs_error'] == 0.0 for f in checked['folds']), name
    lines = ['# FedSTHG 实验与复现评估', '',
             '本报告由 `python make_report.py` 从实际运行记录生成。完成了独立公式实现、测试、'
             'ABIDE 公共衍生数据的完整模型与六项消融；**尚未完成论文原始协议及四数据集的严格结果复现**。', '',
             '## 1. 实际执行范围', '',
             '- 真实 ABIDE：484 人、4 个客户端，种子 42，5 折，每折 50 轮。'
             '完整模型及六个变体共 35 次折训练、1750 轮通信。',
             '- 合成数据：60 人、2 个虚拟客户端；完整模型和六项消融各 5 折、每折 5 轮。'
             '这些数据只验证软件功能，不是疾病诊断证据。',
             '- 另有 ABIDE 单折单轮 smoke 实验，不计入下面的主结果。',
             '- 测试集仅在第 50 轮评价；未按测试指标挑选轮次或修改超参数。'
             '验证集日志、划分、配置、软件版本、源文件哈希、逐人概率和最终检查点均保留。', '',
             '## 2. ABIDE 主结果', '',
             '单位为百分数，± 为五折样本标准差（ddof=1），不是置信区间。'
             '实测为每折汇集四客户端测试预测的 pooled 指标。论文汇总细节不完整，'
             '队列和预处理也不同，因此这里只作描述性对照，不能解释为同条件算法优劣。', '',
             '| 指标 | 论文 | 实测 | 实测−论文（百分点） |',
             '|---|---:|---:|---:|']
    csvrows = []
    for metric in ['ACC', 'AUC', 'SEN', 'SPE', 'F1']:
        reference = paper['main']['ABIDE'][metric]
        actual = runs['full']['pooled'][metric]
        delta = 100 * actual['mean'] - reference[0]
        lines.append(f'| {metric} | {reference[0]:.2f} ± {reference[1]:.2f} | {formatted(actual)} | {delta:+.2f} |')
        csvrows.append([metric, *reference, actual['mean'] * 100, actual['sd'] * 100, delta])
    with (ROOT / 'results/main_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'paper_mean_percent', 'paper_sd_percent', 'actual_mean_percent',
                         'actual_sd_percent', 'difference_percentage_points'])
        writer.writerows(csvrows)
    lines += ['', '各折结果：', '', '| 折 | 测试人数 | ACC | AUC |', '|---|---:|---:|---:|']
    for fold in range(5):
        m = read(f'results/abide_full/fold_{fold}/metrics.json')['pooled']
        lines.append(f"| {fold} | {m['n']} | {100*m['ACC']:.2f} | {100*m['AUC']:.2f} |")
    macro = runs['full']['macro_client']
    lines += ['', f"客户端等权宏平均 ACC 为 {formatted(macro['ACC'])}，AUC 为 {formatted(macro['AUC'])}。"
              '完整五指标、混淆矩阵及每站点结果见各折 metrics.json；不把 pooled 与 macro 混用。', '',
              '## 3. 真实数据消融', '',
              '所有变体沿用相同的受试者划分、种子、轮数及数据缓存。删除路径后参数量变化，'
              '未人为补足容量；相同种子也不意味着不同结构具有完全相同的随机数消耗。'
              '未进行多种子显著性检验。', '',
              '| 变体 | 论文 ACC | 实测 ACC | 论文 AUC | 实测 AUC |', '|---|---:|---:|---:|---:|']
    for name, run in runs.items():
        reference = paper['main']['ABIDE'] if name == 'full' else paper['ABIDE_ablation'][name]
        lines.append(f"| {LABELS[name]} | {reference['ACC'][0]:.2f} ± {reference['ACC'][1]:.2f} | "
                     f"{formatted(run['pooled']['ACC'])} | {reference['AUC'][0]:.2f} ± {reference['AUC'][1]:.2f} | "
                     f"{formatted(run['pooled']['AUC'])} |")
    better = [LABELS[name] for name in VARIANTS
              if runs[name]['pooled']['ACC']['mean'] > runs['full']['pooled']['ACC']['mean']]
    lines += ['', ('本次 ' + '、'.join(better) + ' 的平均 ACC 高于完整模型，未重现论文“完整模型最好”的排序。'
                    if better else '本次完整模型平均 ACC 不低于各消融变体。') +
              '折间波动较大，不能仅凭均值判断模块有益或无益；本报告不根据这些测试结果重新调参。', '',
              '去个性化消融使用共享编码器与共享分类头、仅执行阶段 I；这是对该消融的显式实现假设，'
              '不是将完整 FedSTHG 替换为 FedAvg。其他未明确的消融细节见 research.md 的 B11。', '',
              '## 4. 验证证据', '',
              '- `results/tests.txt` / `tests.xml`：37 项测试通过，覆盖 FC、邻域、GIN/HGIN 解析公式、'
              '四路梯度、融合、B=1、参数隔离、近端参考、加权聚合、划分、保存恢复与小规模训练。',
              '- `docs/cohort_audit.json`：484 个 ROI 文件的列顺序一致；sFC/dFC 全部有限、对称、零对角线。'
              '5 人存在恒定 ROI，相关值按显式数值规则处理并记录，未为贴合论文人数强行筛选。',
              '- `results/*/audit.json`：独立从 CSV 重算指标，检查测试 ID、标签、站点、完整五折、'
              '检查点轮数与原始 manifest 哈希；带 cache 的审计另从检查点重算逐人预测。',
              '- 主实验与六项真实消融共35个检查点的逐人概率精确重现，最大绝对差为 0.0；'
              'CPU 断点续训（含 dropout）测试验证恢复 RNG 后与不中断运行一致。',
              '- GPU 上实际完成全尺寸 batch16 前向、反向和训练；主实验首折峰值已分配显存约 178.6 MiB。'
              '该数值是 PyTorch allocated memory，不是整个进程/驱动的显存占用。', '',
              '## 5. 差异排查与解释边界', '',
              '| 排查项 | 已查事实 | 仍未解决的影响 |', '|---|---|---|',
              '| 数据来源与名单 | 官方 PCP cpac/filt_global/AAL116，484 人；论文529人 | 作者名单、QC排除及缺失衍生物不明 |',
              '| 预处理 | 从现有 ROI 开始；对齐/归一化/Pearson/Fisher/滑窗已实现 | 未从原始 NIfTI 重建论文完整影像管线 |',
              '| 结构 | 四路并行与 HGIN 均按公式；局部参数不上传 | 当前推理630,348参数，论文约1.914M，容量明显不一致 |',
              '| 超参数 | 已用论文明确的维度、层数、邻域、lr、wd、μ、λ及50轮 | 窗长30/步长2、适配器/分类头等为显式假设 |',
              '| 优化 | 临时共享与持久个人编码器分离；阶段II参考固定 | Adam状态跨阶段是否保留未公开，本实现每次本地优化重置 |',
              '| 随机性与统计 | 固定seed42、逐折初始化42+fold、确定性设置、折SD | 未取得作者种子；未做5种子×5折统计及BH-FDR |',
              '| 环境 | Python3.13.9、torch2.9.1+cu128、RTX5070 Laptop | 与论文Python3.11/A6000不同；跨平台不保证逐位相同 |',
              '| 汇总方式 | pooled和macro均保留；正类患者，阈值0.5 | 原文汇总细节与SD定义需作者确认 |', '',
              '这些差异是结果差距的候选解释，尚无控制实验能够把差距归因到单一因素。'
              '尤其参数量差异尚未解决，不能将核心公式通过测试等同于作者架构完全一致。', '',
              '## 6. 合成验证（不可与论文诊断指标对照）', '',
              '| 变体 | ACC | AUC |', '|---|---:|---:|']
    for name in runs:
        path = 'results/synthetic/summary.json' if name == 'full' else f'results/synthetic_ablations/{name}/summary.json'
        run = read(path)
        assert run['synthetic'] and run['rounds'] == 5
        lines.append(f"| {LABELS[name]} | {formatted(run['pooled']['ACC'])} | {formatted(run['pooled']['AUC'])} |")
    lines += ['', '## 7. 可行性结论与未完成项', '',
              '现有8GB GPU足以运行本实现的116 ROI、batch16及四客户端顺序模拟。'
              '实测证明本次工程可运行，不证明作者完整网络或原始影像预处理的资源需求相同。', '',
              '| 数据集 | 实际训练 | 状态 |', '|---|---|---|',
              '| ABIDE | 完整模型及六消融，5折×50轮 | 公共ROI协议，未达论文原始结果 |',
              '| ADHD-200 | 未运行 | 官方Athena AAL训练/测试下载链接当前跳转登录；管线GSR也待核实 |',
              '| REST-meta-MDD | 未运行 | 官方要求数据使用协议及申请密码/表型，未代用户申请 |',
              '| 私有失眠ID | 未运行 | 未发现可合法直接下载的队列，不使用虚构患者替代 |', '',
              '方法层面已覆盖论文全部核心计算与两阶段参数流；原始影像预处理、作者精确网络配置、'
              '官方划分和四数据集结果尚未复现。不计算没有明确分母的“复现百分比”。', '',
              '优先后续工作：取得作者代码/补充材料与受试者名单，核实参数量差异；'
              '重建匹配的预处理和公开数据接入；在预先冻结配置后完成多种子实验。'
              '其后再补13种外部基线、统计检验、解释性及隐私压力实验。'
              '本交付是单机联邦模拟，没有跨机构部署、安全聚合或差分隐私机制。', '',
              '资源检索、公式—实现—验证对应与A/B/C假设见 [research.md](research.md)；'
              '下载和访问证据保存在同目录 JSON；学习说明见 [learning_guide.md](learning_guide.md)。', '']
    if plot:
        lines += ['![主结果与消融结果，误差线为折标准差](../results/comparison.png)', '']
    (ROOT / 'docs/experiment_report.md').write_text('\n'.join(lines), encoding='utf-8')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        x = np.arange(2)
        axes[0].bar(x-.18, [paper['main']['ABIDE'][k][0] for k in ['ACC','AUC']], .36,
                    yerr=[paper['main']['ABIDE'][k][1] for k in ['ACC','AUC']], label='Paper', capsize=4)
        axes[0].bar(x+.18, [100*runs['full']['pooled'][k]['mean'] for k in ['ACC','AUC']], .36,
                    yerr=[100*runs['full']['pooled'][k]['sd'] for k in ['ACC','AUC']], label='This run', capsize=4)
        axes[0].set(xticks=x, xticklabels=['ACC','AUC'], ylabel='Percent', ylim=(0,100),
                    title='Different cohorts / protocols; descriptive comparison')
        axes[0].legend()
        names=list(runs)
        axes[1].barh(names, [100*runs[n]['pooled']['ACC']['mean'] for n in names],
                     xerr=[100*runs[n]['pooled']['ACC']['sd'] for n in names], capsize=3)
        axes[1].set(xlim=(0,100), xlabel='Accuracy (%)', title='ABIDE: 5 folds, seed 42, 50 rounds')
        axes[1].invert_yaxis()
        fig.suptitle('FedSTHG independent implementation — mean ± fold SD')
        fig.savefig(ROOT/'results/comparison.png', dpi=180)
        fig.savefig(ROOT/'results/comparison.pdf')
        plt.close(fig)
    print('Wrote docs/experiment_report.md and results/main_comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='Also produce plots; requires matplotlib')
    main(parser.parse_args().plot)
