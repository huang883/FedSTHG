"""Summarize the frozen projection experiment from actual five-fold outputs."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/projection_study'
NAMES = ['baseline', 'temporal256']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def avg_sd(values, percent=False):
    scale = 100 if percent else 1
    return f'{statistics.mean(values)*scale:.2f} ± {statistics.stdev(values)*scale:.2f}'


def collect():
    curves = {}; finals = {}; timings = {}; summaries = {}; rows=[]
    for name in NAMES:
        folder=OUT/name
        summary=read(folder/'summary.json')
        assert summary['evaluated_folds']==list(range(5)) and summary['rounds']==50
        summaries[name]=summary;curves[name]=[];finals[name]=[];timings[name]=[]
        assert read(folder/'splits.json')==read(ROOT/'results/abide_full/splits.json')
        for fold in range(5):
            path=folder/f'fold_{fold}'
            log=read(path/'training_log.json')
            trajectory=[read(path/'initial_diagnostics.json')]+[r['diagnostics'] for r in log if 'diagnostics' in r]
            assert [r['round'] for r in trajectory]==list(range(0,51,5))
            curves[name].append(trajectory);finals[name].append(trajectory[-1])
            timings[name].append(read(path/'metrics.json'))
            for record in trajectory:
                for split in ['train','validation']:
                    rows.append({'variant':name,'fold':fold,'round':record['round'],'split':split,
                                 **record[split]['pooled']})
    with (OUT/'curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return curves,finals,timings,summaries


def main():
    curves,finals,timings,summaries=collect()
    # Historical baseline predictions must remain unchanged by diagnostic evaluation.
    replay=[]
    for fold in range(5):
        def records(path):
            with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))
        old=records(ROOT/f'results/abide_full/fold_{fold}/predictions.csv')
        new=records(OUT/f'baseline/fold_{fold}/predictions.csv')
        assert old==new, f'Diagnostics changed baseline predictions at fold {fold}'
        replay.append({'fold':fold,'historical_predictions_exact':True,'n':len(new)})
    audits={}
    for name in NAMES:
        a=read(OUT/name/'audit.json')
        assert a['all_checks_passed'] and a['complete_five_fold_run']
        audits[name]=a
    evidence={'historical_baseline_reproduction':replay,'checkpoint_audits':audits,
              'comparison_is_exploratory':True,'base_seeds':[42], 'test_used_for_selection':False}
    evidence['curve_audit']=read(OUT/'curve_audit.json')
    assert evidence['curve_audit']['all_checks_passed']
    (OUT/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    lines=['# 两层时间投影对照与学习曲线诊断', '',
           '原始成果已上传 GitHub main（e6d0ad4）。本报告对应独立实验分支；新版本是结构假设，不是已确认的作者网络。'
           '实验方案先于训练写定，见 [projection_protocol.md](projection_protocol.md)。', '',
           '两组均使用484名ABIDE受试者、同一客户端内五折、seed42、50轮、原有训练配置。'
           '唯一结构变量为时间输入投影。每5轮在eval模式下分别计算训练/验证指标，测试只在最终轮评估。'
           '这里的±是五折样本标准差，不是五次独立种子实验或置信区间。', '',
           '## 1. 结构与验收', '',
           '| 项目 | 原始单层 | 两层256 |','|---|---:|---:|',
           '| 时间投影 | 6670→64→ReLU→LN | 6670→256→ReLU→64→LN |',
           '| 推理参数 | 630,348 | 1,927,628 |',
           '| 共享编码器参数 | 588,554 | 1,885,834 |', '',
           '- 41项测试通过：原有37项加投影、诊断、损失和保存加载测试。',
           '- 插入诊断后的五折基线逐人预测与历史结果完全相同；诊断没有改变该次训练结果。',
           '- 两组共10个最终检查点经独立审计重算预测，详见 verification.json 和各组 audit.json。',
           '- 当前两层结构接近论文的参数分布及时间分支35个参数张量，但总参数仍不是论文1.914M；'
           '本地模块、时间窗口等假设仍未解决。', '',
           '## 2. 固定第50轮的训练与验证表现', '',
           '| 模型 | 训练ACC % | 验证ACC % | 训练AUC % | 验证AUC % | 训练CE | 验证CE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        cells=[]
        for metric in ['ACC','AUC','ce']:
            for split in ['train','validation']:
                cells.append(avg_sd([v[split]['pooled'][metric] for v in finals[name]],metric!='ce'))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    validation={name:statistics.mean(v['validation']['pooled']['AUC'] for v in finals[name]) for name in NAMES}
    selected=max(validation,key=validation.get)
    lines+=['',f'按预先指定的第50轮平均验证AUC，选择 **{selected}**。'
            '该选择只是在本次两候选中比较，不代表已证明跨队列稳定优越。', '',
            '## 3. 固定最终轮测试（不用于选择）', '',
            '| 指标 | 原始单层 | 两层256 | 两层−单层（百分点） |','|---|---:|---:|---:|']
    for metric in ['ACC','AUC','SEN','SPE','F1']:
        a=summaries['baseline']['pooled'][metric];b=summaries['temporal256']['pooled'][metric]
        lines.append(f"| {metric} | {a['mean']*100:.2f} ± {a['sd']*100:.2f} | "
                     f"{b['mean']*100:.2f} ± {b['sd']*100:.2f} | {(b['mean']-a['mean'])*100:+.2f} |")
    lines+=['','逐折记录：','','| 折 | 单层验证AUC % | 两层验证AUC % | 单层测试AUC % | 两层测试AUC % |',
            '|---|---:|---:|---:|---:|']
    for f in range(5):
        lines.append(f"| {f} | {finals['baseline'][f]['validation']['pooled']['AUC']*100:.2f} | "
                     f"{finals['temporal256'][f]['validation']['pooled']['AUC']*100:.2f} | "
                     f"{timings['baseline'][f]['pooled']['AUC']*100:.2f} | "
                     f"{timings['temporal256'][f]['pooled']['AUC']*100:.2f} |")
    lines+=['','## 4. 曲线与站点诊断','','![训练验证曲线](../results/projection_study/learning_curves.png)', '',
            '分类CE直接从logits计算，未混入近端惩罚或一致性损失；训练与验证都关闭dropout，'
            '分别在各自数据内以固定顺序构图。曲线阴影为五折标准差。','']
    for name in NAMES:
        train_ce=statistics.mean(v['train']['pooled']['ce'] for v in finals[name])
        val_ce=statistics.mean(v['validation']['pooled']['ce'] for v in finals[name])
        initial_val=statistics.mean(c[0]['validation']['pooled']['ce'] for c in curves[name])
        gap=statistics.mean(v['train']['pooled']['ACC']-v['validation']['pooled']['ACC'] for v in finals[name])*100
        lines.append(f'- {name}：第50轮训练CE {train_ce:.4f}，验证CE {val_ce:.4f}；'
                     f'初始验证CE {initial_val:.4f}；训练−验证ACC差 {gap:.2f} 个百分点。')
    lines+=['','训练很好而验证较差是泛化差距的证据；若验证CE上升同时AUC保持相近，'
            '还可能是错误预测更自信，不等于所有排序能力都下降。该实验不能把差距唯一归因为过拟合或站点偏移。', '',
            '| 站点 | 单层验证ACC % | 两层验证ACC % | 单层时间门控均值 | 两层时间门控均值 |',
            '|---|---:|---:|---:|---:|']
    sites=list(finals['baseline'][0]['validation']['clients'])
    for site in sites:
        vals=[]
        for name in NAMES:vals.append(statistics.mean(v['validation']['clients'][site]['ACC'] for v in finals[name])*100)
        for name in NAMES:vals.append(statistics.mean(v['validation']['clients'][site]['gate_mean'][1] for v in finals[name]))
        lines.append(f'| {site} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.6f} | {vals[3]:.6f} |')
    gates=[v['validation']['clients'][s]['gate_mean'][1] for name in NAMES for v in finals[name] for s in sites]
    lines+=['',f'两组各折各站点最终验证集的平均时间门控范围为 {min(gates):.6f}–{max(gates):.6f}，'
            '融合权重高度偏向时间分支。空间路径仍参与一致性损失和门控计算，不能据此断言完全没有作用。'
            '后续应优先追踪门控饱和、两分支特征尺度及梯度，再判断是否需要调整；'
            '不能把强制等权或新增约束伪装成论文原始方法。', '',
            '## 5. 成本与结论边界', '',
            '| 模型 | 平均每折耗时秒（含诊断/保存） | 峰值PyTorch allocated MiB |', '|---|---:|---:|']
    for name in NAMES:
        lines.append(f"| {name} | {statistics.mean(m['elapsed_this_invocation_s'] for m in timings[name]):.1f} | "
                     f"{max(m['peak_gpu_allocated_MiB'] for m in timings[name]):.1f} |")
    lines+=['','本次只有一个基础种子、五折，且之前已看过基线测试结果，因此属于探索性比较，'
            '不能宣称统计显著、临床有效或完整复现论文。更多参数不自动代表更好。'
            '后续应先依据验证结果确定是否保留两层投影，再冻结多种子确认方案；'
            '若泛化差距明显，优先研究正则化/学习率/个性化约束，不继续盲目扩容。', '',
            '原始数据与划分差异未改变；参数增大也改变初始化随机数消耗及近端惩罚的参数集合。'
            '本次保持论文μ不变，未把近端项改成均方，否则会成为另一项算法改动。'
            '候选投影的变化包含层数、宽度和ReLU位置，不能把观测差异单独归因于参数量。','']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,constrained_layout=True)
    for i,name in enumerate(NAMES):
        for j,(metric,title,scale) in enumerate([('ce','Classification CE',1),('ACC','Accuracy (%)',100),('AUC','AUC (%)',100)]):
            ax=axes[i,j]
            for split,color in [('train','#2576b7'),('validation','#d66b18')]:
                a=np.array([[r[split]['pooled'][metric]*scale for r in trajectory] for trajectory in curves[name]])
                mean=a.mean(0);sd=a.std(0,ddof=1);x=list(range(0,51,5))
                ax.plot(x,mean,label=split,color=color)
                ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12)
            ax.set_title(name+' / '+title);ax.set_xlabel('Communication round');ax.grid(alpha=.2)
            if metric!='ce':ax.set_ylim(0,105)
            else:ax.set_ylim(bottom=0)
            if j==0:ax.legend()
    fig.suptitle('Same evaluation protocol; mean ± fold SD; seed 42')
    fig.savefig(OUT/'learning_curves.png',dpi=170);plt.close(fig)
    (ROOT/'docs/projection_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected_by_validation_auc':selected,'validation_auc':validation},indent=2))


if __name__=='__main__':main()
