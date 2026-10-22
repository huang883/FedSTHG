"""Binary metrics with explicit undefined-case handling; patient is positive."""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score


METRICS=('ACC','AUC','SEN','SPE','F1')


def binary_metrics(labels, probability):
    y=np.asarray(labels);p=np.asarray(probability)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid metric input')
    if not np.isin(y,[0,1]).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected binary labels and probabilities in [0,1]')
    pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'ACC':float((pred==y).mean()),
            'AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'SEN':float(tp/(tp+fn)) if tp+fn else None,
            'SPE':float(tn/(tn+fp)) if tn+fp else None,
            'F1':float(f1_score(y,pred,zero_division=0)),
            'n':len(y),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def fold_summary(values):
    return {k:{'mean':float(np.mean(v)), 'sd':float(np.std(v,ddof=1)) if len(v)>1 else None,
               'folds':len(v)}
            for k in METRICS if (v:=[m[k] for m in values if m[k] is not None])}
