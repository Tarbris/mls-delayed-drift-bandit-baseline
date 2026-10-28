"""Data-free valid worst-case certificates, to be improved by measurement."""
import numpy as np

def plan(observations):
    return {'allocation':[int(v) for v in observations['audit_design']['minimum']]}

def finish(observations,audit_result):
    x=np.asarray(observations['target']['x'])
    bank=np.asarray(observations['policy_bank'])
    logits=np.einsum('nd,pkd->npk',np.column_stack([np.ones(len(x)),x]),bank)
    logits-=logits.max(axis=2,keepdims=True)
    p=np.exp(logits);p/=p.sum(axis=2,keepdims=True)
    diff=p-p[:,:1]
    lower=np.minimum(diff,0).sum(axis=2).mean(axis=0)
    lower[0]=0
    return {'lower_bounds':lower.tolist(),'selection':0}
