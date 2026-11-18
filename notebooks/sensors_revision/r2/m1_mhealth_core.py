"""M1 pure NumPy/standard-library curation, decoding and fixed-vocabulary metrics."""
from collections import Counter
import numpy as np

def curate(raw,subject,mapping):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=24:raise ValueError('Expected exactly24columns')
    if not 1<=subject<=10 or len(raw)>=1000000000:raise ValueError('Subject/ID bounds')
    codes=raw[:,23]
    if not np.isfinite(codes).all() or not np.equal(codes,np.floor(codes)).all():raise ValueError('Noninteger label')
    codes=codes.astype(np.int64)
    if not set(codes)<=(set(mapping)|{0}):raise ValueError('Unexpected label code')
    xyz=raw[:,14:17].astype(np.float32)
    if not np.isfinite(xyz).all():raise ValueError('Nonfinite selected XYZ')
    mag=np.sqrt(np.sum(xyz*xyz,axis=1,dtype=np.float32)).astype(np.float32)
    if not np.isfinite(mag).all():raise ValueError('Nonfinite magnitude')
    signal=np.column_stack([xyz,mag]);xs=[];ys=[];ids=[];starts=[];rejected=0
    for start in range(0,max(0,len(raw)-127),64):
        block=codes[start:start+128]
        if block[0]==0 or not np.all(block==block[0]):rejected+=1;continue
        xs.append(signal[start:start+128].T.copy());ys.append(mapping[int(block[0])]);ids.append(subject*1000000000+start);starts.append(start)
    x=np.stack(xs) if xs else np.empty((0,4,128),dtype=np.float32)
    return x,np.asarray(ys,dtype='<U32'),np.asarray(ids,dtype=np.int64),dict(subject=subject,rows=len(raw),eligible_windows=len(xs),excluded_null_or_transition=rejected,starts=starts)

def normalize(x,stats):
    # Preserve Frozen.normalized_input operation ordering and final cast.
    return np.concatenate([(x[:,:3]-stats['xyz_mean'])/stats['xyz_std'],(x[:,3:4]-stats['mag_mean'])/stats['mag_std']],1).astype(np.float32)

def metrics(y,p,known,novel):
    assert len(y)==len(p)
    yc=Counter(map(str,y));pc=Counter(map(str,p));tp=Counter(str(a) for a,b in zip(y,p) if a==b)
    vocab=list(known)+list(novel);f={a:2*tp[a]/(yc[a]+pc[a]) if yc[a]+pc[a] else 0. for a in vocab}
    return dict(known_macro_f1=float(np.mean([f[a] for a in known])),novel_macro_f1=float(np.mean([f[a] for a in novel])),
        combined_macro_f1=float(np.mean(list(f.values()))),per_class_f1=f,per_class_recall={a:tp[a]/yc[a] if yc[a] else 0. for a in vocab},per_class_support={a:yc[a] for a in vocab})

def subject_metrics(y,p,subjects,known,novel,scheduled):
    scores={str(s):metrics(y[subjects==s],p[subjects==s],known,novel) for s in scheduled}
    mean={k:float(np.mean([scores[str(s)][k] for s in scheduled])) for k in ['known_macro_f1','novel_macro_f1','combined_macro_f1']}
    return dict(subject_mean=mean,per_subject=scores,pooled=metrics(y,p,known,novel))

def validate_roles(acquisition,evaluation):
    if set(acquisition)&set(evaluation) or sorted(list(acquisition)+list(evaluation))!=list(range(1,11)):raise ValueError('Subject roles')

def generated():
    mapping={i:'activity_'+str(i) for i in range(1,13)}
    raw=np.zeros((384,24),np.float64);raw[:,14:17]=[3,4,12];raw[:128,23]=1;raw[128:256,23]=0;raw[256:,23]=12
    x,y,ids,info=curate(raw,2,mapping)
    assert x.shape==(2,4,128) and x.dtype==np.float32 and np.all(x[:,3]==13)
    assert y.tolist()==[mapping[1],mapping[12]] and ids.tolist()==[2000000000,2000000256]
    assert info['excluded_null_or_transition']==3
    raw[:,23]=2;raw[127,23]=1
    xx,yy,ii,_=curate(raw,1,mapping);assert ii.tolist()==[1000000128,1000000192,1000000256]
    raw[:,23]=1
    for code in range(1,13):
        raw[:,23]=code;assert set(curate(raw,1,mapping)[1])=={mapping[code]}
    for defect in ['width','nan_xyz','label_fraction','label_unknown']:
        bad=raw.copy()
        if defect=='width':bad=bad[:,:23]
        elif defect=='nan_xyz':bad[0,14]=np.nan
        elif defect=='label_fraction':bad[0,23]=1.5
        else:bad[0,23]=13
        try:curate(bad,1,mapping)
        except ValueError:pass
        else:raise AssertionError(defect)
    raw[:,23]=0;assert curate(raw,1,mapping)[0].shape==(0,4,128)
    for subjects in [([1,2,3,4,5],[6,7,8,9,10]),([1,2,3,4,5],[5,7,8,9,10])]:
        try:validate_roles(*subjects)
        except ValueError:assert subjects[1][0]==5
        else:assert subjects[1][0]==6
    stats=dict(xyz_mean=np.array([[[1],[2],[3]]]),xyz_std=np.array([[[2],[4],[8]]]),mag_mean=np.array([[[5]]]),mag_std=np.array([[[2]]]))
    a=np.ones((2,4,128),np.float32)*13;n=normalize(a,stats)
    assert n.dtype==np.float32 and np.array_equal(n[0,:,0],np.array([6,2.75,1.25,4],np.float32))
    known=['k1','k2','k3','k4'];novel=['n'+str(i) for i in range(8)]
    q=metrics(['n0'],['n0'],known,novel);assert q['novel_macro_f1']==1/8 and q['combined_macro_f1']==1/12
    q=subject_metrics(np.array(['n0','n0','n0']),np.array(['n0','n0','k1']),np.array([6,6,7]),known,novel,[6,7])
    assert q['subject_mean']['novel_macro_f1']==1/16 and q['pooled']['novel_macro_f1']!=1/16
    return dict(columns_and_magnitude=True,float32=True,boundary_null_grid=True,unique_large_ids=True,all12labels=True,invalid_inputs=True,empty_population_shape=True,subject_isolation=True,normalization_hand_values=True,eight_novel_denominator=True,subject_vs_pool_aggregation=True)
