"""Feature-only matched agreement interventions; no labels or file access."""
import numpy as np
import reference_acquisition as A

METHODS=('O_all','O_agreed','H_all','H_agreed','random_windows')

def from_partition(rows, groups, view, agreed):
    eligible={i for pair in groups for i in pair} if agreed else set(range(len(rows)))
    shared=sorted(range(len(rows)),key=lambda i:(-rows[i]['support'],rows[i]['client'],rows[i]['medoid']))
    order=[i for i in shared if i in eligible][:4]
    index={int(w):i for i,w in enumerate(view.window_ids)}
    # Cluster support ranks medoids, but only the purchased window is a prototype.
    candidates=[dict(r,signature=view.embeddings[index[r['medoid']]].copy(),member_clusters=[i],
                     signature_basis='purchased_window_only') for i,r in enumerate(rows)]
    assert len({candidates[i]['medoid'] for i in order})==len(order)
    return dict(candidates=candidates,order=order,shared_order=shared,eligible=sorted(eligible))

def construct(method,view,clients,known_prototypes,calibration,average_fn):
    if method not in METHODS:raise ValueError(method)
    cal=dict(calibration,local_distance=calibration['local_distance']*(.5 if method.startswith('H_') else 1.))
    rows,pools,discarded=A.construct(view,clients,known_prototypes,cal,average_fn)
    groups,margins=A.associate(rows,calibration['agreement_similarity'])
    if method=='random_windows':
        ids=sorted(w for p in pools for w in p['window_ids']);assert len(ids)==len(set(ids))
        chosen=np.random.default_rng(20261116).choice(ids,size=min(4,len(ids)),replace=False).tolist()
        lookup={int(w):i for i,w in enumerate(view.window_ids)}
        candidates=[dict(client=int(view.client_ids[lookup[w]]),medoid=w,support=1,window_ids=[w],
                         signature=view.embeddings[lookup[w]].copy(),member_clusters=[],signature_basis='purchased_window_only') for w in chosen]
        selected=dict(candidates=candidates,order=list(range(len(chosen))),shared_order=None,eligible=None)
    else:selected=from_partition(rows,groups,view,method.endswith('_agreed'))
    return dict(rows=rows,pools=pools,discarded=discarded,groups=groups,association_margins=margins,**selected)

def generated():
    from data_access import CandidateView,BudgetedOracle
    import stage_c_mechanisms as C
    z=np.eye(4,dtype=np.float32);view=CandidateView(np.array([30,10,20,40]),np.array([2,1,1,2]),z)
    rows=[dict(client=2,medoid=30,support=20,window_ids=[30],signature=np.full(4,.25)),
          dict(client=1,medoid=10,support=20,window_ids=[10],signature=np.full(4,.25)),
          dict(client=1,medoid=20,support=30,window_ids=[20],signature=np.full(4,.25)),
          dict(client=2,medoid=40,support=10,window_ids=[40],signature=np.full(4,.25))]
    all_=from_partition(rows,[[0,1]],view,False);agreed=from_partition(rows,[[0,1]],view,True)
    assert all_['shared_order']==agreed['shared_order']==[2,1,0,3]
    assert agreed['order']==[1,0] and all_['order']==[2,1,0,3]
    assert from_partition(rows,[],view,True)['order']==[]
    assert all(np.array_equal(c['signature'],z[i]) for c,i in zip(all_['candidates'],range(4)))
    # Shared query and decoder contract; known/duplicate answers still cost calls.
    labels={30:'new',10:'known',20:'new',40:'other'}
    oracle=BudgetedOracle(labels.__getitem__,view.window_ids,4)
    entries,trace=C.purchase(all_['candidates'],all_['order'],frozenset(['known']),oracle)
    assert len(trace)==4 and len(entries)==2 and [t['status'] for t in trace]==['promoted','known','duplicate_novel','promoted']
    assert np.array_equal(entries[0]['signature'],z[2])
    chosen=np.random.default_rng(20261116).choice(sorted(labels),4,replace=False)
    assert len(set(chosen))==4 and set(chosen)==set(labels)
    # Exercise all constructor arms with generated embeddings and a cluster stub.
    zz=np.zeros((24,64),np.float32);zz[:12,0]=1;zz[12:,1]=1
    vv=CandidateView(np.arange(24),np.repeat([1,2],12),zz)
    pp=np.zeros((1,64),np.float32);pp[0,2]=1;cutoffs=[]
    def cluster_stub(z,cutoff):cutoffs.append(cutoff);return np.zeros(len(z),int)
    cal=dict(local_distance=.4,agreement_similarity=.9)
    built={m:construct(m,vv,[1,2],pp,cal,cluster_stub) for m in METHODS}
    assert built['O_agreed']['order']==built['H_agreed']['order']==[]
    assert len(built['O_all']['order'])==len(built['H_all']['order'])==2
    assert len(built['random_windows']['order'])==4
    assert cutoffs==[.4,.4,.4,.4,.2,.2,.2,.2,.4,.4]
    for b in built.values():
        for c in b['candidates']:assert np.array_equal(c['signature'],zz[c['medoid']])
    return dict(shared_rank_filter=True,support_client_id_ties=True,empty_agreement=True,
                purchased_window_prototype=True,charged_known_duplicate=True,random_without_replacement=True,
                all_five_constructor_arms=True,cutoff_isolation=True)
