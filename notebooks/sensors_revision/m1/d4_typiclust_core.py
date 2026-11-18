"""Independent TC-Frozen-Sensor-v1. NumPy core; sklearn imported only for fit."""
import numpy as np

def validate(ids,z):
    ids=np.asarray(ids);z=np.asarray(z,dtype=np.float64)
    if ids.ndim!=1 or z.ndim!=2 or len(z)!=len(ids) or len(set(map(int,ids)))!=len(ids):raise ValueError('ID/feature mismatch')
    if not np.isfinite(z).all() or np.any(np.linalg.norm(z,axis=1)<=0):raise ValueError('Invalid vector')
    order=np.argsort(ids);return ids[order].astype(np.int64),z[order]

def typicality(z):
    z=np.asarray(z,dtype=np.float64);n=len(z)
    if n<2:raise ValueError('At least two rows needed')
    k=min(20,n-1);means=[]
    for i in range(n):
        # Direct subtraction avoids cancellation in squared-norm distance identities.
        d=np.linalg.norm(z-z[i],axis=1);d[i]=np.inf
        means.append(float(np.partition(d,k-1)[:k].mean()))
    means=np.asarray(means);return 1/(means+1e-12),means,k

def select_partition(ids,z,assignments,budget=4):
    if not isinstance(budget,int) or not 0<=budget<=4:raise ValueError('Budget')
    ids=np.asarray(ids);z=np.asarray(z,dtype=np.float64);assignments=np.asarray(assignments)
    if len(ids)!=len(z) or assignments.shape!=(len(ids),):raise ValueError('Partition mismatch')
    groups=[np.flatnonzero(assignments==lab) for lab in np.unique(assignments)]
    groups.sort(key=lambda ix:int(ids[ix].min()));clusters=[]
    for i,ix in enumerate(groups):
        eligible=len(ix)>=5
        scores,means,k=typicality(z[ix]) if eligible else (None,None,None)
        clusters.append(dict(cluster=i,window_ids=ids[ix].tolist(),support=len(ix),eligible=eligible,
                             typicality=scores.tolist() if eligible else None,
                             mean_neighbor_distance=means.tolist() if eligible else None,k=k))
    count=[0]*len(groups);used=set();chosen=[];selection=[]
    for step in range(budget):
        available=[i for i,c in enumerate(clusters) if c['eligible'] and any(w not in used for w in c['window_ids'])]
        if not available:break
        ci=min(available,key=lambda i:(count[i],-clusters[i]['support'],min(clusters[i]['window_ids'])))
        c=clusters[ci];positions=[j for j,w in enumerate(c['window_ids']) if w not in used]
        j=min(positions,key=lambda j:(-c['typicality'][j],c['window_ids'][j]))
        wid=c['window_ids'][j];chosen.append(wid);used.add(wid)
        selection.append(dict(call=step+1,window_id=wid,cluster=ci,previous_selected_count=count[ci],typicality=c['typicality'][j]))
        count[ci]+=1
    return dict(query_ids=chosen,clusters=clusters,selection=selection,unused_budget=budget-len(chosen))

def select(ids,z,budget=4):
    from sklearn.cluster import KMeans
    ids,z=validate(ids,z)
    if not len(ids):return dict(query_ids=[],clusters=[],selection=[],unused_budget=budget,requested_clusters=0)
    k=min(budget,len(ids))
    if k==0:return dict(query_ids=[],clusters=[],selection=[],unused_budget=budget,requested_clusters=0)
    model=KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=42,max_iter=300,tol=1e-4,algorithm='lloyd')
    assignments=model.fit_predict(z)
    result=select_partition(ids,z,assignments,budget)
    return dict(result,requested_clusters=k,realized_clusters=len(np.unique(assignments)),inertia=float(model.inertia_),iterations=int(model.n_iter_))

def random_ids(ids,seed,budget=4):
    ids=np.asarray(sorted(map(int,ids)));assert len(ids)==len(set(ids))
    return np.random.default_rng(seed).choice(ids,min(budget,len(ids)),replace=False).tolist()

def generated(with_fit=False):
    scores,means,k=typicality(np.array([[1.],[2.],[4.]]))
    assert k==2 and np.allclose(means,[2,1.5,2.5],rtol=0,atol=1e-15)
    assert np.argmax(scores)==1 # Euclidean, not squared distances; self excluded.
    _,means,_=typicality(np.array([[1.],[1.],[2.]]));assert np.allclose(means,[.5,.5,1.])
    sc,means,_=typicality(np.ones((5,2)));assert np.all(means==0) and np.all(sc==1e12)
    ids=np.arange(10,20);z=np.ones((10,2));a=np.repeat([0,1],5)
    r=select_partition(ids,z,a);assert r['query_ids']==[10,15,11,16]
    assert [s['previous_selected_count'] for s in r['selection']]==[0,0,1,1]
    assert select_partition(np.arange(4),np.ones((4,2)),np.zeros(4,int))['query_ids']==[]
    assert select_partition(np.arange(5),np.ones((5,2)),np.zeros(5,int))['query_ids']==[0,1,2,3]
    try:validate([1,1],np.ones((2,2)))
    except ValueError:pass
    else:raise AssertionError('Duplicate ID accepted')
    for bad in [np.zeros((2,2)),np.full((2,2),np.nan)]:
        try:validate([1,2],bad)
        except ValueError:pass
        else:raise AssertionError('Invalid features accepted')
    assert len(set(random_ids(range(30),20261117)))==4
    assert random_ids(range(30),20261117)==random_ids(reversed(range(30)),20261117)
    if with_fit:
        # Distinct generated clusters, no research examples or optimizer/model.
        z=np.concatenate([np.tile(np.eye(4)[i],(8,1)) for i in range(4)])
        fitted=select(np.arange(32),z);assert fitted['realized_clusters']==4 and len(fitted['query_ids'])==4
        assert all(len(c['window_ids'])==8 for c in fitted['clusters'])
        collapsed=select(np.arange(8),np.ones((8,4)))
        assert collapsed['realized_clusters']==1 and len(collapsed['query_ids'])==4
        assert select([],np.empty((0,4)))['query_ids']==[]
    return dict(hand_euclidean_typicality=True,self_duplicate_distance=True,finite_zero_distance=True,
                five_member_boundary=True,cluster_point_ties=True,selected_count_updates=True,invalid_features=True,
                random_without_replacement=True,generated_KMeans='passed' if with_fit else 'pending_Colab')
