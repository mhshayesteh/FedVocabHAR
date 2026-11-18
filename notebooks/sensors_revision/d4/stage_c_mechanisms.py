"""Label-free structural pilot mechanisms. No activity arrays or oracle histograms."""
import numpy as np
import reference_acquisition as A

METHODS=('original_mnn','original_support','average_half_mnn','complete_mnn',
         'parent_refinement_mnn','diversity_local','diversity_windows')

def cosine_matrix(z):
    z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    return np.clip(1-z@z.T,0,2)

def complete(z,d):
    if len(z)<2:return np.zeros(len(z),dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None,distance_threshold=d,metric='cosine',linkage='complete').fit_predict(z)

def make_row(view,positions,client,parent=None):
    w=view.window_ids[positions];z=view.embeddings[positions]
    signature=A.unit(z.mean(0,keepdims=True))[0];scores=z@signature
    row=dict(client=int(client),signature=signature,support=len(w),medoid=int(w[np.argmax(scores)]),window_ids=w.tolist())
    if parent is not None:row['parent_cluster']=int(parent)
    return row

def refine(view,original_rows,original_pools,original_discarded,d,cluster_fn=complete):
    positions={int(w):i for i,w in enumerate(view.window_ids)}
    rows=[];discarded=list(original_discarded);triggered=0;child_drops=0
    for parent,row in enumerate(original_rows):
        ix=np.array([positions[w] for w in row['window_ids']],dtype=int)
        diameter=float(cosine_matrix(view.embeddings[ix]).max())
        if diameter<=d+1e-6:
            rows.append(dict(row,parent_cluster=parent));continue
        triggered+=1;assignments=cluster_fn(view.embeddings[ix],d)
        for label in np.unique(assignments):
            members=ix[assignments==label]
            if len(members)<12:
                child_drops+=len(members)
                discarded.append(dict(client=row['client'],window_ids=view.window_ids[members].tolist(),reason='refined_child_below_12',parent_cluster=parent))
            else:
                child=make_row(view,members,row['client'],parent)
                assert float(cosine_matrix(view.embeddings[members]).max())<=d+1e-6
                rows.append(child)
    cluster_of={w:i for i,r in enumerate(rows) for w in r['window_ids']}
    pools=[dict(client=p['client'],window_ids=p['window_ids'],
        cluster_assignments=[cluster_of.get(w,-1) for w in p['window_ids']],
        assignment_convention='global retained row index, -1 discarded') for p in original_pools]
    return rows,pools,discarded,dict(triggered_parents=triggered,refinement_discarded_windows=child_drops)

def greedy_diversity(candidates,known_prototypes,budget=4):
    """Each client sends only its current best proposal; count all proposals."""
    if not candidates:return [],0
    signatures=np.asarray([r['signature'] for r in candidates])
    distance=1-(signatures@known_prototypes.T).max(1)
    available=set(range(len(candidates)));chosen=[];proposals=0
    clients=sorted({r['client'] for r in candidates})
    for _ in range(min(budget,len(candidates))):
        offered=[]
        for client in clients:
            eligible=[i for i in available if candidates[i]['client']==client]
            if eligible:
                offered.append(min(eligible,key=lambda i:(-float(distance[i]),candidates[i]['medoid'])))
        proposals+=len(offered)
        winner=min(offered,key=lambda i:(-float(distance[i]),candidates[i]['client'],candidates[i]['medoid']))
        chosen.append(winner);available.remove(winner)
        distance=np.minimum(distance,1-signatures@signatures[winner])
    return chosen,proposals

def construct(method,view,clients,known_prototypes,calibration,average_fn):
    original=A.construct(view,clients,known_prototypes,calibration,average_fn)
    rows,pools,discarded=original;extra={}
    if method=='average_half_mnn':
        rows,pools,discarded=A.construct(view,clients,known_prototypes,
            dict(calibration,local_distance=.5*calibration['local_distance']),average_fn)
    elif method=='complete_mnn':
        rows,pools,discarded=A.construct(view,clients,known_prototypes,calibration,complete)
    elif method=='parent_refinement_mnn':
        rows,pools,discarded,extra=refine(view,*original,calibration['local_distance'])
    groups=[]
    if method in ('original_mnn','average_half_mnn','complete_mnn','parent_refinement_mnn'):
        groups,_=A.associate(rows,calibration['agreement_similarity'])
        cands=A.candidates(rows,groups)
        order=sorted(range(len(cands)),key=lambda i:cands[i]['support'],reverse=True)[:4]
        proposals=len(rows)
    elif method in ('original_support','diversity_local'):
        cands=[dict(c,client=rows[i]['client']) for i,c in enumerate(A.candidates(rows,[[j] for j in range(len(rows))]))]
        if method=='original_support':
            order=sorted(range(len(cands)),key=lambda i:cands[i]['support'],reverse=True)[:4];proposals=len(rows)
        else:order,proposals=greedy_diversity(cands,known_prototypes)
    elif method=='diversity_windows':
        position={int(w):i for i,w in enumerate(view.window_ids)}
        cands=[dict(client=p['client'],medoid=w,support=1,window_ids=[w],
                    signature=view.embeddings[position[w]],member_clusters=[],representative_margin=0.)
               for p in pools for w in p['window_ids']]
        order,proposals=greedy_diversity(cands,known_prototypes)
    else:raise ValueError(method)
    return dict(rows=rows,pools=pools,discarded=discarded,groups=groups,candidates=cands,order=order,
                proposal_records=proposals,proposal_payload_bytes=268*proposals,**extra)

def purchase(candidates,order,known_labels,oracle):
    promoted=[];trace=[];seen=set()
    for candidate in order[:4]:
        row=candidates[candidate];label=oracle.query(row['medoid'])
        status='known' if label in known_labels else ('duplicate_novel' if label in seen else 'promoted')
        trace.append(dict(call=len(trace)+1,candidate=candidate,window_id=row['medoid'],label=label,status=status,support=row['support']))
        if status=='promoted':promoted.append(dict(row,label=label));seen.add(label)
    return promoted,trace
