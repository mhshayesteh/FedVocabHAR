"""Original ranked-MNN mechanism with labels removed from candidate construction.

Inputs: unlabeled windows, known-only calibration/prototypes, cluster callable.
Labels enter only through the counted oracle. Audit enrichment lives elsewhere.
Keep legacy numerical ordering here so equivalence can be checked before redesign.
"""
import hashlib
import numpy as np

def unit(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def construct(view, clients, prototypes, calibration, cluster_fn,
              max_candidates=600, min_cluster=12):
    rows, pools, discarded = [], [], []
    emb, ids = view.embeddings, view.window_ids
    for client in clients:
        idx = np.where(view.client_ids == client)[0]
        novelty = 1 - (emb[idx] @ prototypes.T).max(1)
        order = np.argsort(novelty)[-min(max_candidates, len(idx)):]
        q = idx[order]
        assignments = cluster_fn(emb[q], calibration['local_distance'])
        pools.append(dict(client=int(client), window_ids=ids[q].tolist(),
                          cluster_assignments=np.asarray(assignments).tolist()))
        for lab in np.unique(assignments):
            members = q[np.where(assignments == lab)[0]]
            if len(members) < min_cluster:
                discarded.append(dict(client=int(client), window_ids=ids[members].tolist()))
                continue
            sig = unit(emb[members].mean(0, keepdims=True))[0]
            scores = emb[members] @ sig
            medoid = members[np.argmax(scores)]
            sorted_scores = np.sort(scores)
            rows.append(dict(client=int(client), signature=sig, support=len(members),
                             medoid=int(ids[medoid]), window_ids=ids[members].tolist(),
                             medoid_margin=float(sorted_scores[-1]-sorted_scores[-2]) if len(scores)>1 else None))
    return rows, pools, discarded

def associate(rows, threshold):
    if not rows: return [], []
    z = np.asarray([r['signature'] for r in rows]); sim = z @ z.T
    best, margins = {}, []
    for i, row in enumerate(rows):
        eligible = [j for j, other in enumerate(rows) if other['client'] != row['client'] and sim[i,j] >= threshold]
        if eligible:
            best[i] = max(eligible, key=lambda j: sim[i,j])
            scores = sorted((float(sim[i,j]) for j in eligible), reverse=True)
            margins.append(dict(cluster=i, chosen=best[i], threshold_margin=scores[0]-threshold,
                                nearest_margin=scores[0]-scores[1] if len(scores)>1 else None))
    groups, seen = [], set()
    for i,j in best.items():
        pair = tuple(sorted((i,j)))
        if best.get(j) == i and pair not in seen:
            groups.append(list(pair)); seen.add(pair)
    return groups, margins

def candidates(rows, groups):
    result = []
    for group in groups:
        members = [rows[i] for i in group]
        sig = unit(np.stack([r['signature'] for r in members]).mean(0, keepdims=True))[0]
        # Deliberately reproduce legacy tie behavior; this is not a revised rule.
        chosen = max(range(len(members)), key=lambda i: sig @ members[i]['signature'])
        scores = [float(sig @ r['signature']) for r in members]
        result.append(dict(signature=sig, medoid=members[chosen]['medoid'],
            support=sum(r['support'] for r in members), member_clusters=list(group),
            window_ids=sum([r['window_ids'] for r in members], []),
            representative_scores=scores, representative_margin=max(scores)-min(scores)))
    return result

def acquire(cands, known_labels, oracle, budget=4):
    ordered = sorted(enumerate(cands), key=lambda pair: pair[1]['support'], reverse=True)[:budget]
    promoted, trace, seen = [], [], set()
    for index, row in ordered:
        label = oracle.query(row['medoid'])
        status = 'known' if label in known_labels else ('duplicate_novel' if label in seen else 'promoted')
        trace.append(dict(call=len(trace)+1, candidate=index, window_id=row['medoid'],
                          label=label, status=status, support=row['support']))
        if status == 'promoted':
            promoted.append(dict(row, label=label)); seen.add(label)
    return promoted, trace

def membership_reference(rows, pools, discarded, groups, cands):
    """JSON-safe label-free reference; includes complete selected memberships."""
    return dict(pools=pools, discarded=discarded, groups=groups,
        clusters=[{k:v for k,v in r.items() if k != 'signature'} | {'signature_sha256':array_sha(r['signature'])} for r in rows],
        candidates=[{k:v for k,v in r.items() if k != 'signature'} | {'signature_sha256':array_sha(r['signature'])} for r in cands])
