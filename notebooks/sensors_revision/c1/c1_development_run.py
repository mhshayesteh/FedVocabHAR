"""Colab-only C1 development; cached embeddings, no model inference/training."""
from pathlib import Path
import json, signal, time, resource, sys, traceback
import numpy as np
import torch
import d3_matched_agreement as D3
import stage_c_mechanisms as C
from data_access import BudgetedOracle
from c1_aggregate_selector import Client, select
from test_c1_aggregate_selector import run_report

P, B, Pilot = D3.P, D3.B, D3.Pilot
save, sha = D3.save, D3.sha
METHOD = 'C1_Aggregate_Centroid_Sensor_v1_global'


def run(project, out, contract):
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError('C1 fixed resource limit; preserve partial outputs')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    torch.set_num_threads(4)
    if B.runtime() != contract['runtime']:
        raise RuntimeError('Runtime differs from the pinned cached-embedding runtime; no install/fallback')
    for item in contract['inputs']:
        if sha(project/item['path']) != item['sha256']:
            raise RuntimeError('Input hash mismatch: '+item['path'])
    P.DEVICE = torch.device('cpu')
    P.PROJECT = project
    for dataset in P.DATASETS.values():
        for key in ('cache','manifest','models'):
            dataset[key] = project/dataset[key].relative_to(Path(P.CFG.project))
    P.load = P.encode = P.Frozen.embedding = P.Encoder = D3.blocked
    # Additionally refuse every Torch checkpoint/forward/optimization route.
    torch.load = D3.blocked
    torch.nn.Module._call_impl = D3.blocked
    torch.optim.Optimizer.step = D3.blocked
    P.assert_locked()
    qualification = run_report()
    if not qualification['passed']:
        raise RuntimeError('C1 synthetic qualification failed')
    save(out/'PREFLIGHT.json',dict(c1=qualification,oracle_and_scorer=D3.generated(),
                                  runtime=B.runtime(),model_routes_disabled=True))
    if (out/'STARTED.json').exists():
        raise RuntimeError('Attempt already started; no repeat')
    save(out/'STARTED.json',dict(model_calls=0,optimizer_steps=0))
    rows=[]; new_calls=0; reused_calls=0
    base=project/contract['d4_base']
    def limit():
        remaining=900-(time.monotonic()-start)
        if remaining <= 0:
            raise TimeoutError('C1 overall limit')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= 2*1024*1024:
            raise MemoryError('C1 peak RSS exceeded 2 GiB')
        signal.alarm(max(1,min(60,int(remaining))))
    for dataset in ('realworld','pamap2'):
        manifest=json.loads(P.DATASETS[dataset]['manifest'].read_text(encoding='utf-8-sig'))
        for fold_record in manifest['outer_folds']:
            limit()
            D,ids,fold,emb,_,_,_=Pilot.load_case(P,project,dataset,fold_record['fold'])
            stem=f'{dataset}_fold_{fold["fold"]}'
            old_plans=json.loads((base/(stem+'_QUERY_PLANS.json')).read_text())
            plan_seal=json.loads((base/(stem+'_PLAN_SEAL.json')).read_text())
            if sha(base/(stem+'_QUERY_PLANS.json')) != plan_seal['sha256']:
                raise RuntimeError('Old query-plan seal mismatch')
            old_pred=json.loads((base/(stem+'_PREDICTIONS.json')).read_text())
            pred_seal=json.loads((base/(stem+'_PREDICTION_SEAL.json')).read_text())
            if sha(base/(stem+'_PREDICTIONS.json')) != pred_seal['sha256'] or pred_seal['plan_sha256'] != plan_seal['sha256']:
                raise RuntimeError('Old prediction seal mismatch')
            plans={k:list(v) for k,v in old_plans['plans'].items()}
            if len(plans)!=25 or METHOD in plans:
                raise RuntimeError('Expected exactly 25 unchanged D4 arms')
            pool=old_plans['pool_ids'];index={int(w):i for i,w in enumerate(ids)}
            if len(pool)!=len(set(pool)) or len(pool)>1800:
                raise RuntimeError('Candidate-pool IDs/ceiling')
            # Each simulated client receives only its own fixed top-N pool rows.
            clients=[]
            for cid in sorted(map(int,fold[D.ds['client_key']])):
                local=[w for w in pool if int(D.subjects[index[w]])==cid]
                clients.append(Client(cid,local,emb[[index[w] for w in local]]))
            if sum(len(c.ids) for c in clients)!=len(pool):
                raise RuntimeError('Pool/client role mismatch')
            limit()
            chosen=select(clients,budget=4,cap_mode='global',seed=20261117,max_iter=100,tolerance=1e-8)
            plans[METHOD]=[row['window_id'] for row in chosen['plan']]
            if not chosen['fit'] or chosen['fit']['converged']:
                pass
            else:
                raise RuntimeError('C1 clustering hit iteration cap; preserve as failure, no rescue run')
            plan_file=out/(stem+'_QUERY_PLANS.json')
            save(plan_file,dict(plans=plans,c1=chosen,pool_ids=pool,
                role='development only',access='C1 simulated aggregate messages; original D4 privileges retained',
                cap_interpretation='global B4; fixed client allocation relaxed, not native paper protocol'))
            save(out/(stem+'_PLAN_SEAL.json'),dict(sha256=sha(plan_file)))
            # Only after C1 selection and plan seal may oracle answers be joined.
            lookup={w:str(D.activities[index[w]]) for w in pool}
            evalix=np.flatnonzero(np.isin(D.subjects,fold[D.ds['eval_key']]))
            if old_pred['window_ids']!=ids[evalix].tolist():
                raise RuntimeError('Evaluation ID mismatch')
            kp=P.known_prototypes(D,emb,fold)
            zero=np.asarray(D.known)[(emb[evalix]@kp.T).argmax(1)].tolist()
            predictions={};details={}
            for method in old_plans['plans']:
                old=json.loads((base/f'{stem}_{method}.json').read_text())
                if [r['window_id'] for r in old['trace']]!=plans[method] or any(r['label']!=lookup[r['window_id']] for r in old['trace']):
                    raise RuntimeError('Reused ledger mismatch: '+method)
                predictions[method]=old_pred['predictions'][method]
                if predictions[method]['0']!=zero:
                    raise RuntimeError('Known-only decoder mismatch')
                details[method]={key:old[key] for key in ('trace','oracle_trace','queries','recovered','prefix_recovery','known_answers','duplicate_answers')}
                details[method]['origin']='reused_exact_D4_ledger_and_predictions'
                reused_calls+=old['queries']
            qids=plans[METHOD]
            candidates=[dict(medoid=w,support=1,window_ids=[w],signature=emb[index[w]]) for w in qids]
            oracle=BudgetedOracle(lookup.__getitem__,pool,4)
            entries,trace=C.purchase(candidates,list(range(len(candidates))),frozenset(D.known),oracle)
            if len(trace)!=len(qids) or len(oracle.trace)!=len(qids):
                raise RuntimeError('C1 query charging mismatch')
            new_calls+=len(trace)
            ledger={r['window_id']:r['label'] for r in trace};prefix={};recovery={}
            for q in range(len(trace)+1):
                cached=BudgetedOracle(ledger.__getitem__,ledger,q)
                ee,tt=C.purchase(candidates,list(range(q)),frozenset(D.known),cached)
                if tt!=trace[:q]:
                    raise RuntimeError('Prefix replay mismatch')
                labels=np.asarray(list(D.known)+[r['label'] for r in ee])
                protos=np.asarray([*kp]+[r['signature'] for r in ee])
                prefix[str(q)]=labels[(emb[evalix]@protos.T).argmax(1)].tolist()
                recovery[str(q)]=len(ee)
            if prefix['0']!=zero:
                raise RuntimeError('C1 known-only mismatch')
            predictions[METHOD]=prefix
            details[METHOD]=dict(trace=trace,oracle_trace=list(oracle.trace),queries=len(trace),recovered=len(entries),
                prefix_recovery=recovery,known_answers=sum(t['status']=='known' for t in trace),
                duplicate_answers=sum(t['status']=='duplicate_novel' for t in trace),origin='new_C1_simulation')
            prediction_file=out/(stem+'_PREDICTIONS.json')
            save(prediction_file,dict(window_ids=ids[evalix].tolist(),predictions=predictions))
            seal=sha(prediction_file)
            save(out/(stem+'_PREDICTION_SEAL.json'),dict(sha256=seal,plan_sha256=sha(plan_file)))
            # Rescore ALL 26 arms on a new shared unpaid denominator, retaining
            # original D4 scores in their immutable run directory.
            union={w for plan in plans.values() for w in plan}
            truth=D.activities[evalix]
            common=~np.isin(ids[evalix],list(union))
            for method,detail in details.items():
                limit();own=~np.isin(ids[evalix],plans[method]);scores={}
                for q,pred in predictions[method].items():
                    pred=np.asarray(pred)
                    scores[q]=dict(full=D3.metrics(truth,pred,D.known,D.novel),
                        common_unpaid=D3.metrics(truth[common],pred[common],D.known,D.novel),
                        own_unpaid=D3.metrics(truth[own],pred[own],D.known,D.novel))
                primary=scores[str(detail['queries'])]['common_unpaid']
                detail.update(method=method,dataset=dataset,fold=fold['fold'],prefix_metrics=scores,primary_metrics=primary,
                    full_windows=len(evalix),common_unpaid_windows=int(common.sum()),prediction_sha256=seal)
                save(out/f'{stem}_{method}.json',detail)
                rows.append(dict(dataset=dataset,fold=fold['fold'],method=method,queries=detail['queries'],
                    recovered=detail['recovered'],novel_f1=primary['novel_macro_f1'],known_f1=primary['known_macro_f1'],
                    joint_f1=primary['combined_macro_f1'],common_unpaid_windows=int(common.sum())))
            if sha(prediction_file)!=seal:
                raise RuntimeError('Predictions changed during scoring')
            save(out/'RESULTS.json',rows)
            print('C1_FOLD_COMPLETE',dataset,fold['fold'],len(rows),flush=True)
    if len(rows)!=234 or new_calls>36:
        raise RuntimeError('Unexpected final condition/query count')
    save(out/'C1_RESULT.json',dict(status='complete',conditions=len(rows),new_simulated_calls=new_calls,
        reused_ledger_calls=reused_calls,model_calls=0,optimizer_steps=0,elapsed_seconds=time.monotonic()-start,
        interpretation='Inspected development; independent aggregate-centroid sensor adaptation; no native reproduction or independent generalization claim'))
    signal.alarm(0)
    print('FINISHED: C1 development only. Preserve every result and await audit.',flush=True)


if __name__=='__main__':
    project,out=map(Path,sys.argv[1:3])
    try:
        run(project,out,json.loads(Path(sys.argv[3]).read_text()))
    except Exception as exc:
        signal.alarm(0)
        save(out/'C1_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
