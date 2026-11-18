"""Colab D4: independent frozen TypiClust adaptation and fixed random replications."""
from pathlib import Path
import json,signal,time,resource
import numpy as np
import torch
import d3_matched_agreement as D3
import d4_typiclust_core as TC
from data_access import BudgetedOracle
import stage_c_mechanisms as C
P,B,Pilot=D3.P,D3.B,D3.Pilot
save,sha=D3.save,D3.sha
OLD=('O_all','O_agreed','H_all','H_agreed')

def timeout(signum,frame):raise TimeoutError('D4 resource ceiling; preserve all outputs')

def run(project,out,contract):
    begin=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    torch.set_num_threads(4);assert B.runtime()==contract['runtime']
    P.DEVICE=torch.device('cpu');P.PROJECT=project
    for ds in P.DATASETS.values():
        for key in ('cache','manifest','models'):ds[key]=project/ds[key].relative_to(Path(P.CFG.project))
    for item in contract['inputs']:assert sha(project/item['path'])==item['sha256'],item['path']
    P.load=P.encode=P.Frozen.embedding=P.Encoder=D3.blocked;P.assert_locked()
    save(out/'PREFLIGHT.json',dict(core=TC.generated(True),oracle_and_scorer=D3.generated(),runtime=B.runtime(),model_routes_disabled=True))
    assert not (out/'STARTED.json').exists();save(out/'STARTED.json',dict(model_calls=0,optimizer_steps=0))
    summary=[];new_calls=0;reused_calls=0;reference_checks=[]
    def limit():
        remaining=900-(time.monotonic()-begin)
        if remaining<=0:raise TimeoutError('Overall D4 limit')
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024*1024,'RSS limit'
        signal.alarm(max(1,min(60,int(remaining))))
    for name in ('realworld','pamap2'):
        man=json.loads(P.DATASETS[name]['manifest'].read_text(encoding='utf-8-sig'))
        for fold in man['outer_folds']:
            limit();D,ids,fold,emb,known,cal,prior=Pilot.load_case(P,project,name,fold['fold'])
            stem=f'{name}_fold_{fold["fold"]}';oldbase=project/contract['d3_base']
            oldseal=json.loads((oldbase/(stem+'_PREDICTION_SEAL.json')).read_text())
            assert sha(oldbase/(stem+'_PREDICTIONS.json'))==oldseal['sha256']
            assert sha(oldbase/(stem+'_CANDIDATES.json'))==oldseal['candidate_sha256']
            assert oldseal['candidate_sha256']==json.loads((oldbase/(stem+'_CANDIDATE_SEAL.json')).read_text())['sha256']
            oldc=json.loads((oldbase/(stem+'_CANDIDATES.json')).read_text())
            pools=oldc['O_all']['reference']['pools'];poolids=sorted(w for pool in pools for w in pool['window_ids'])
            assert len(poolids)==len(set(poolids)) and len(poolids)<=1800
            index={int(w):i for i,w in enumerate(ids)};z=emb[[index[w] for w in poolids]]
            plans={m:[oldc[m]['reference']['candidates'][i]['medoid'] for i in oldc[m]['order']] for m in OLD}
            limit();typical=TC.select(poolids,z);plans['TC_Frozen_Sensor_v1']=typical['query_ids']
            for seed in contract['random_seeds']:plans[f'random_{seed}']=TC.random_ids(poolids,seed)
            assert len(plans)==25 and all(len(x)==len(set(x))<=4 for x in plans.values())
            planfile=out/(stem+'_QUERY_PLANS.json');save(planfile,dict(plans=plans,typiclust=typical,pool_ids=poolids,
                access='TC centralized selected embeddings; existing arms retain their own logical access',query_unit='one_window'))
            save(out/(stem+'_PLAN_SEAL.json'),dict(sha256=sha(planfile)))
            oldpred=json.loads((oldbase/(stem+'_PREDICTIONS.json')).read_text())
            evalix=np.flatnonzero(np.isin(D.subjects,fold[D.ds['eval_key']]))
            assert oldpred['window_ids']==ids[evalix].tolist()
            kp=P.known_prototypes(D,emb,fold);zero=np.asarray(D.known)[(emb[evalix]@kp.T).argmax(1)].tolist()
            predictions={};details={};union=set();lookup={w:str(D.activities[index[w]]) for w in poolids}
            for method,queryids in plans.items():
                limit();union.update(queryids)
                if method in OLD:
                    old=json.loads((oldbase/f'{stem}_{method}.json').read_text())
                    assert [r['window_id'] for r in old['trace']]==queryids
                    assert all(r['label']==lookup[r['window_id']] for r in old['trace'])
                    predictions[method]=oldpred['predictions'][method]
                    assert predictions[method]['0']==zero
                    details[method]=dict(method=method,trace=old['trace'],oracle_trace=old['oracle_trace'],queries=old['queries'],
                        recovered=old['recovered'],prefix_recovery=old['prefix_recovery'],known_answers=old['known_answers'],
                        duplicate_answers=old['duplicate_answers'],origin='reused_exact_D3_ledger_and_predictions')
                    reused_calls+=old['queries'];reference_checks.append([name,fold['fold'],method])
                else:
                    candidates=[dict(medoid=w,support=1,window_ids=[w],signature=emb[index[w]]) for w in queryids]
                    oracle=BudgetedOracle(lookup.__getitem__,poolids,4)
                    entries,trace=C.purchase(candidates,list(range(len(candidates))),frozenset(D.known),oracle)
                    assert len(trace)==len(oracle.trace)==len(queryids)<=4
                    new_calls+=len(trace);ledger={r['window_id']:r['label'] for r in trace};prefix={};recovered={}
                    for q in range(len(trace)+1):
                        cached=BudgetedOracle(ledger.__getitem__,ledger,q)
                        ee,tt=C.purchase(candidates,list(range(q)),frozenset(D.known),cached)
                        assert tt==trace[:q]
                        labels=np.asarray(list(D.known)+[r['label'] for r in ee]);protos=np.asarray([*kp]+[r['signature'] for r in ee])
                        prefix[str(q)]=labels[(emb[evalix]@protos.T).argmax(1)].tolist();recovered[str(q)]=len(ee)
                    assert prefix['0']==zero
                    predictions[method]=prefix;details[method]=dict(method=method,trace=trace,oracle_trace=list(oracle.trace),queries=len(trace),
                        recovered=len(entries),prefix_recovery=recovered,known_answers=sum(t['status']=='known' for t in trace),
                        duplicate_answers=sum(t['status']=='duplicate_novel' for t in trace),origin='new_independent_simulation')
                print('D4_POLICY_READY',name,fold['fold'],method,flush=True)
            pfile=out/(stem+'_PREDICTIONS.json');save(pfile,dict(window_ids=ids[evalix].tolist(),predictions=predictions))
            seal=sha(pfile);save(out/(stem+'_PREDICTION_SEAL.json'),dict(sha256=seal,plan_sha256=sha(planfile)))
            # All policies were fixed and predictions sealed before evaluator truth.
            truth=D.activities[evalix];common=~np.isin(ids[evalix],list(union))
            for method,detail in details.items():
                limit();own=~np.isin(ids[evalix],plans[method]);scores={}
                for q,pred in predictions[method].items():
                    pred=np.asarray(pred)
                    scores[q]=dict(full=D3.metrics(truth,pred,D.known,D.novel),common_unpaid=D3.metrics(truth[common],pred[common],D.known,D.novel),
                        own_unpaid=D3.metrics(truth[own],pred[own],D.known,D.novel))
                primary=scores[str(detail['queries'])]['common_unpaid']
                detail.update(dataset=name,fold=fold['fold'],prefix_metrics=scores,primary_metrics=primary,full_windows=len(evalix),
                    common_unpaid_windows=int(common.sum()),own_unpaid_windows=int(own.sum()),prediction_sha256=seal,
                    own_population_definition='Excludes the arm full-budget queried IDs for every prefix; no prefix-dependent denominator')
                save(out/f'{stem}_{method}.json',detail)
                summary.append(dict(dataset=name,fold=fold['fold'],method=method,queries=detail['queries'],recovered=detail['recovered'],
                    novel_f1=primary['novel_macro_f1'],known_f1=primary['known_macro_f1'],joint_f1=primary['combined_macro_f1'],
                    full_novel_f1=scores[str(detail['queries'])]['full']['novel_macro_f1'],common_unpaid_windows=int(common.sum())))
            assert sha(pfile)==seal
            save(out/'RESULTS.json',summary);print('D4_FOLD_COMPLETE',name,fold['fold'],flush=True)
    assert len(summary)==225 and len(reference_checks)==36 and new_calls<=756 and reused_calls+new_calls<=900
    random=[]
    for name in ('realworld','pamap2'):
        for fid in sorted({r['fold'] for r in summary if r['dataset']==name}):
            rows=[r for r in summary if r['dataset']==name and r['fold']==fid and r['method'].startswith('random_')];assert len(rows)==20
            a=np.asarray([r['novel_f1'] for r in rows]);random.append(dict(dataset=name,fold=fid,seeds=20,mean=float(a.mean()),std=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max())))
    save(out/'RANDOM_WITHIN_FOLD.json',random)
    signal.alarm(0)
    result=dict(status='complete',conditions=225,new_simulated_oracle_calls=new_calls,reused_D3_ledger_calls=reused_calls,model_calls=0,optimizer_steps=0,
        reference_checks=reference_checks,elapsed_seconds=time.monotonic()-begin,
        interpretation='Inspected development; TC has centralized selected-feature access. No native TypiClust or independent validation claim.')
    save(out/'D4_RESULT.json',result);print('FINISHED',json.dumps(result),flush=True)

if __name__=='__main__':
    import sys,traceback
    project,out=map(Path,sys.argv[1:3]);contract=json.loads(Path(sys.argv[3]).read_text())
    try:run(project,out,contract)
    except Exception as e:
        signal.alarm(0);save(out/'D4_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
