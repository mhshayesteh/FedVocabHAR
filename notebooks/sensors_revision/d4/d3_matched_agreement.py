"""Colab-only saved-embedding acquisition comparison; no model computation."""
from pathlib import Path
import hashlib,json,signal,time,resource
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import frozen_audit as P
import stage_b_diagnosis as B
import stage_c_pilot as Pilot
import stage_c_mechanisms as C
import d3_matched_core as M
import reference_acquisition as A
from data_access import CandidateView,BudgetedOracle

METHODS=M.METHODS
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blocked(*args,**kwargs):raise RuntimeError('Model/encoding/checkpoint route disabled in D3')
def timeout(signum,frame):raise TimeoutError('D3 resource ceiling; preserve outputs')
def canonical(rows):return sorted((r['client'],tuple(sorted(r['window_ids'])),r['medoid']) for r in rows)
def metrics(truth,pred,known,novel):
    vocab=list(known)+list(novel)
    _,recall,f,support=precision_recall_fscore_support(truth,pred,labels=vocab,zero_division=0);n=len(known)
    return dict(known_macro_f1=float(f[:n].mean()),novel_macro_f1=float(f[n:].mean()),combined_macro_f1=float(f.mean()),per_class_f1=dict(zip(vocab,map(float,f))), per_class_recall=dict(zip(vocab,map(float,recall))),per_class_support=dict(zip(vocab,map(int,support))))
def generated():
    o=BudgetedOracle({1:'known',2:'new',3:'new'}.__getitem__,[1,2,3],3)
    c=[dict(medoid=i,support=1,signature=np.eye(3)[i-1],window_ids=[i]) for i in [1,2,3]]
    entries,trace=C.purchase(c,[0,1,2],frozenset(['known']),o)
    assert len(trace)==3 and len(entries)==1 and [r['status'] for r in trace]==['known','promoted','duplicate_novel']
    assert o.remaining==0
    try:o.query(1)
    except RuntimeError:pass
    else:raise AssertionError('Budget exceeded')
    ledger={r['window_id']:r['label'] for r in trace}
    ee,tt=C.purchase(c,[0,1],frozenset(['known']),BudgetedOracle(ledger.__getitem__,ledger,2))
    assert tt==trace[:2] and len(ee)==1
    fixture=metrics(['a','b'],['a','a'],['a'],['b','c'])
    assert fixture['novel_macro_f1']==0 and abs(fixture['known_macro_f1']-2/3)<1e-12
    assert abs(fixture['combined_macro_f1']-2/9)<1e-12
    return dict(budget_duplicate_known_costs=True,prefix_ledger_reuse=True,fixed_missing_class_denominator=True)

def run(project,out,contract):
    begin=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    torch.set_num_threads(4)
    assert B.runtime()==contract['runtime'],B.runtime()
    P.DEVICE=torch.device('cpu');P.PROJECT=project
    for ds in P.DATASETS.values():
        for key in ('cache','manifest','models'):ds[key]=project/ds[key].relative_to(Path(P.CFG.project))
    for item in contract['inputs']:
        assert sha(project/item['path'])==item['sha256'],item['path']
    P.load=P.encode=P.Frozen.embedding=P.Encoder=blocked
    P.assert_locked();save(out/'PREFLIGHT.json',dict(generated=generated(),matched_core=M.generated(),runtime=B.runtime(),model_routes_disabled=True))
    assert not (out/'STARTED.json').exists();save(out/'STARTED.json',dict(model_calls=0,optimizer_steps=0))
    d1=json.loads((project/contract['d1_geometry']).read_text());summaries=[];references=[];contrasts=[];calls=0
    def limit():
        remaining=900-(time.monotonic()-begin)
        if remaining<=0:raise TimeoutError('Overall D3 limit')
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2*1024*1024,'RSS limit'
        signal.alarm(max(1,min(60,int(remaining))))
    for name in ('realworld','pamap2'):
        man=json.loads(P.DATASETS[name]['manifest'].read_text(encoding='utf-8-sig'))
        for fold in man['outer_folds']:
            limit();D,ids,fold,emb,prototypes,cal,prior=Pilot.load_case(P,project,name,fold['fold'])
            clients=tuple(fold[D.ds['client_key']]);mask=np.isin(D.subjects,clients)
            view=CandidateView(ids[mask],D.subjects[mask],emb[mask]);built={};stem=f'{name}_fold_{fold["fold"]}'
            for method in METHODS:
                limit();built[method]=M.construct(method,view,clients,prototypes,cal,P.cluster)
                if method in ('O_all','H_all'):
                    ref=next(r for r in d1 if (r['dataset'],r['fold'],r['condition'])==(name,fold['fold'],'original' if method=='O_all' else 'half_distance'))
                    assert canonical(built[method]['rows'])==canonical(ref['clusters'])
                    assert built[method]['groups']==ref['groups'],'D1 agreement-group mismatch'
                    references.append([name,fold['fold'],method])
            for all_arm,filtered in [('O_all','O_agreed'),('H_all','H_agreed')]:
                assert canonical(built[all_arm]['rows'])==canonical(built[filtered]['rows'])
                shared=built[all_arm]['shared_order'];eligible=set(built[filtered]['eligible'])
                assert shared==built[filtered]['shared_order']
                assert built[filtered]['order']==[i for i in shared if i in eligible][:4]
                assert all(np.array_equal(a['signature'],b['signature']) for a,b in zip(built[all_arm]['candidates'],built[filtered]['candidates']))
            candidate_file=out/(stem+'_CANDIDATES.json')
            save(candidate_file,{m:dict(reference=A.membership_reference(b['rows'],b['pools'],b['discarded'],b['groups'],b['candidates']),order=b['order'],shared_order=b['shared_order'],eligible=b['eligible'],association_margins=b['association_margins']) for m,b in built.items()})
            save(out/(stem+'_CANDIDATE_SEAL.json'),dict(sha256=sha(candidate_file)))
            eval_ix=np.flatnonzero(np.isin(D.subjects,fold[D.ds['eval_key']]))
            kp=P.known_prototypes(D,emb,fold);predictions={};details={};union=set()
            lookup=dict(zip(map(int,view.window_ids),map(str,D.activities[mask])))
            for method in METHODS:
                limit();b=built[method];oracle=BudgetedOracle(lookup.__getitem__,view.window_ids,4)
                entries,trace=C.purchase(b['candidates'],b['order'],frozenset(D.known),oracle)
                assert len(trace)==len(oracle.trace)<=4;calls+=len(trace);union.update(r['window_id'] for r in trace)
                ledger={r['window_id']:r['label'] for r in trace};prefix_pred={};recoveries={}
                for q in range(len(trace)+1):
                    initial=BudgetedOracle(ledger.__getitem__,ledger,q)
                    ee,tt=C.purchase(b['candidates'],b['order'][:q],frozenset(D.known),initial)
                    assert tt==trace[:q]
                    labels=np.asarray(list(D.known)+[r['label'] for r in ee]);proto=np.asarray([*kp]+[r['signature'] for r in ee])
                    prefix_pred[str(q)]=labels[(emb[eval_ix]@proto.T).argmax(1)].tolist();recoveries[str(q)]=len(ee)
                predictions[method]=prefix_pred
                details[method]=dict(method=method,oracle_trace=list(oracle.trace),trace=trace,queries=len(trace),
                    known_answers=sum(r['status']=='known' for r in trace),duplicate_answers=sum(r['status']=='duplicate_novel' for r in trace),
                    recovered=len(entries),prefix_recovery=recoveries,signature_basis='purchased_window_only')
            pfile=out/(stem+'_PREDICTIONS.json');save(pfile,dict(window_ids=ids[eval_ix].tolist(),predictions=predictions))
            seal=sha(pfile);save(out/(stem+'_PREDICTION_SEAL.json'),dict(sha256=seal,candidate_sha256=sha(candidate_file)))
            # Evaluator: all methods and prefixes were fixed before this truth join.
            truth=D.activities[eval_ix];common=~np.isin(ids[eval_ix],list(union));qcommon=min(d['queries'] for d in details.values())
            for method in METHODS:
                detail=details[method];scores={}
                for q,pred in predictions[method].items():
                    pred=np.asarray(pred);scores[q]=dict(full=metrics(truth,pred,D.known,D.novel),common_unpaid=metrics(truth[common],pred[common],D.known,D.novel))
                full=scores[str(detail['queries'])]['full'];detail.update(dataset=name,fold=fold['fold'],metrics=full,prefix_metrics=scores,
                    q_common=qcommon,reference_queries=details['O_all']['queries'],common_unpaid_windows=int(common.sum()),
                    full_windows=len(eval_ix),prediction_sha256=seal,truth_support={a:int(np.sum(truth==a)) for a in D.vocabulary})
                detail['primary_metrics']=scores[str(detail['queries'])]['common_unpaid']
                save(out/f'{stem}_{method}.json',detail)
                summaries.append(dict(dataset=name,fold=fold['fold'],method=method,queries=detail['queries'],recovered=detail['recovered'],
                    known_f1=detail['primary_metrics']['known_macro_f1'],novel_f1=detail['primary_metrics']['novel_macro_f1'],joint_f1=detail['primary_metrics']['combined_macro_f1'],full_novel_f1=full['novel_macro_f1'],
                    common_unpaid_novel_f1=scores[str(detail['queries'])]['common_unpaid']['novel_macro_f1']))
            for partition in ['O','H']:
                all_d,agreed_d=details[partition+'_all'],details[partition+'_agreed']
                qs=range(min(all_d['queries'],agreed_d['queries'])+1)
                contrasts.append(dict(dataset=name,fold=fold['fold'],partition=partition,
                    full_cap_novel_delta=agreed_d['primary_metrics']['novel_macro_f1']-all_d['primary_metrics']['novel_macro_f1'],
                    full_cap_known_delta=agreed_d['primary_metrics']['known_macro_f1']-all_d['primary_metrics']['known_macro_f1'],
                    matched_prefixes=[dict(q=q,zero_query=q==0,novel_delta=agreed_d['prefix_metrics'][str(q)]['common_unpaid']['novel_macro_f1']-all_d['prefix_metrics'][str(q)]['common_unpaid']['novel_macro_f1']) for q in qs]))
            o,h=contrasts[-2:];h['half_minus_original_interaction']=h['full_cap_novel_delta']-o['full_cap_novel_delta']
            save(out/'CONTRASTS.json',contrasts)
            assert sha(pfile)==seal
            save(out/'RESULTS.json',summaries);print('D3_FOLD_COMPLETE',name,fold['fold'],flush=True)
    signal.alarm(0);assert len(summaries)==45 and len(references)==18 and calls<=180
    result=dict(status='complete',episodes=45,simulated_oracle_calls=calls,model_calls=0,optimizer_steps=0,
                reference_checks=references,elapsed_seconds=time.monotonic()-begin,interpretation='Matched diagnostic on inspected development only; singleton decoder differs from historical D2; no novel-method admission.')
    save(out/'D3_RESULT.json',result);print('FINISHED',json.dumps(result),flush=True)

if __name__=='__main__':
    import sys,traceback
    project,out=map(Path,sys.argv[1:3]);contract=json.loads(Path(sys.argv[3]).read_text())
    try:run(project,out,contract)
    except Exception as e:
        signal.alarm(0);save(out/'D3_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
