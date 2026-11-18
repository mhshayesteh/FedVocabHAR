"""Colab-only M1 external transfer. Source sealed before first target download."""
from pathlib import Path,PurePosixPath
from html.parser import HTMLParser
from urllib.request import urlopen,Request
from urllib.parse import urljoin,urlparse,unquote
import io,json,re,resource,signal,time,zipfile,hashlib
import numpy as np
import torch
import frozen_audit as P
import stage_b_diagnosis as B
import stage_c_pilot as Pilot
import stage_c_mechanisms as C
import d3_matched_core as M
import d4_typiclust_core as TC
import m1_mhealth_core as K
from data_access import CandidateView,BudgetedOracle

def sha(p):return B.sha(p)
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def timeout(signum,frame):raise TimeoutError('M1 resource ceiling: preserve outputs; no automatic retry')

class ArchiveLinks(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if '/static/public/319/' in unquote(href) and unquote(href).lower().endswith('.zip'):self.links.append(href)

def official_archive(out,metadata_url,limit):
    limit(300)
    with urlopen(Request(metadata_url,headers={'User-Agent':'FedVocabHAR-M1/1.0'}),timeout=60) as response:
        html=response.read(4000001);assert len(html)<=4000000,'Metadata page size'
    (out/'UCI_METADATA_PAGE.html').write_bytes(html)
    parser=ArchiveLinks();parser.feed(html.decode('utf-8'))
    urls=sorted(set(urljoin(metadata_url,h) for h in parser.links));assert len(urls)==1,'Official archive link ambiguous/missing'
    url=urls[0];assert urlparse(url).scheme=='https' and urlparse(url).hostname=='archive.ics.uci.edu'
    path=out/'RAW_MHEALTH.zip';assert not path.exists()
    total=0
    with urlopen(Request(url,headers={'User-Agent':'FedVocabHAR-M1/1.0'}),timeout=60) as response,path.open('xb') as f:
        assert urlparse(response.geturl()).hostname=='archive.ics.uci.edu','Archive redirect host changed'
        while True:
            block=response.read(1048576)
            if not block:break
            total+=len(block);assert total<=150*1024**2,'Archive byte cap';f.write(block)
    save(out/'RAW_ARCHIVE_IDENTITY.json',dict(official_page=metadata_url,archive_url=url,bytes=total,sha256=sha(path),metadata_sha256=sha(out/'UCI_METADATA_PAGE.html'),
        identity='First fetched byte identity; not a previously verified official checksum'))
    return path

def curate_archive(path,out,mapping,limit):
    xs=[];ys=[];ids=[];subjects=[];records=[]
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None,'Archive CRC'
        infos=z.infolist();assert sum(i.file_size for i in infos)<=600*1024**2 and len(infos)<=100,'Expanded archive caps'
        logs={};docs=[]
        for entry in infos:
            pure=PurePosixPath(entry.filename)
            assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in entry.filename and ':' not in entry.filename,'Unsafe archive path'
            assert ((entry.external_attr>>16)&0o170000)!=0o120000,'Archive symlink'
            if entry.is_dir():continue
            match=re.fullmatch(r'mHealth_subject([1-9]|10)\.log',pure.name)
            if match:
                subject=int(match.group(1));assert subject not in logs and entry.file_size<=60*1024**2
                logs[subject]=entry
            else:
                assert entry.file_size<=2*1024**2 and pure.suffix.lower() in ['.txt','.md','.pdf','.html'],'Unexpected archive member/version'
                docs.append(entry)
        assert sorted(logs)==list(range(1,11)),'Missing/extra subject logs'
        for j,entry in enumerate(docs):
            (out/f'ARCHIVE_DOCUMENT_{j}{PurePosixPath(entry.filename).suffix}').write_bytes(z.read(entry))
        for subject in range(1,11):
            limit(120);raw=z.read(logs[subject]);values=np.loadtxt(io.BytesIO(raw),dtype=np.float64,ndmin=2)
            x,y,w,info=K.curate(values,subject,mapping)
            assert len(x)>0,'No eligible windows for scheduled subject; preserve structural failure'
            xs.append(x);ys.append(y);ids.append(w);subjects.append(np.full(len(w),subject,np.int64))
            records.append(dict(info,file=logs[subject].filename,file_bytes=len(raw),file_sha256=hashlib.sha256(raw).hexdigest()))
            del raw,values
            print('M1_CURATED_SUBJECT',subject,flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys);wid=np.concatenate(ids);sub=np.concatenate(subjects)
    assert len(wid)==len(set(map(int,wid)))
    np.savez_compressed(out/'MHEALTH_WINDOWS.npz',x=x,activity=y,window_ids=wid,subject=sub)
    save(out/'CURATION_RECORDS.json',records)
    filehash={r['subject']:r['file_sha256'] for r in records}
    save(out/'WINDOW_ID_LOOKUP.json',[dict(window_id=int(w),subject=int(s),original_start_row=int(w)-int(s)*1000000000,file_sha256=filehash[int(s)]) for w,s in zip(wid,sub)])
    return x,y,wid,sub

def run(project,out,execution):
    begin=time.monotonic();signal.signal(signal.SIGALRM,timeout)
    def limit(seconds=120):
        remaining=1800-(time.monotonic()-begin)
        if remaining<=0:raise TimeoutError('M1 overall cap')
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'RSS3GiB cap'
        signal.alarm(max(1,min(seconds,int(remaining))))
    limit(180);torch.set_num_threads(4)
    contract=json.loads(Path(execution['scientific_contract']).read_text());assert B.runtime()==contract['runtime']
    acquisition=contract['target']['acquisition_subjects'];evaluation=contract['target']['evaluation_subjects'];K.validate_roles(acquisition,evaluation)
    provenance=json.loads(Path(execution['source_provenance']).read_text())
    for pin in contract['source_inputs']:
        rel=pin['path']
        if rel in provenance:
            src=provenance[rel];assert src['original_sha256']==pin['sha256']
            assert sha(Path(src['embedded_name']))==src['normalized_sha256']
        elif 'expected_metadata' in pin:assert json.loads((project/rel).read_text())==pin['expected_metadata']
        else:assert sha(project/rel)==pin['sha256'],rel
    from d3_matched_agreement import generated as oracle_checks
    generated=dict(data_and_metrics=K.generated(),matched=M.generated(),typiclust=TC.generated(True),oracle=oracle_checks())
    # Compare the independent scorer against sklearn on missing-class fixture.
    from d3_matched_agreement import metrics as reference_metrics
    known=contract['target']['known'];novel=contract['target']['novel']
    assert K.metrics([known[0],novel[0]],[known[0],known[0]],known,novel)==reference_metrics([known[0],novel[0]],[known[0],known[0]],known,novel)
    save(out/'PREFLIGHT.json',dict(runtime=B.runtime(),generated=generated,source_pins_checked=True))
    assert not (out/'STARTED.json').exists();save(out/'STARTED.json',dict(target_download_started=False,optimizer_steps=0))
    P.PROJECT=project;P.DEVICE=torch.device('cpu')
    for ds in P.DATASETS.values():
        for key in ('cache','manifest','models'):ds[key]=project/ds[key].relative_to(Path(P.CFG.project))
    P.assert_locked();limit(180)
    D,sourceids,fold,emb,calproto,cal,prior=Pilot.load_case(P,project,'realworld',1)
    kp=P.known_prototypes(D,emb,fold);assert cal==contract['source']['calibration'] and tuple(known)==D.known
    bundle=P.load(project/contract['source']['checkpoint']);assert bundle.get('input_scale',1.0)==1.0
    model=P.Encoder(6).to(P.DEVICE);model.load_state_dict(P.adapt_checkpoint_keys(bundle['anchor_state_dict']),strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    stats=bundle['stats']
    for v in ['xyz_std','mag_std']:assert np.isfinite(stats[v]).all() and np.all(stats[v]>0)
    fixture=np.arange(2*4*128,dtype=np.float32).reshape(2,4,128)/100
    stub=object.__new__(P.Frozen);stub.x=fixture
    assert np.array_equal(K.normalize(fixture,stats),stub.normalized_input(stats,1.0)),'Source normalization parity'
    np.savez_compressed(out/'SOURCE_TRANSFER_BUNDLE.npz',calibration_prototypes=calproto,recognition_prototypes=kp,**stats)
    save(out/'SOURCE_TRANSFER_SEAL.json',dict(sha256=sha(out/'SOURCE_TRANSFER_BUNDLE.npz'),calibration=cal,
        checkpoint_sha256=sha(project/contract['source']['checkpoint']),source_embedding_sha256=prior['cache']['embedding_sha256'],target_access=False,normalization_parity=True))
    # No MHEALTH fetch may precede this sealed source-only bundle.
    del D,sourceids,emb,stub,bundle
    print('M1_SOURCE_BUNDLE_SEALED; starting first official MHEALTH fetch',flush=True)
    raw=official_archive(out,contract['target']['official_metadata_url'],limit)
    mapping={int(k):v for k,v in contract['target']['label_mapping'].items()}
    x,truth,wids,subjects=curate_archive(raw,out,mapping,limit)
    limit(900);normalized=K.normalize(x,stats);assert np.isfinite(normalized).all()
    target=P.encode(model,normalized,batch=512);del model,normalized,x
    assert target.shape==(len(wids),64) and np.isfinite(target).all() and np.all(np.linalg.norm(target,axis=1)>0)
    np.savez_compressed(out/'MHEALTH_EMBEDDINGS.npz',window_ids=wids,embeddings=target)
    save(out/'EMBEDDING_SEAL.json',dict(sha256=sha(out/'MHEALTH_EMBEDDINGS.npz'),windows_sha256=sha(out/'MHEALTH_WINDOWS.npz'),
        source_transfer_sha256=sha(out/'SOURCE_TRANSFER_BUNDLE.npz'),runtime=B.runtime(),encoder_forward_batches=(len(wids)+511)//512,training_steps=0))
    mask=np.isin(subjects,acquisition);view=CandidateView(wids[mask],subjects[mask],target[mask]);built={}
    for method in ['O_all','O_agreed','H_all','H_agreed']:
        limit();built[method]=M.construct(method,view,tuple(acquisition),calproto,cal,P.cluster)
    pools=built['O_all']['pools'];poolids=sorted(w for p in pools for w in p['window_ids'])
    assert len(poolids)==len(set(poolids))<=3000
    for b in built.values():assert sorted(w for p in b['pools'] for w in p['window_ids'])==poolids
    index={int(w):i for i,w in enumerate(wids)}
    plans={m:[b['candidates'][i]['medoid'] for i in b['order']] for m,b in built.items()}
    limit();typical=TC.select(poolids,target[[index[w] for w in poolids]])
    plans['TC_Frozen_Sensor_v1']=typical['query_ids']
    for seed in contract['random_seeds']:plans[f'random_{seed}']=TC.random_ids(poolids,seed)
    plans['known_only']=[];assert set(plans)==set(contract['methods']) and len(plans)==26
    assert all(len(q)==len(set(q))<=4 and set(q)<=set(poolids) for q in plans.values())
    import reference_acquisition as A
    candidates={m:dict(reference=A.membership_reference(b['rows'],b['pools'],b['discarded'],b['groups'],b['candidates']),order=b['order'],shared_order=b['shared_order'],eligible=b['eligible']) for m,b in built.items()}
    save(out/'CANDIDATES.json',candidates)
    save(out/'QUERY_PLANS.json',dict(plans=plans,pool_ids=poolids,typiclust=typical,acquisition_subjects=acquisition,evaluation_subjects=evaluation))
    save(out/'PLAN_SEAL.json',dict(sha256=sha(out/'QUERY_PLANS.json'),candidates_sha256=sha(out/'CANDIDATES.json')))
    evalmask=np.isin(subjects,evaluation);ez=target[evalmask];evalw=wids[evalmask];evalsubjects=subjects[evalmask]
    assert not set(map(int,evalw))&set(poolids)
    lookup={w:str(truth[index[w]]) for w in poolids};predictions={};details={};calls=0
    for method,qids in plans.items():
        limit();cs=[dict(medoid=w,support=1,signature=target[index[w]],window_ids=[w]) for w in qids]
        if method in built:cs=[built[method]['candidates'][i] for i in built[method]['order']]
        oracle=BudgetedOracle(lookup.__getitem__,poolids,4)
        entries,trace=C.purchase(cs,list(range(len(cs))),frozenset(known),oracle)
        assert len(trace)==len(oracle.trace)==len(qids)<=4;calls+=len(trace)
        ledger={r['window_id']:r['label'] for r in trace};prefix={};recovered={}
        for q in range(len(trace)+1):
            ee,tt=C.purchase(cs,list(range(q)),frozenset(known),BudgetedOracle(ledger.__getitem__,ledger,q))
            assert tt==trace[:q]
            labels=np.asarray(known+[e['label'] for e in ee]);protos=np.asarray([*kp]+[e['signature'] for e in ee])
            prefix[str(q)]=labels[(ez@protos.T).argmax(1)].tolist();recovered[str(q)]=len(ee)
        predictions[method]=prefix;details[method]=dict(method=method,queries=len(trace),recovered=len(entries),trace=trace,oracle_trace=list(oracle.trace),prefix_recovery=recovered,
            known_answers=sum(r['status']=='known' for r in trace),duplicate_answers=sum(r['status']=='duplicate_novel' for r in trace))
        print('M1_POLICY_PREDICTIONS_READY',method,flush=True)
    assert calls<=100
    save(out/'PREDICTIONS.json',dict(window_ids=evalw.tolist(),subject_ids=evalsubjects.tolist(),predictions=predictions))
    save(out/'PREDICTION_SEAL.json',dict(sha256=sha(out/'PREDICTIONS.json'),plan_sha256=sha(out/'QUERY_PLANS.json')))
    # Only now expose evaluation truth to scoring/analyst outcomes.
    y=truth[evalmask];rows=[]
    for method,detail in details.items():
        limit();scores={q:K.subject_metrics(y,np.asarray(p),evalsubjects,known,novel,evaluation) for q,p in predictions[method].items()}
        detail['prefix_metrics']=scores;detail['primary_metrics']=scores[str(detail['queries'])]['subject_mean']
        save(out/(method+'.json'),detail)
        rows.append(dict(method=method,queries=detail['queries'],recovered=detail['recovered'],**detail['primary_metrics'],pooled_novel_f1=scores[str(detail['queries'])]['pooled']['novel_macro_f1']))
    save(out/'RESULTS.json',rows)
    contrasts=[]
    for cutoff in ['O','H']:
        all_=details[cutoff+'_all'];ag=details[cutoff+'_agreed'];q=min(all_['queries'],ag['queries'])
        per_subject={str(s):{k:ag['prefix_metrics'][str(ag['queries'])]['per_subject'][str(s)][k]-all_['prefix_metrics'][str(all_['queries'])]['per_subject'][str(s)][k]
            for k in ['known_macro_f1','novel_macro_f1','combined_macro_f1']} for s in evaluation}
        contrasts.append(dict(cutoff=cutoff,agreed_minus_all={k:ag['primary_metrics'][k]-all_['primary_metrics'][k] for k in all_['primary_metrics']},
            queries_all=all_['queries'],queries_agreed=ag['queries'],matched_q=q,
            per_subject_deltas=per_subject,known_loss_over_0_02_subjects=[s for s in evaluation if per_subject[str(s)]['known_macro_f1']<-.02-1e-12],
            matched_novel_delta=ag['prefix_metrics'][str(q)]['subject_mean']['novel_macro_f1']-all_['prefix_metrics'][str(q)]['subject_mean']['novel_macro_f1']))
    save(out/'CONTRASTS.json',dict(contrasts=contrasts,half_minus_original_interaction=contrasts[1]['agreed_minus_all']['novel_macro_f1']-contrasts[0]['agreed_minus_all']['novel_macro_f1']))
    random=[]
    for subject in evaluation:
        a=np.asarray([details[f'random_{seed}']['prefix_metrics'][str(details[f'random_{seed}']['queries'])]['per_subject'][str(subject)]['novel_macro_f1'] for seed in contract['random_seeds']])
        means={k:float(np.mean([details[f'random_{seed}']['prefix_metrics'][str(details[f'random_{seed}']['queries'])]['per_subject'][str(subject)][k] for seed in contract['random_seeds']]))
            for k in ['known_macro_f1','novel_macro_f1','combined_macro_f1']}
        random.append(dict(subject=subject,mean=float(a.mean()),std=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max()),seeds=len(a),mean_metrics=means))
    save(out/'RANDOM_SUBJECT_SUMMARY.json',dict(subjects=random,mean_subject_novel_f1=float(np.mean([r['mean'] for r in random])),
        mean_subject_metrics={k:float(np.mean([r['mean_metrics'][k] for r in random])) for k in ['known_macro_f1','novel_macro_f1','combined_macro_f1']}))
    # Post-seal evaluator diagnostics only: candidate support label coverage/purity.
    diagnostics={}
    for method,b in built.items():
        diagnostics[method]=dict(local_medoid_novel_coverage=sorted({str(truth[index[r['medoid']]]) for r in b['rows']} & set(novel)),
            eligible_medoid_novel_coverage=sorted({str(truth[index[b['candidates'][i]['medoid']]]) for i in b['eligible']} & set(novel)),
            selected_pool_novel_coverage=sorted(set(lookup.values())&set(novel)))
    save(out/'EVALUATOR_DIAGNOSTICS.json',diagnostics)
    assert sha(out/'PREDICTIONS.json')==json.loads((out/'PREDICTION_SEAL.json').read_text())['sha256']
    signal.alarm(0);result=dict(status='complete',conditions=26,simulated_oracle_calls=calls,optimizer_steps=0,target_training=False,
        target_encoder_forward_batches=(len(wids)+511)//512,source_bundle_sealed_before_target_download=True,predictions_sealed_before_scoring=True,
        acquisition_subjects=acquisition,evaluation_subjects=evaluation,known_classes=4,novel_classes=8,elapsed_seconds=time.monotonic()-begin,
        interpretation='One external transfer episode, conditional provided centralized encoder and reported prior non-use; no end-to-end federated/nativeTC/significance claim')
    save(out/'M1_RESULT.json',result);print('FINISHED',json.dumps(result),flush=True)

if __name__=='__main__':
    import sys,traceback
    project,out=map(Path,sys.argv[1:3]);execution=json.loads(Path(sys.argv[3]).read_text())
    try:run(project,out,execution)
    except Exception as e:
        signal.alarm(0);save(out/'M1_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
