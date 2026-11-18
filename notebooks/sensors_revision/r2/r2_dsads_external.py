"""Colab-only R2 external transfer. Source sealed before first target download."""
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
def timeout(signum,frame):raise TimeoutError('R2 resource ceiling: preserve outputs; no automatic retry')

class ArchiveLinks(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if '/static/public/256/' in unquote(href) and unquote(href).lower().endswith('.zip'):self.links.append(href)

def official_archive(out,metadata_url,limit):
    limit(300)
    with urlopen(Request(metadata_url,headers={'User-Agent':'FedVocabHAR-R2/1.0'}),timeout=60) as response:
        html=response.read(4000001);assert len(html)<=4000000,'Metadata page size'
    (out/'UCI_METADATA_PAGE.html').write_bytes(html)
    parser=ArchiveLinks();parser.feed(html.decode('utf-8'))
    urls=sorted(set(urljoin(metadata_url,h) for h in parser.links));assert len(urls)==1,'Official archive link ambiguous/missing'
    url=urls[0];assert urlparse(url).scheme=='https' and urlparse(url).hostname=='archive.ics.uci.edu'
    path=out/'RAW_DSADS.zip';assert not path.exists()
    total=0
    with urlopen(Request(url,headers={'User-Agent':'FedVocabHAR-R2/1.0'}),timeout=60) as response,path.open('xb') as f:
        assert urlparse(response.geturl()).hostname=='archive.ics.uci.edu','Archive redirect host changed'
        while True:
            block=response.read(1048576)
            if not block:break
            total+=len(block);assert total<=220*1024**2,'Archive byte cap';f.write(block)
    save(out/'RAW_ARCHIVE_IDENTITY.json',dict(official_page=metadata_url,archive_url=url,bytes=total,sha256=sha(path),metadata_sha256=sha(out/'UCI_METADATA_PAGE.html'),
        identity='First fetched byte identity; not a previously verified official checksum'))
    return path

def curate_dsads(path,out,limit):
    """Accept official direct-data ZIP or one nested data.zip; no filesystem extraction."""
    from r2_dsads_core import segment_windows,opaque_ids,MAPPING
    def validate(z):
        infos=z.infolist()
        if len(infos)>10000 or sum(i.file_size for i in infos)>650*1024**2:raise ValueError('Archive expansion cap')
        for e in infos:
            p=PurePosixPath(e.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in e.filename or ':' in e.filename or ((e.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe archive member')
        return infos
    with zipfile.ZipFile(path) as outer:
        infos=validate(outer);nested=[e for e in infos if not e.is_dir() and PurePosixPath(e.filename).suffix.lower()=='.zip']
        if nested:
            if len(nested)!=1 or PurePosixPath(nested[0].filename).name.lower()!='data.zip' or nested[0].file_size>220*1024**2:
                raise ValueError('Unexpected nested archive')
            for e in infos:
                if e.is_dir() or e==nested[0]:continue
                if e.file_size>2*1024**2 or PurePosixPath(e.filename).suffix.lower() not in ('.txt','.md','.pdf','.html'):
                    raise ValueError('Unexpected wrapper member')
                (out/('WRAPPER_'+hashlib.sha256(e.filename.encode()).hexdigest()[:12]+PurePosixPath(e.filename).suffix)).write_bytes(outer.read(e))
            data=outer.read(nested[0]);save(out/'INNER_ARCHIVE_IDENTITY.json',dict(member=nested[0].filename,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            z=zipfile.ZipFile(io.BytesIO(data))
        else:z=outer
        try:
            members={};docs=[]
            for e in validate(z):
                if e.is_dir():continue
                match=re.search(r'(?:^|/)a([0-9]{2})/p([1-8])/s([0-9]{2})\.txt$',e.filename)
                if match:
                    a,s,g=map(int,match.groups())
                    if not 1<=a<=19 or not 1<=g<=60 or e.file_size>128*1024 or (s,a,g) in members:
                        raise ValueError('Invalid/duplicate data member')
                    members[(s,a,g)]=e
                else:
                    if e.file_size>2*1024**2 or PurePosixPath(e.filename).suffix.lower() not in ('.txt','.md','.pdf','.html'):
                        raise ValueError('Unexpected member')
                    docs.append(e)
            expected={(s,a,g) for s in range(1,9) for a in range(1,20) for g in range(1,61)}
            if set(members)!=expected:raise ValueError('Expected complete9120segment manifest')
            for j,e in enumerate(docs):(out/f'ARCHIVE_DOCUMENT_{j}{PurePosixPath(e.filename).suffix}').write_bytes(z.read(e))
            xs=[];truth=[];ids=[];subjects=[];records=[]
            for s,a,g in sorted(expected):
                limit(120);e=members[(s,a,g)];raw=z.read(e)
                values=np.loadtxt(io.BytesIO(raw),delimiter=',',dtype=np.float64,ndmin=2)
                x,w=segment_windows(values,s,a,g);xs.append(x);ids.extend(w.tolist());subjects.extend([s]*len(w));truth.extend([MAPPING[a]]*len(w))
                records.append(dict(member=e.filename,subject=s,raw_activity=a,segment=g,sha256=hashlib.sha256(raw).hexdigest(),curator_ids=w.tolist()))
            x=np.concatenate(xs);ids=np.asarray(ids,np.int64);opaque=opaque_ids(ids);subjects=np.asarray(subjects,np.int64)
            assert x.shape==(18240,4,128) and len(set(ids.tolist()))==18240
            np.savez_compressed(out/'DSADS_WINDOWS.npz',x=x,activity=np.asarray(truth),window_ids=opaque,curator_ids=ids,subject=subjects)
            save(out/'CURATION_RECORDS.json',records)
            save(out/'ID_LOOKUP.json',[dict(window_id=int(w),curator_id=int(i)) for w,i in zip(opaque,ids)])
            return x,np.asarray(truth),opaque,subjects
        finally:
            if z is not outer:z.close()

def compare_episodes(target,truth,wids,subjects,calproto,cal,kp,contract,out,limit,P):
    import r2_dsads_core as R
    import c1_aggregate_selector as AC
    index={int(w):i for i,w in enumerate(wids)};prepared={};all_plans={}
    for episode,acquisition in R.EPISODES.items():
        mask=np.isin(subjects,acquisition);view=CandidateView(wids[mask],subjects[mask],target[mask])
        built={}
        for method in ['O_all','O_agreed','H_all','H_agreed']:
            limit();built[method]=M.construct(method,view,tuple(acquisition),calproto,cal,P.cluster)
        poolids=sorted(w for p in built['O_all']['pools'] for w in p['window_ids'])
        assert len(poolids)==len(set(poolids))<=1200
        assert all(sorted(w for p in b['pools'] for w in p['window_ids'])==poolids for b in built.values())
        plans={m:[b['candidates'][i]['medoid'] for i in b['order']] for m,b in built.items()}
        limit();tc=TC.select(poolids,target[[index[w] for w in poolids]])
        plans['TC_Frozen_Sensor_v1']=tc['query_ids']
        clients=[]
        for s in acquisition:
            ww=[w for w in poolids if int(subjects[index[w]])==s]
            clients.append(AC.Client(s,ww,target[[index[w] for w in ww]]))
        limit();c1=AC.select(clients,budget=4,cap_mode='global',seed=20261117,max_iter=100,tolerance=1e-8)
        if c1['fit'] is not None and not c1['fit']['converged']:raise RuntimeError('Aggregate iteration cap')
        plans[AC.PROFILE]=[r['window_id'] for r in c1['plan']]
        for seed in contract['random_seeds']:plans[f'random_{seed}']=TC.random_ids(poolids,seed)
        plans['known_only']=[]
        assert len(plans)==27 and set(plans)==set(contract['methods'])
        assert all(len(q)==len(set(q))<=4 and set(q)<=set(poolids) for q in plans.values())
        all_plans[episode]=dict(plans=plans,pool_ids=poolids,tc=tc,aggregate=c1,acquisition_subjects=acquisition)
        prepared[episode]=(plans,built)
    save(out/'QUERY_PLANS.json',all_plans)
    save(out/'PLAN_SEAL.json',dict(sha256=sha(out/'QUERY_PLANS.json'),both_episodes_committed_before_answers=True))
    evalmask=np.isin(subjects,R.EVALUATION);ez=target[evalmask];evalw=wids[evalmask];evalsubjects=subjects[evalmask]
    predictions={};details={};calls=0
    for episode,(plans,built) in prepared.items():
        predictions[episode]={};details[episode]={}
        acquisition=R.EPISODES[episode]
        allowed=set(w for w in wids if int(subjects[index[int(w)]]) in acquisition)
        assert not allowed&set(evalw.tolist())
        lookup={int(w):str(truth[index[int(w)]]) for w in allowed}
        for method,qids in plans.items():
            limit()
            cs=[dict(medoid=w,support=1,signature=target[index[w]],window_ids=[w]) for w in qids]
            if method in built:cs=[built[method]['candidates'][i] for i in built[method]['order']]
            oracle=BudgetedOracle(lookup.__getitem__,all_plans[episode]['pool_ids'],4)
            entries,trace=C.purchase(cs,list(range(len(cs))),frozenset(R.KNOWN),oracle)
            assert len(trace)==len(oracle.trace)==len(qids)<=4;calls+=len(trace)
            ledger={r['window_id']:r['label'] for r in trace};prefix={}
            for q in range(len(trace)+1):
                ee,tt=C.purchase(cs,list(range(q)),frozenset(R.KNOWN),BudgetedOracle(ledger.__getitem__,ledger,q))
                assert tt==trace[:q]
                labels=np.asarray(R.KNOWN+[e['label'] for e in ee]);protos=np.asarray([*kp]+[e['signature'] for e in ee])
                prefix[str(q)]=labels[(ez@protos.T).argmax(1)].tolist()
            predictions[episode][method]=prefix
            details[episode][method]=dict(queries=len(trace),recovered=len(entries),trace=trace,oracle_trace=list(oracle.trace),
                known_answers=sum(r['status']=='known' for r in trace),duplicate_answers=sum(r['status']=='duplicate_novel' for r in trace))
        print('R2_EPISODE_PREDICTIONS_READY',episode,flush=True)
    assert calls<=208
    save(out/'PREDICTIONS.json',dict(window_ids=evalw.tolist(),subject_ids=evalsubjects.tolist(),predictions=predictions))
    save(out/'PREDICTION_SEAL.json',dict(sha256=sha(out/'PREDICTIONS.json'),plan_sha256=sha(out/'QUERY_PLANS.json'),both_episodes=True))
    y=truth[evalmask];rows=[];contrasts=[]
    for episode in R.EPISODES:
        for method,d in details[episode].items():
            d['prefix_metrics']={q:K.subject_metrics(y,np.asarray(p),evalsubjects,R.KNOWN,R.NOVEL,R.EVALUATION) for q,p in predictions[episode][method].items()}
            d['primary_metrics']=d['prefix_metrics'][str(d['queries'])]['subject_mean']
            assert d['primary_metrics']['novel_macro_f1']<=d['recovered']/10+1e-12
            rows.append(dict(episode=episode,method=method,queries=d['queries'],recovered=d['recovered'],**d['primary_metrics']))
        for cutoff in ['O','H']:
            a=details[episode][cutoff+'_all'];g=details[episode][cutoff+'_agreed'];q=min(a['queries'],g['queries'])
            contrasts.append(dict(episode=episode,cutoff=cutoff,full_cap_delta={k:g['primary_metrics'][k]-a['primary_metrics'][k] for k in a['primary_metrics']},
                per_subject_deltas={str(s):{k:g['prefix_metrics'][str(g['queries'])]['per_subject'][str(s)][k]-a['prefix_metrics'][str(a['queries'])]['per_subject'][str(s)][k] for k in a['primary_metrics']} for s in R.EVALUATION},
                matched_q=q,matched_q_novel_delta=g['prefix_metrics'][str(q)]['subject_mean']['novel_macro_f1']-a['prefix_metrics'][str(q)]['subject_mean']['novel_macro_f1']))
    save(out/'DETAILS.json',details);save(out/'RESULTS.json',rows);save(out/'CONTRASTS.json',contrasts)
    random_summary={}
    for episode in R.EPISODES:
        random_summary[episode]={k:float(np.mean([details[episode][f'random_{s}']['primary_metrics'][k] for s in contract['random_seeds']])) for k in ['known_macro_f1','novel_macro_f1','combined_macro_f1']}
    save(out/'RANDOM_SUMMARY.json',random_summary)
    analysis={}
    for cutoff in ['O','H']:
        cc=[r for r in contrasts if r['cutoff']==cutoff]
        analysis[cutoff]=dict(per_episode={r['episode']:r['full_cap_delta'] for r in cc},equal_episode_mean={k:float(np.mean([r['full_cap_delta'][k] for r in cc])) for k in cc[0]['full_cap_delta']})
    analysis['interaction']=dict(per_episode={e:next(r['full_cap_delta']['novel_macro_f1'] for r in contrasts if r['episode']==e and r['cutoff']=='H')-next(r['full_cap_delta']['novel_macro_f1'] for r in contrasts if r['episode']==e and r['cutoff']=='O') for e in R.EPISODES},interpretation='predeclared sensitivity; shared evaluation and source')
    save(out/'DESCRIPTIVE_ANALYSIS.json',analysis)
    save(out/'INTERPRETATION.json',dict(episodes=2,disjoint_acquisition_cohorts=True,shared_evaluation_subjects=R.EVALUATION,
       fixed_source_encoder=True,independent_full_studies=False,formal_significance_claim=False,native_learned_baseline_gap_closed=False,
       no_target_training=True,known4=R.KNOWN,novel10=R.NOVEL,all19_raw_activities_retained=True,novel_f1_budget_ceiling=0.4))
    assert sha(out/'QUERY_PLANS.json')==json.loads((out/'PLAN_SEAL.json').read_text())['sha256']
    assert sha(out/'PREDICTIONS.json')==json.loads((out/'PREDICTION_SEAL.json').read_text())['sha256']
    return calls

def run(project,out,execution):
    begin=time.monotonic();signal.signal(signal.SIGALRM,timeout)
    def limit(seconds=120):
        remaining=1800-(time.monotonic()-begin)
        if remaining<=0:raise TimeoutError('R2 overall cap')
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'RSS3GiB cap'
        signal.alarm(max(1,min(seconds,int(remaining))))
    limit(180);torch.set_num_threads(4)
    contract=json.loads(Path(execution['scientific_contract']).read_text());assert B.runtime()==contract['runtime']
    import r2_dsads_core as R
    R.roles_check(contract['target']['episodes'],contract['target']['evaluation_subjects'])
    provenance=json.loads(Path(execution['source_provenance']).read_text())
    for pin in contract['source_inputs']:
        rel=pin['path']
        if rel in provenance:
            src=provenance[rel];assert src['original_sha256']==pin['sha256']
            assert sha(Path(src['embedded_name']))==src['normalized_sha256']
        elif 'expected_metadata' in pin:assert json.loads((project/rel).read_text())==pin['expected_metadata']
        else:assert sha(project/rel)==pin['sha256'],rel
    from d3_matched_agreement import generated as oracle_checks
    import test_c1_aggregate_selector as c1checks
    c1report=c1checks.run_report();assert c1report['passed']
    generated=dict(dsads=R.generated(),data_and_metrics=K.generated(),matched=M.generated(),typiclust=TC.generated(True),oracle=oracle_checks(),aggregate=c1report)
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
    print('R2_SOURCE_BUNDLE_SEALED; starting first official DSADS fetch',flush=True)
    raw=official_archive(out,contract['target']['official_metadata_url'],limit)
    x,truth,wids,subjects=curate_dsads(raw,out,limit)
    limit(900);normalized=K.normalize(x,stats);assert np.isfinite(normalized).all()
    target=P.encode(model,normalized,batch=512);del model,normalized,x
    assert target.shape==(len(wids),64) and np.isfinite(target).all() and np.all(np.linalg.norm(target,axis=1)>0)
    np.savez_compressed(out/'DSADS_EMBEDDINGS.npz',window_ids=wids,embeddings=target)
    save(out/'EMBEDDING_SEAL.json',dict(sha256=sha(out/'DSADS_EMBEDDINGS.npz'),windows_sha256=sha(out/'DSADS_WINDOWS.npz'),source_transfer_sha256=sha(out/'SOURCE_TRANSFER_BUNDLE.npz'),runtime=B.runtime(),optimizer_steps=0))
    calls=compare_episodes(target,truth,wids,subjects,calproto,cal,kp,contract,out,limit,P)
    signal.alarm(0);save(out/'R2_RESULT.json',dict(status='complete',conditions=54,episodes=2,simulated_oracle_calls=calls,optimizer_steps=0,native_baseline_gap_closed=False,elapsed_seconds=time.monotonic()-begin))
    print('FINISHED R2 DSADS: preserve all results for audit',flush=True)

if __name__=='__main__':
    import sys,traceback
    project,out=map(Path,sys.argv[1:3]);execution=json.loads(Path(sys.argv[3]).read_text())
    try:run(project,out,execution)
    except Exception as e:
        signal.alarm(0);save(out/'R2_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
