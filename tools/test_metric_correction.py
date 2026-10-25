"""Focused tests for the metric-definition bug, independent of model libraries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

p=Path(__file__).resolve().parents[1]/'notebooks/src/phase5_metrics_corrected.py'
tree=ast.parse(p.read_text())
names={'unit','group_candidates','summarize','spend_budget'}
ns={'np':np,'CFG':SimpleNamespace(min_agreeing_clients=2,label_query_budget=4)}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(p),'exec'),ns)
D=SimpleNamespace(activities=np.array(['A','B']),novel=['A','B'],known=['K'])
rows=[dict(client=1,signature=np.array([1.,0.]),support=10,medoid=0,dominant='A',label_counts={'A':9,'B':1}),
      dict(client=2,signature=np.array([1.,0.]),support=10,medoid=1,dominant='B',label_counts={'A':1,'B':9})]
candidate=ns['group_candidates'](D,rows,[[0,1]])[0]
assert candidate['purity']==.5, candidate
# High individual majority fractions must not manufacture cross-client purity.
assert candidate['label']=='A'
e=[dict(label='A',purity=.2,support=10),dict(label='B',purity=.8,support=30)]
assert abs(ns['summarize'](D,e,4,['K','A','A','B'],{})['entry_purity']-.65)<1e-12
assert ns['summarize'](D,[],0,[],{})['entry_purity']==0
entries,n,labels=ns['spend_budget'](D,[{'label':x} for x in ['K','A','A','B','B']])
assert n==4 and [r['label'] for r in entries]==['A','B']
print('PASS: queried-label agreement, support weighting, empty entries, and query accounting')
