"""R2 DSADS segment adapter. NumPy only; no network, model, or label acquisition."""
from numbers import Integral
import numpy as np

MAPPING={1:'sitting',2:'standing',3:'lying',4:'lying',5:'stairs_up',6:'stairs_down',
7:'standing',8:'moving_in_elevator',9:'walking',10:'walking',11:'walking',12:'running',
13:'stepper',14:'cross_trainer',15:'cycling',16:'cycling',17:'rowing',18:'jumping',19:'basketball'}
KNOWN=['walking','running','sitting','standing']
NOVEL=['lying','stairs_up','stairs_down','moving_in_elevator','stepper','cross_trainer','cycling','rowing','jumping','basketball']
EPISODES={'A':[1,2],'B':[3,4]}
EVALUATION=[5,6,7,8]
STARTS=(0,64)

def bounded_int(value,lo,hi):
    if isinstance(value,bool) or not isinstance(value,Integral) or not lo<=value<=hi:
        raise ValueError('Invalid subject/activity/segment index')
    return int(value)

def segment_windows(raw,subject,activity,segment):
    """Right-arm acceleration, one-based cols10..12. Never joins segments."""
    subject=bounded_int(subject,1,8);activity=bounded_int(activity,1,19);segment=bounded_int(segment,1,60)
    raw=np.asarray(raw,dtype=np.float64)
    if raw.shape!=(125,45) or not np.isfinite(raw).all():raise ValueError('Expected finite125x45 segment')
    # j/2 is measured in original25Hz sample intervals: no extrapolation.
    grid=np.arange(249,dtype=np.float64)/2
    xyz=np.column_stack([np.interp(grid,np.arange(125),raw[:,c]) for c in (9,10,11)]).astype(np.float32)
    mag=np.sqrt(np.sum(xyz*xyz,axis=1,dtype=np.float32)).astype(np.float32)
    signal=np.column_stack([xyz,mag])
    if not np.isfinite(signal).all():raise ValueError('Nonfinite float32 signal')
    x=np.stack([signal[s:s+128].T.copy() for s in STARTS])
    # Curator IDs retain activity ordinals for audit and must not reach selectors.
    # Segment ordinal covers 19*60 files without exposing their activity grouping.
    ordinal=(activity-1)*60+(segment-1)
    ids=np.asarray([subject*1000000000+ordinal*1000+s for s in STARTS],np.int64)
    return x,ids

def roles_check(episodes,evaluation):
    groups=[list(v) for v in episodes.values()]+[list(evaluation)]
    flat=[s for g in groups for s in g]
    if len(groups)!=3 or any(len(g)==0 for g in groups) or sorted(flat)!=list(range(1,9)):
        raise ValueError('Expected disjoint acquisition cohorts and evaluation covering8subjects')

def opaque_ids(raw_ids):
    """A fixed bijection hides activity-coded segment ordinals from selectors."""
    ids=np.asarray(raw_ids,dtype=np.int64)
    if ids.ndim!=1 or len(ids)!=len(set(ids.tolist())):raise ValueError('Duplicate ID')
    # Stable source order must be canonical before this call; fixed random permutation
    # destroys the direct activity-block coding while retaining a reversible curator map.
    permutation=np.random.default_rng(2026111701).permutation(len(ids))
    return permutation.astype(np.int64)+1

def generated():
    raw=np.zeros((125,45),np.float64);raw[:,9:12]=[3,4,12]
    x,ids=segment_windows(raw,2,19,60)
    assert x.shape==(2,4,128) and x.dtype==np.float32 and np.all(x[:,3]==13)
    assert ids.tolist()==[2001139000,2001139064]
    # Independent time-domain ramp: x(t)=2t+3; sample interval.04s.
    raw[:,9]=3+2*np.arange(125)/25
    x,_=segment_windows(raw,1,1,1)
    for j,start in enumerate(STARTS):
        np.testing.assert_allclose(x[j,0],3+2*(start+np.arange(128))/50,rtol=0,atol=1e-6)
    # Norm after interpolation, not interpolation of endpoint norms.
    raw[:]=0;raw[:,9]=1;raw[:,10]=np.arange(125)
    x,_=segment_windows(raw,1,1,1)
    assert abs(float(x[0,3,1])-np.sqrt(1.25))<1e-6
    for shape in [(124,45),(125,44),(126,45)]:
        try:segment_windows(np.zeros(shape),1,1,1)
        except ValueError:pass
        else:raise AssertionError('shape accepted')
    for args in [(0,1,1),(1,20,1),(1,1,61),(True,1,1),(1.5,1,1)]:
        try:segment_windows(raw,*args)
        except ValueError:pass
        else:raise AssertionError('indices accepted')
    raw[0,0]=np.nan
    try:segment_windows(raw,1,1,1)
    except ValueError:pass
    else:raise AssertionError('nonfinite accepted')
    roles_check(EPISODES,EVALUATION)
    try:roles_check({'A':[1,2],'B':[2,4]},EVALUATION)
    except ValueError:pass
    else:raise AssertionError('overlap accepted')
    assert set(MAPPING)==set(range(1,20)) and set(MAPPING.values())==set(KNOWN+NOVEL)
    assert len(KNOWN)==4 and len(NOVEL)==10 and len(set(KNOWN+NOVEL))==14
    a=opaque_ids(np.arange(18240));assert len(set(a))==18240 and np.array_equal(a,opaque_ids(np.arange(18240)))
    return dict(ramp_time_grid=True,magnitude_after_interpolation=True,shape_and_finite_failures=True,
      subject_isolation=True,all19_raw_classes_mapped=True,canonical_classes=14,novel_denominator=10,
      activity_ordinal_hidden=True,expected_windows=18240,local_research_data_read=False)
