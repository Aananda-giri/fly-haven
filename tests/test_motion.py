import math
import numpy as np
from heaven.blender.motion import Timeline,Blend,Travel,angle_delta

def test_sampler_preserves_event_onset_and_interpolates(tmp_path):
    p=tmp_path/'t.npz'
    np.savez(p,t=[0.,1.,2.],fly_x=[0.,2.,4.],fly_y=[0.,0.,0.],fly_heading=np.deg2rad([179,-179,-177]),pose=['walk','feed','flight'],female_present=[False,True,True],female_x=[-1.,3.,4.],female_y=[-1.,0.,0.])
    tl=Timeline(p)
    assert tl.sample(.5)['pose']=='walk'
    assert tl.sample(1)['pose']=='feed'
    assert tl.sample(.5)['fly_x']==1
    assert abs(abs(tl.sample(.5)['fly_heading'])-math.pi)<1e-6
    assert tl.sample(.5)['female_x']==-1
    assert tl.sample(10)['fly_x']==4

def test_interrupted_blend_is_continuous():
    b=Blend();b.update('walk',[0],0)
    b.update('feed',[1],1)
    shown=b.update('feed',[1],1.15)
    assert 0<shown[0]<1
    assert np.array_equal(b.update('flight',[-1],1.15),shown)
    assert np.allclose(b.update('flight',[-1],1.6),[-1])

def test_stationary_gait_does_not_advance():
    t=Travel();t.update([0,0],0);phase,_,_=t.update([.006,0],0)
    for _ in range(30):p,s,h=t.update([.006,0],math.pi)
    assert p==phase==1
    assert s<.00001
    assert h==0

def test_heading_uses_shortest_arc():
    assert math.isclose(angle_delta(math.radians(179),math.radians(-179)),math.radians(2))

def test_continuous_edit_covers_each_frame_once(tmp_path):
    import json
    import pandas as pd
    from heaven.director import build
    p=tmp_path/'timeline.parquet';out=tmp_path/'edl.json'
    pd.DataFrame({'t':[.05,1.,2.]}).to_parquet(p)
    result=build(p,out)
    assert result['mode']=='continuous'
    assert result['ranges']==[[0,48,1]]
    assert result['total_output_frames']==49
    assert json.loads(out.read_text())['fps']==24
