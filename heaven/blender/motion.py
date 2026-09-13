"""Deterministic presentation sampling, independent of Blender and brain decisions."""
import math
import numpy as np

FPS = 24
OFFSET = np.array([-0.6, -0.4])
CONTINUOUS = {'fly_x', 'fly_y', 'fly_altitude', 'leaf_x', 'leaf_y', 'leaf_height', 'motor_hz', 'proboscis'}

def ease(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3 - 2 * x)

def angle_delta(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def terrain_height(x, y):
    # Same shallow mesh surface used by world builder and foot targets.
    return .0007 * math.sin(17*x) * math.sin(19*y) + .00035 * math.sin(41*x + 13*y)

class Timeline:
    def __init__(self, path):
        with np.load(path, allow_pickle=True) as data:
            self.columns = {k: data[k] for k in data.files}
        self.t = self.columns['t']
        if len(self.t) < 2 or not np.all(np.diff(self.t) > 0):
            raise ValueError('Timeline must contain increasing timestamps')
    def max_t(self):
        return float(self.t[-1])
    def row(self, i):
        return {k: v[i] for k,v in self.columns.items()}
    def sample(self, t):
        i = max(0, min(len(self.t)-1, int(np.searchsorted(self.t,t,side='right')-1)))
        j = min(i+1,len(self.t)-1)
        row = self.row(i)
        a = np.clip((t-self.t[i]) / max(self.t[j]-self.t[i],1e-9),0,1)
        for k in CONTINUOUS & self.columns.keys():
            row[k] = (1-a)*self.columns[k][i]+a*self.columns[k][j]
        row['fly_heading'] = float(self.columns['fly_heading'][i] + a*angle_delta(self.columns['fly_heading'][i],self.columns['fly_heading'][j]))
        if row['female_present'] and self.columns['female_present'][j]:
            for k in ('female_x','female_y'):
                row[k] = (1-a)*self.columns[k][i]+a*self.columns[k][j]
        return row

class Blend:
    """Interrupted transitions restart from the displayed value, never an old pose."""
    def __init__(self):
        self.key=None; self.start=0.; self.source=None; self.value=None
    def update(self,key,target,t,duration=.3):
        target=np.asarray(target,dtype=float)
        if self.value is None:
            self.value=target.copy()
        if key != self.key:
            self.key=key; self.start=t; self.source=self.value.copy()
        a=ease((t-self.start)/max(duration,1e-6))
        self.value=(1-a)*self.source+a*target
        return self.value.copy()

class Travel:
    def __init__(self):
        self.previous=None; self.distance=0.; self.speed=0.; self.heading=None
    def update(self,xy,heading,dt=1/FPS):
        xy=np.asarray(xy)
        d=0 if self.previous is None else float(np.linalg.norm(xy-self.previous))
        self.distance+=d
        self.speed += (d/dt-self.speed)*(1-math.exp(-dt/.10))
        self.previous=xy.copy()
        if self.heading is None: self.heading=heading
        if d>1e-7: self.heading+=angle_delta(self.heading,heading)*(1-math.exp(-dt/.16))
        return self.distance/.006, self.speed, self.heading
