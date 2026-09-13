"""Build a continuous, metre-scaled macro forest floor. No external scenery required."""
import math
import random
import sys
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from heaven.blender.motion import terrain_height
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/fly-heaven/continuous'
RNG=random.Random(173)

def material(name,color,rough=.7,noise=0):
    m=bpy.data.materials.new(name); m.use_nodes=True
    n=m.node_tree.nodes; l=m.node_tree.links; p=n.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough
    if noise:
        tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=noise;tex.inputs['Detail'].default_value=4
        coord=n.new('ShaderNodeTexCoord');l.new(coord.outputs['Object'],tex.inputs['Vector'])
        ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(*(c*.45 for c in color),1);ramp.color_ramp.elements[1].color=(*(min(1,c*1.6) for c in color),1)
        l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],p.inputs['Base Color'])
        bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.45;bump.inputs['Distance'].default_value=.00018
        l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],p.inputs['Normal'])
    return m

def mesh(name,verts,faces,mat):
    d=bpy.data.meshes.new(name);d.from_pydata(verts,[],faces);d.update()
    o=bpy.data.objects.new(name,d);bpy.context.scene.collection.objects.link(o);o.data.materials.append(mat)
    for p in d.polygons:p.use_smooth=True
    return o

def ellipsoid(name,loc,scale,mat,detail=2):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=detail,radius=1,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(mat)
    for p in o.data.polygons:p.use_smooth=True
    return o

def leaf(name,loc,length,width,mat,angle=0):
    verts=[];faces=[]
    for i in range(13):
        u=i/12;w=width*math.sin(math.pi*u)**.8
        for side in (-1,0,1):verts.append((side*w,length*(u-.5),.12*length*math.sin(math.pi*u)+abs(side)*length*.035*math.sin(9*u)))
    for i in range(12):
        for j in range(2):a=i*3+j;faces.append((a,a+1,a+4,a+3))
    o=mesh(name,verts,faces,mat);o.location=loc;o.rotation_euler.z=angle
    sol=o.modifiers.new('Thin leaf edge','SOLIDIFY');sol.thickness=.00008
    return o

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    soil=material('Damp umber soil',(.095,.054,.027),.92,1100)
    moss=material('Velvet moss',(.085,.14,.025),.9,800)
    bark=material('Weathered bark',(.11,.065,.033),.9,500)
    stone=material('Warm mineral',(.24,.21,.15),.8,900)
    greens=[material('Leaf green '+str(i),c,.62,180) for i,c in enumerate([(.06,.12,.018),(.11,.19,.028),(.16,.22,.045)])]
    dry=[material('Leaf litter '+str(i),c,.85,320) for i,c in enumerate([(.16,.055,.014),(.24,.13,.034),(.085,.035,.012)])]
    # Continuous surface extends well beyond the camera's travel envelope.
    N=180;verts=[(-1.25+2.5*i/N,-1.05+2.1*j/N,terrain_height(-1.25+2.5*i/N,-1.05+2.1*j/N)) for i in range(N+1) for j in range(N+1)]
    faces=[(i*(N+1)+j,i*(N+1)+j+1,(i+1)*(N+1)+j+1,(i+1)*(N+1)+j) for i in range(N) for j in range(N)]
    mesh('ForestFloor',verts,faces,soil)
    import numpy as np
    tl=np.load(ROOT/'runs/fly-heaven/final4/timeline.npz',allow_pickle=True)
    path=np.column_stack([tl['fly_x'][::6]-.6,tl['fly_y'][::6]-.4])
    female=tl['female_present'][::6].astype(bool)
    path=np.concatenate([path,np.column_stack([tl['female_x'][::6][female]-.6,tl['female_y'][::6][female]-.4])])
    def clear(x,y,r=.026):return np.min(np.sum((path-[x,y])**2,axis=1))>r*r
    # Instanced stones give close-up scale cues without per-frame geometry rebuilds.
    templates=[ellipsoid('Pebble master '+str(i),(0,0,-2),(1,1,1),stone,1) for i in range(4)]
    for i in range(1800):
        x=RNG.uniform(-.85,.85);y=RNG.uniform(-.65,.65);r=RNG.uniform(.00035,.0025)
        o=bpy.data.objects.new('Soil grain',templates[i%4].data);bpy.context.scene.collection.objects.link(o)
        o.location=(x,y,terrain_height(x,y)-r*.35);o.scale=(r,r*RNG.uniform(.6,1.2),r*.5);o.rotation_euler=(RNG.random(),RNG.random(),RNG.random()*6)
    for i in range(310):
        x=RNG.uniform(-.85,.85);y=RNG.uniform(-.65,.65)
        if not clear(x,y):continue
        z=terrain_height(x,y)
        if i%3:
            leaf('Curled litter',(x,y,z+.0001),RNG.uniform(.016,.045),RNG.uniform(.004,.012),RNG.choice(dry),RNG.random()*6.28)
        else:
            ellipsoid('Moss cushion',(x,y,z-.0005),(RNG.uniform(.009,.023),RNG.uniform(.008,.018),.003),moss)
            for j in range(7):
                a=RNG.random()*6.28;rad=RNG.random()*.007
                leaf('Moss frond',(x+rad*math.cos(a),y+rad*math.sin(a),z),.007,.0012,RNG.choice(greens),a)
    for i in range(65):
        x=RNG.uniform(-.9,.9);y=RNG.uniform(-.7,.7)
        if not clear(x,y,.05):continue
        z=terrain_height(x,y)
        for j in range(5):
            o=leaf('Understory',(x,y,z),RNG.uniform(.045,.1),.011,RNG.choice(greens),RNG.random()*6.28)
            o.rotation_euler.x=RNG.uniform(.3,1.15)
    for i in range(18):
        x=RNG.uniform(-.8,.8);y=RNG.uniform(-.6,.6)
        if clear(x,y,.05):
            o=ellipsoid('Fallen twig',(x,y,terrain_height(x,y)+.001),(.002,.05,.0018),bark)
            o.rotation_euler.z=RNG.random()*6.28
    # Low, ruptured fruit: feeding radius is 5 cm, so edible flesh reaches the fly.
    pulp=material('Exposed ripe flesh',(.36,.075,.022),.3,620)
    p=pulp.node_tree.nodes.get('Principled BSDF');p.inputs['Subsurface Weight'].default_value=.12;p.inputs['Subsurface Radius'].default_value=(.001,.0004,.0002)
    skin=material('Oxidized fruit skin',(.20,.017,.009),.48,430)
    fx,fy=-.18,.11
    ellipsoid('Split fallen fruit',(fx,fy,terrain_height(fx,fy)-.005),(.052,.046,.006),pulp,4)
    for i in range(13):
        a=i*2*math.pi/13;x=fx+.047*math.cos(a);y=fy+.041*math.sin(a)
        ellipsoid('Torn peel',(x,y,terrain_height(x,y)-.0015),(.009,.005,.002),skin)
    ellipsoid('Sun warmed resting stone',(-.30,.15,terrain_height(-.30,.15)-.003),(.032,.025,.0035),stone,3)
    leaf('FallingLeaf',(.15,-.20,.5),.055,.016,dry[1],.4)
    world=bpy.data.worlds.new('Soft forest sky');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(.55,.68,.85,1);world.node_tree.nodes['Background'].inputs[1].default_value=.22
    bpy.context.scene.world=world
    bpy.ops.object.light_add(type='SUN');sun=bpy.context.object;sun.name='Filtered afternoon sun';sun.rotation_euler=(.45,-.65,-.5);sun.data.energy=2.0;sun.data.angle=.12;sun.data.color=(1,.83,.60)
    bpy.ops.object.light_add(type='AREA',location=(0,-.3,.7));fill=bpy.context.object;fill.data.energy=12;fill.data.shape='DISK';fill.data.size=1.2;fill.data.color=(.65,.8,1)
    # Overhead leaves cast dappled shadows while staying above the camera.
    for i in range(50):
        x=RNG.uniform(-1,1);y=RNG.uniform(-.8,.8)
        leaf('Canopy',(x,y,RNG.uniform(.5,.8)),.18,.08,greens[0],RNG.random()*6.28)
    s=bpy.context.scene;s.render.fps=24;s.unit_settings.system='METRIC';s.view_settings.view_transform='AgX'
    OUT.mkdir(parents=True,exist_ok=True);bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'world.blend'))
    print('WORLD_BUILD_COMPLETE',len(s.objects),flush=True)
if __name__=='__main__':main()
