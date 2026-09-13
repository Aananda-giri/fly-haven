"""Bake the BlenderKit flies and one continuous macro camera from a neural timeline.

blender --factory-startup -b world.blend --python heaven/blender/animate.py -- timeline.npz scene.blend [end_seconds]
"""
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Euler
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from heaven.blender.motion import Timeline, Blend, Travel, FPS, OFFSET, ease, angle_delta, terrain_height, surface_height
ROOT=Path(__file__).resolve().parents[2]
RIG_BLEND=ROOT/'assets/blenderkit-housefly/source.blend'
PARTS=('Front','Center','Rear')
SOURCE_MESHES={}
CONTROLS=['Haed','Torax.front.001','Wing.L','Wing.R','dabbing snout up','dabbing snout down']

def append_fly(suffix,scale=1.):
    with bpy.data.libraries.load(str(RIG_BLEND),link=False) as (src,dst):
        names=list(src.objects);dst.objects=names
    objects=[o for o in dst.objects if o]
    root=next(o for o in objects if o.type=='EMPTY' and o.parent is None)
    rig=next(o for o in objects if o.type=='ARMATURE')
    for source_name,o in zip(names,objects):
        if o.type=='MESH':
            if source_name in SOURCE_MESHES:o.data=SOURCE_MESHES[source_name]
            else:SOURCE_MESHES[source_name]=o.data
        bpy.context.scene.collection.objects.link(o)
        o.name=o.name+'_'+suffix;o.animation_data_clear()
    root.name='Fly_'+suffix;rig.name='Rig_'+suffix;root.scale=(scale,)*3
    for b in rig.pose.bones:
        b.rotation_mode='QUATERNION';b.rotation_quaternion=(1,0,0,0)
    for name in CONTROLS:
        if name not in rig.pose.bones:raise ValueError('Missing BlenderKit control: '+name)
    bpy.context.view_layer.update()
    feet=[]
    for part in PARTS:
        for side in ('L','R'):
            name=f'Leg.{part}.{side}.004';pb=rig.pose.bones[name]
            rest=rig.matrix_world @ pb.tail
            target=bpy.data.objects.new(f'Foot_{part}_{side}_{suffix}',None);bpy.context.scene.collection.objects.link(target)
            target.empty_display_size=.001;target.location=rest
            ik=pb.constraints.new('IK');ik.name='Ground contact';ik.target=target;ik.chain_count=3;ik.use_stretch=False;ik.iterations=64
            # The end of .004 is above the final claw by this authored offset.
            toe=rig.matrix_world @ rig.pose.bones[f'Leg.{part}.{side}.008'].tail
            feet.append(dict(part=part,side=side,target=target,rest=root.matrix_world.inverted()@rest,clearance=max(.0002,rest.z-toe.z+.00015),plant=None,swing=False,from_pos=None))
    return root,rig,objects,feet

def poses(pose,t,speed):
    v=np.zeros(22) # six Euler controls, body pitch/roll/height, contact mode
    a=v[:18].reshape(6,3)
    a[0,2]=1.5*math.sin(t*2.1);a[1,0]=.7*math.sin(t*3.2)
    if pose=='feed':
        a[0,0]=12+2*math.sin(t*8);a[4,0]=-24;a[5,0]=24+5*math.sin(t*9)
    elif pose=='groom':a[0,0]=5*math.sin(t*10);v[21]=1
    elif pose=='flight':
        a[2]=(0,65*math.sin(t*2*math.pi*8.3),-45)
        a[3]=(0,-65*math.sin(t*2*math.pi*8.3),45)
        v[18]=-9;v[21]=2
    elif pose=='sing':
        a[3]=(0,8*math.sin(t*2*math.pi*7.3),62)
        a[0,0]=-3
    elif pose=='tap':v[21]=3
    elif pose in ('mate','mount_approach'):
        a[1,0]=-14;a[2,2]=-12;a[3,2]=12;v[18]=12;v[20]=.0055;v[21]=4
    elif pose=='bask_belly_up':
        # Roll about the body's forward axis with clearance for the thorax.
        v[19]=170;v[20]=.007;v[21]=2
    elif pose=='bask':a[2,2]=-8;a[3,2]=8
    return v

def positions(tl,end):
    rows=[];male=[];female=[];mh=[];fh=[]
    mtravel=Travel();ftravel=Travel();moffset=np.zeros(3);foffset=np.zeros(2)
    last_fh=0.;first_f=None
    for frame in range(round(end*FPS)+1):
        t=frame/FPS;r=tl.sample(t);rows.append(r)
        xy=np.array([r['fly_x'],r['fly_y']])+OFFSET
        _,_,h=mtravel.update(xy,float(r['fly_heading']));mh.append(h)
        fxy=np.array([r['female_x'],r['female_y']])+OFFSET
        present=bool(r['female_present'])
        if present:
            if first_f is None:first_f=t
            # Entrance comes from the nearby scene edge, settles before courtship.
            fxy=fxy+np.array([.14,.06])*(1-ease((t-first_f)/3))
            dist=float(np.linalg.norm(fxy-xy))
            court=r['state']=='COURT'
            desired=np.array([math.cos(h),math.sin(h)])*max(0,.019-dist) if dist<.019 else np.zeros(2)
            if r['pose'] in ('mate','mount_approach'):desired*=.4
            foffset+=(desired-foffset)*(1-math.exp(-1/FPS/.35))
            fxy+=foffset
            _,fs,last_fh=ftravel.update(fxy,last_fh if ftravel.previous is None else math.atan2(fxy[1]-ftravel.previous[1],fxy[0]-ftravel.previous[0]))
            if court:
                last_fh+=angle_delta(last_fh,h)*(1-math.exp(-1/FPS/.3))
                ftravel.heading=last_fh
        fh.append(last_fh)
        female.append([*fxy,surface_height(*fxy)])
        male.append([*xy,surface_height(*xy)+float(r['fly_altitude'])])
    return rows,np.array(male),np.array(female),np.array(mh),np.array(fh)

def bake_fly(root,rig,objects,feet,rows,path,headings,is_female=False):
    blend=Blend();travel=Travel();last_visible=None
    for frame,(r,xyz,h) in enumerate(zip(rows,path,headings)):
        t=frame/FPS
        if frame%1200==0:print('BAKING',root.name,frame,flush=True)
        visible=not is_female or bool(r['female_present'])
        if visible!=last_visible:
            for o in objects:
                if o.type=='MESH':
                    o.hide_render=not visible;o.keyframe_insert('hide_render',frame=frame)
            last_visible=visible
        if not visible:continue
        phase,speed,_=travel.update(xyz[:2],h)
        pose='flight' if float(r['fly_altitude'])>.001 else str(r['pose'])
        if is_female:pose='bask' if r['state']=='COURT' else 'walk'
        value=blend.update(pose,poses(pose,t,speed),t,.12 if pose=='flight' else .35)
        root.location=tuple(xyz+np.array([0,0,value[20]+.00015]))
        root.rotation_mode='QUATERNION'
        root.rotation_quaternion=Euler((math.radians(value[18]),math.radians(value[19]),h+math.pi/2),'XYZ').to_quaternion()
        root.keyframe_insert('location',frame=frame);root.keyframe_insert('rotation_quaternion',frame=frame)
        for name,angles in zip(CONTROLS,value[:18].reshape(6,3)):
            pb=rig.pose.bones[name];pb.rotation_quaternion=Euler(tuple(math.radians(x) for x in angles)).to_quaternion();pb.keyframe_insert('rotation_quaternion',frame=frame)
        forward=Vector((math.cos(h),math.sin(h),0))
        for foot in feet:
            rest=root.matrix_basis @ foot['rest'];rest.z=surface_height(rest.x,rest.y)+foot['clearance']
            group=(foot['part'],foot['side']) in {('Front','L'),('Center','R'),('Rear','L')}
            p=(phase+(0 if group else .5))%1
            walking=speed>.0002 and pose in ('walk','follow','retreat','mount_approach','dismount')
            swing=walking and p>.60
            if foot['plant'] is None:foot['plant']=rest.copy()
            if swing:
                if not foot['swing']:foot['from_pos']=foot['plant'].copy()
                target=rest+forward*.002
                u=(p-.60)/.40;loc=foot['from_pos'].lerp(target,float(ease(u)));loc.z+=.0014*math.sin(math.pi*u)
                foot['plant']=loc.copy()
            else:
                loc=foot['plant'].copy()
                # Replant after very sharp turns; bounded reach prevents stretched legs.
                if (loc-rest).length>.004:loc=rest.copy();foot['plant']=loc.copy()
            foot['swing']=swing
            mode=value[21]
            special=rest.copy()
            if pose=='groom' and foot['part']=='Front':
                special=root.matrix_basis @ Vector((.0015*(1 if foot['side']=='L' else -1),-.0065,.004))
                special+=forward*(.0005*math.sin(t*22+(0 if foot['side']=='L' else math.pi)))
            elif pose=='tap' and foot['part']=='Front':special+=forward*.002;special.z+=.0008*(.5+.5*math.sin(t*15))
            elif pose in ('flight','bask_belly_up'):
                special=root.matrix_basis @ foot['rest'];special.z+=.002
            elif pose in ('mate','mount_approach'):
                special=root.matrix_basis @ foot['rest'];special.z+=.001 if foot['part']=='Front' else 0
            else:special=loc.copy()
            # Blend special foot motions from last displayed target at state boundaries.
            if 'blend' not in foot:foot['blend']=Blend()
            loc=Vector(foot['blend'].update(pose,np.array(special),t,.12 if pose=='flight' else .3))
            foot['target'].location=loc;foot['target'].keyframe_insert('location',frame=frame)
    print('BAKED',root.name,flush=True)

def camera(rows,path,female,headings):
    scene=bpy.context.scene
    data=bpy.data.cameras.new('Macro lens');cam=bpy.data.objects.new('ContinuousCamera',data);scene.collection.objects.link(cam);scene.camera=cam
    data.lens=48;data.clip_start=.0005;data.clip_end=30;data.dof.use_dof=True;data.dof.aperture_fstop=18
    focus=bpy.data.objects.new('Focus',None);scene.collection.objects.link(focus);data.dof.focus_object=focus
    pos=target=None;camera_heading=float(headings[0]);violations=[]
    for frame,(r,xyz,h) in enumerate(zip(rows,path,headings)):
        camera_heading+=angle_delta(camera_heading,h)*(1-math.exp(-1/FPS/1.2))
        f=np.array([math.cos(camera_heading),math.sin(camera_heading),0]);side=np.array([-f[1],f[0],0])
        center=xyz+np.array([math.cos(h)*.002,math.sin(h)*.002,.0055]);court=r['state']=='COURT' and r['female_present']
        if court:center=(center+female[frame]+np.array([0,0,.005]))*.5
        airborne=float(r['fly_altitude'])>.001 or r['pose']=='flight'
        desired=center+f*(.085 if airborne else (.042 if court else .036))+side*(.09 if airborne else .065)+np.array([0,0,.05 if airborne else .027])
        if pos is None:pos=desired.copy();target=center.copy()
        pos+=(desired-pos)*(1-math.exp(-1/FPS/(.055 if airborne else .3)));target+=(center-target)*(1-math.exp(-1/FPS/(.025 if airborne else .16)))
        pos[2]=max(pos[2],terrain_height(pos[0],pos[1])+.018)
        cam.location=pos;cam.rotation_mode='QUATERNION';cam.rotation_quaternion=(Vector(target)-cam.location).to_track_quat('-Z','Y')
        focus.location=target
        cam.keyframe_insert('location',frame=frame);cam.keyframe_insert('rotation_quaternion',frame=frame);focus.keyframe_insert('location',frame=frame)
    return cam

def linear_keys():
    # Blender 5 actions use channel bags. Dense samples must not overshoot.
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for key in curve.keyframe_points:key.interpolation='CONSTANT' if curve.data_path=='hide_render' else 'LINEAR'

def main():
    argv=sys.argv[sys.argv.index('--')+1:];tl=Timeline(argv[0]);end=min(float(argv[2]),tl.max_t()) if len(argv)>2 else tl.max_t()
    rows,male,female,mh,fh=positions(tl,end)
    root,rig,objects,feet=append_fly('Male');bake_fly(root,rig,objects,feet,rows,male,mh)
    if any(r['female_present'] for r in rows):
        root,rig,objects,feet=append_fly('Female',1.08);bake_fly(root,rig,objects,feet,rows,female,fh,True)
    camera(rows,male,female,mh)
    leaf=bpy.data.objects.get('FallingLeaf')
    if leaf:
        for frame,r in enumerate(rows):
            leaf.location=(r['leaf_x']-.6,r['leaf_y']-.4,max(.001,r['leaf_height']))
            t=frame/FPS;fall=max(0,t-85);leaf.rotation_euler=(.2*math.sin(fall*8),.3*math.sin(fall*5),.4+min(fall,2.2)*1.8)
            leaf.keyframe_insert('location',frame=frame);leaf.keyframe_insert('rotation_euler',frame=frame)
    scene=bpy.context.scene;scene.render.fps=FPS;scene.frame_start=0;scene.frame_end=len(rows)-1
    linear_keys();scene.frame_set(0)
    Path(argv[1]).parent.mkdir(parents=True,exist_ok=True);bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(Path(argv[1]).resolve()))
    print('ANIMATE_COMPLETE',scene.frame_end,flush=True)
if __name__=='__main__':main()
