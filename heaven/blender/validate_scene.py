"""Inspect baked rig contacts and camera framing without changing the scene."""
import bpy,json,sys,math
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from heaven.blender.motion import terrain_height
scene=bpy.context.scene
report={'frames':scene.frame_end+1,'fps':scene.render.fps,'characters':{},'camera_outside':[],'contact_samples':[]}
for name in ('Male','Female'):
    root=bpy.data.objects.get('Fly_'+name)
    if root:report['characters'][name]={'scale':list(root.scale),'bones':len(bpy.data.objects['Rig_'+name].pose.bones)}
for frame in range(scene.frame_start,scene.frame_end+1,12):
    scene.frame_set(frame)
    root=bpy.data.objects['Fly_Male'];p=root.matrix_world.translation+Vector((0,0,.004))
    projected=world_to_camera_view(scene,scene.camera,p)
    if not (.08<projected.x<.92 and .08<projected.y<.92 and projected.z>0):report['camera_outside'].append({'frame':frame,'ndc':list(projected)})
for frame in (0,240,540,2100,3600,3750,4200,5280):
    if frame>scene.frame_end:continue
    scene.frame_set(frame);rig=bpy.data.objects['Rig_Male']
    for part in ('Front','Center','Rear'):
        for side in ('L','R'):
            pb=rig.pose.bones[f'Leg.{part}.{side}.004'];target=bpy.data.objects[f'Foot_{part}_{side}_Male']
            error=(rig.matrix_world@pb.tail-target.location).length
            toe=rig.matrix_world@rig.pose.bones[f'Leg.{part}.{side}.008'].tail
            report['contact_samples'].append({'frame':frame,'leg':f'{part}.{side}','ik_error_m':error,'toe_height_m':toe.z-terrain_height(toe.x,toe.y)})
print('VALIDATION',json.dumps(report),flush=True)
if '--' in sys.argv:Path(sys.argv[sys.argv.index('--')+1]).write_text(json.dumps(report,indent=2))
