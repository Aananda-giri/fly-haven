"""Render a continuous scene or explicit diagnostic ranges, with safe resume.

blender --factory-startup -b scene.blend --python heaven/blender/render.py -- continuous frames [width height samples engine]
Engine: CYCLES (default) or BLENDER_EEVEE for previews.
"""
import json
import sys
import time
import hashlib
from pathlib import Path
import bpy

def render(ranges,out_dir,width=1920,height=1080,samples=64,engine='CYCLES'):
    scene=bpy.context.scene;scene.render.engine=engine
    scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.compression=15
    scene.view_settings.view_transform='AgX'
    if engine=='CYCLES':
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='OPTIX';prefs.get_devices()
        gpu=False
        for d in prefs.devices:d.use=d.type=='OPTIX';gpu|=d.use
        scene.cycles.device='GPU' if gpu else 'CPU';scene.cycles.samples=samples
        scene.cycles.use_denoising=True;scene.cycles.adaptive_threshold=.045
        scene.cycles.max_bounces=6;scene.cycles.transparent_max_bounces=6
        scene.render.use_persistent_data=True
    else:scene.eevee.taa_render_samples=samples
    ranges=[(r[0],r[1],r[2] if len(r)>2 else 1) for r in ranges]
    frames=[f for a,b,s in ranges for f in range(a,b+1,s)]
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    source=Path(bpy.data.filepath)
    manifest=dict(scene=str(source),scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ranges=ranges,width=width,height=height,samples=samples,engine=engine)
    canonical=json.dumps(manifest,sort_keys=True)
    manifest_path=out/'render.json'
    if manifest_path.exists() and manifest_path.read_text()!=canonical:raise RuntimeError('Render settings or scene changed: use a fresh output directory')
    if not manifest_path.exists() and any(out.glob('frame_*.png')):raise RuntimeError('Unverified old frames: use a fresh directory')
    manifest_path.write_text(canonical)
    started=time.monotonic();count=0
    for i,f in enumerate(frames):
        path=out/f'frame_{i:06d}.png'
        if not path.exists():
            scene.frame_set(f);scene.render.filepath=str(path.with_name(path.stem+'.partial.png'))
            bpy.ops.render.render(write_still=True)
            Path(scene.render.filepath).replace(path);count+=1
        if i%24==0 or i==len(frames)-1:print(f'PROGRESS {i+1}/{len(frames)} elapsed={time.monotonic()-started:.1f}s rendered={count}',flush=True)
    print('RENDER_COMPLETE',len(frames),flush=True)

if __name__=='__main__':
    a=sys.argv[sys.argv.index('--')+1:]
    if a[0]=='continuous':ranges=[[bpy.context.scene.frame_start,bpy.context.scene.frame_end,1]]
    else:
        data=json.loads(Path(a[0]).read_text()) if a[0].endswith('.json') else json.loads(a[0]);ranges=data['ranges'] if isinstance(data,dict) else data
    render(ranges,a[1],int(a[2]) if len(a)>2 else 1920,int(a[3]) if len(a)>3 else 1080,int(a[4]) if len(a)>4 else 64,a[5] if len(a)>5 else 'CYCLES')
