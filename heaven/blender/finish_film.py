"""Run a resumable Cycles export and assemble only a complete continuous film.

uv run python -m heaven.blender.finish_film [--out runs/fly-heaven/continuous]
Progress: OUT/render.log; lifecycle: OUT/job.json. Safe to restart after failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def finish(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'job.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scene=out/'scene.blend';frames=out/'frames';movie=out/'fly-heaven.mp4'
    env=os.environ.copy();env['PYTHONHOME']='/usr';env['PATH']='/usr/bin:/bin'
    state={'pid':os.getpid(),'started':time.time(),'status':'rendering','movie':str(movie)}
    def save(): (out/'job.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        with (out/'render.log').open('a') as log:
            subprocess.run(['/usr/bin/blender','--factory-startup','-b',str(scene),'--python-exit-code','1','--python',str(ROOT/'heaven/blender/render.py'),'--','continuous',str(frames),'1920','1080','64','CYCLES'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        state['status']='encoding';save()
        with (out/'encode.log').open('a') as log:
            subprocess.run([sys.executable,'-m','heaven.overlay',str(out/'edl.json'),str(frames),str(movie)],check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(movie)]))['streams'][0]
        expected=json.loads((out/'edl.json').read_text())['total_output_frames']
        if int(info['nb_frames'])!=expected or (info['width'],info['height'])!=(1920,1080) or info['avg_frame_rate']!='24/1':raise RuntimeError(f'Unexpected encoded stream: {info}')
        state.update(status='complete',finished=time.time(),stream=info);save()
    except BaseException as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'runs/fly-heaven/continuous'));a=p.parse_args();finish(a.out)
