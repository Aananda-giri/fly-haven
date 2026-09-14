"""Compare saved V4/V5/V5.5 benchmarks, preserving scale and assistance metadata."""
import argparse
import json
from pathlib import Path


def collect(roots):
    rows=[]
    for root in roots:
        for path in sorted(Path(root).rglob('summary.json')):
            report=json.loads(path.read_text())
            config=report.get('configuration',{})
            if 'reference_elo' not in config:continue
            branch=next((p for p in path.parts if p in ('v4','v5','v55')),Path(root).name)
            rows.append({'branch':branch,'mode':config['mode'],'reference_elo':config['reference_elo'],
                         'scale':report['reference_scale'],'wins':report['wins'],'draws':report['draws'],
                         'losses':report['losses'],'unresolved':report['unresolved'],'pairs':report['complete_pairs'],
                         'score':report['score'],'conditional_elo':report['reference_scale_estimate'],
                         'conditional_elo_interval':report['reference_scale_interval'],'status':report['status'],
                         'simulations':config['simulations'],'assistance':config['assist_mode'],
                         'checkpoint_sha256':config['checkpoint_sha256'],'result':str(path)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();rows=collect(args.runs)
    print('Version / mode                 SF reference   W/D/L   unresolved  paired score  conditional Elo')
    for row in rows:
        score='unknown' if row['score'] is None else f"{row['score']:.1%}"
        elo='no finite estimate' if row['conditional_elo'] is None else f"{row['conditional_elo']:.0f}"
        print(f"{row['branch']+'/'+row['mode']:<31} {row['reference_elo']:>4}       {row['wins']}/{row['draws']}/{row['losses']}"
              f"       {row['unresolved']:>3}          {score:<8}      {elo}")
    if not rows:print('No completed benchmark reports found. Train and run evaluation first.')
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
