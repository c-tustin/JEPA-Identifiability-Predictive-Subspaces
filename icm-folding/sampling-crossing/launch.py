"""Run bounded one-thread workers, maintain progress, and audit on completion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    assert 1<=a.workers<=8
    OUT.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    handles=[];children=[]
    for k in range(a.workers):
        log=(OUT/f'worker-{k}.log').open('ab',buffering=0);handles.append(log)
        children.append(subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),
            '--shard',str(k),'--shards',str(a.workers)],env=env,stdout=log,stderr=subprocess.STDOUT))
    (OUT/'workers.json').write_text(json.dumps(dict(pids=[c.pid for c in children],workers=a.workers),indent=2)+'\n')
    while True:
        codes=[c.poll() for c in children]
        if any(code is not None and code!=0 for code in codes):
            for c in children:
                if c.poll() is None:c.terminate()
            raise RuntimeError(f'Worker failed; inspect worker logs: {codes}')
        subprocess.run([sys.executable,str(HERE/'report.py'),'--progress-only'],env=env,check=True)
        if all(code==0 for code in codes):break
        time.sleep(30)
    subprocess.run([sys.executable,str(HERE/'report.py')],env=env,check=True)
    for log in handles:log.close()

if __name__=='__main__':main()
