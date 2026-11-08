"""Wait for the fixed OU cohort; then analyze, verify and plot without restarting training."""
import json
import subprocess
import sys
import time
import run_ou as run

if __name__=='__main__':
    run.check_lock()
    while not (run.OUT/'TRAINING_COMPLETE.json').exists():time.sleep(30)
    assert json.loads((run.OUT/'TRAINING_COMPLETE.json').read_text())['signature']==run.signature()
    for name in ['analyze_ou.py','verify_ou.py','plot_ou.py']:
        print('Starting',name,flush=True)
        subprocess.run([sys.executable,str(run.HERE/name)],check=True)
    run.save_json(run.OUT/'PIPELINE_COMPLETE.json',{'signature':run.signature(),'complete':True})
    print('Complete OU cohort analyzed and verified.',flush=True)
