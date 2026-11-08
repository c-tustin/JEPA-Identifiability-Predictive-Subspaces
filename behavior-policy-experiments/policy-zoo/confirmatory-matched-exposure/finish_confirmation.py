"""Wait for this fixed cohort, then run locked inference and verification.

Does not start/restart training, replace seeds or infer on partial outcomes.
A failed subprocess stops this pipeline and preserves all completed artifacts.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import run_confirmation as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run.check_lock()
    complete = run.OUT / 'TRAINING_COMPLETE.json'
    while not complete.exists():
        if not args.wait:
            raise SystemExit('Training not complete; no inference run. Use --wait to leave a waiting finisher.')
        time.sleep(30)
    finished = json.loads(complete.read_text())
    assert finished['signature'] == run.signature() and finished['encoders'] == 540 and finished['predictors'] == 1080
    stages = [('analyze_confirmation.py', '--require-complete')]
    if not (run.OUT / 'FRESH_REPLAY_GATES.json').exists():
        stages.append(('replay_confirmation.py',))
    stages += [('verify_confirmation.py', '--workers', str(args.workers)), ('plot_confirmation.py',)]
    for stage in stages:
        print('Starting:', stage[0], flush=True)
        subprocess.run([sys.executable, str(run.HERE / stage[0]), *stage[1:]], check=True)
    run.save_json(run.OUT / 'PIPELINE_COMPLETE.json', {'signature': run.signature(),
                  'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'inference_and_verification_complete': True})
    print('Cohort inference, verification and figures complete. Publication still requires review.', flush=True)


if __name__ == '__main__':
    main()
