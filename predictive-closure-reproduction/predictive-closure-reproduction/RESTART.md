# Pause and restart

**The existing full run is paused, but its current weights are only in memory.** The official trainer had no checkpoint hook when it started. Logs and configuration have been archived, but a computer restart would lose the unsaved weights. We cannot safely add a hook inside that already-running process.

Restart-safe training is now implemented for the traffic experiment. It saves at update zero, every 10 completed updates, and on a pause request. Each checkpoint contains model and optimizer state, the completed-update count, data-generator and Python/NumPy/Torch random states, histories, configuration, code hashes and runtime details. Files are flushed to disk and published through an atomic pointer; the latest two snapshots are retained.

The adapter inserts save/restore hooks into the pinned trainer without changing the model, loss, data sampling or update equations. Its own code hash is recorded. Existing incomplete runs without a restart checkpoint are rejected rather than overwritten.

## Start a checkpoint-enabled model

**This starts the model again, rather than rescuing the paused process's weights.** It has not been started because the user asked to pause. Use a separate output folder to preserve all earlier files:

~~~bash
.venv-intel/bin/python launch.py \
  --upstream /absolute/path/to/predictive-closure-upstream \
  --output runs-local/restartable \
  --mode full --device cpu --cpu-fp32 --local-intel \
  --study traffic --seeds 303 --values 0 \
  --restartable --checkpoint-every 10 --skip-plot
~~~

Run this from the reproduction folder. Repeating the **same command** after a computer restart loads the latest saved state and skips completed updates. Architecture, training budget, data, code and runtime checks prevent silent changes. A crash can lose updates since the most recent checkpoint.

The trainer PID is saved in the model folder's `restart-process.json`. Request a durable pause by sending **SIGUSR1 to that trainer**, not to the old suspended process:

~~~bash
kill -USR1 TRAINER_PID
~~~

It finishes the current update, saves a checkpoint, and exits with code 75. Wait for `restart_checkpoint` and the pause message in `log_0.txt` before restarting the computer. A suspended process or a status JSON alone is not a durable weights checkpoint.

## What was tested

A small real-CIFAR test saved after update 3, exited, and resumed in a fresh Python process. At update 6, weights, optimizer tensors, data/Torch RNG states and training/evaluation histories were exactly equal to an uninterrupted run. The test uses a reduced architecture and one CPU thread; it validates checkpoint mechanics, not paper accuracy or bitwise reproducibility of every possible hardware backend.

Seven planner/runtime checks also pass, including refusal to overwrite the existing uncheckpointed run. No full model or paper accuracy is complete.
