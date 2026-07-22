#!/usr/bin/env python3
"""Headless validation harness for the PyCuVSLAM dataset examples.

Runs any of the stock example scripts (track_kitti.py, track_euroc.py,
track_multicamera_tartan.py, track_multisensor_tartan.py, ...) WITHOUT editing
them, on a machine with no display (e.g. a headless Jetson). It does three
things the stock scripts don't do headlessly:

  1. Forces Rerun out of `spawn=True` (GUI) mode into file-record mode, writing
     a `.rrd` you can copy to a laptop and open with `rerun run.rrd`.
  2. Times every `tracker.track()` call and prints FPS / latency stats so you
     get an on-device performance number.
  3. Dumps the estimated trajectory in TUM format (timestamp tx ty tz qx qy qz
     qw) so you can score accuracy against dataset ground truth with `evo`.

Usage:
    python3 run_headless.py examples/euroc/track_euroc.py
    python3 run_headless.py examples/kitti/track_kitti.py --rrd /tmp/kitti.rrd
    python3 run_headless.py examples/multisensor/track_multisensor_tartan.py -- --no-imu

Everything after a bare `--` is forwarded to the example script as its own argv
(needed for scripts that take flags, e.g. multisensor's --no-imu).
"""

import argparse
import atexit
import os
import runpy
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('script', help='Path to the example script to run.')
    parser.add_argument('--rrd', default=None,
                        help='Output .rrd path (default: <script_name>.rrd in cwd).')
    parser.add_argument('--tum', default=None,
                        help='Output TUM trajectory path (default: <script_name>_traj.tum).')
    parser.add_argument('script_args', nargs=argparse.REMAINDER,
                        help='Args after `--` are forwarded to the example script.')
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        parser.error(f"script not found: {script_path}")

    stem = os.path.splitext(os.path.basename(script_path))[0]
    rrd_path = os.path.abspath(args.rrd) if args.rrd else os.path.abspath(f"{stem}.rrd")
    tum_path = os.path.abspath(args.tum) if args.tum else os.path.abspath(f"{stem}_traj.tum")

    # --- 1. force Rerun into headless file-record mode ---------------------
    import rerun as rr
    _orig_init = rr.init

    def _headless_init(*a, **k):
        k['spawn'] = False               # never try to open a native window
        result = _orig_init(*a, **k)
        rr.save(rrd_path)                # route all subsequent rr.log to a file
        print(f"[headless] Rerun recording to {rrd_path}", flush=True)
        return result

    rr.init = _headless_init

    # --- 2 & 3. time track() and capture the trajectory --------------------
    import cuvslam
    _orig_track = cuvslam.Tracker.track
    durations = []
    trajectory = []  # (timestamp_ns, tx, ty, tz, qx, qy, qz, qw)

    def _instrumented_track(self, timestamp, *a, **k):
        t0 = time.perf_counter()
        estimate, slam_pose = _orig_track(self, timestamp, *a, **k)
        durations.append(time.perf_counter() - t0)
        wfr = getattr(estimate, 'world_from_rig', None)
        if wfr is not None:
            pose = getattr(wfr, 'pose', wfr)  # PoseEstimate exposes .pose
            tx, ty, tz = pose.translation
            qx, qy, qz, qw = pose.rotation    # cuVSLAM quaternion order is xyzw
            trajectory.append((int(timestamp), tx, ty, tz, qx, qy, qz, qw))
        return estimate, slam_pose

    cuvslam.Tracker.track = _instrumented_track

    def _report():
        if durations:
            ms = [d * 1e3 for d in durations]
            fps = len(durations) / sum(durations)
            print("\n[headless] ===== validation summary =====", flush=True)
            print(f"[headless] frames tracked : {len(durations)}")
            print(f"[headless] tracked poses  : {len(trajectory)} "
                  f"({len(durations) - len(trajectory)} failed)")
            print(f"[headless] track() latency: mean {statistics.mean(ms):.2f} ms  "
                  f"median {statistics.median(ms):.2f} ms  max {max(ms):.2f} ms")
            print(f"[headless] throughput     : {fps:.1f} FPS")
        if trajectory:
            with open(tum_path, 'w') as f:
                for ts, tx, ty, tz, qx, qy, qz, qw in trajectory:
                    # TUM expects seconds; dataset timestamps are ns
                    f.write(f"{ts / 1e9:.9f} {tx:.6f} {ty:.6f} {tz:.6f} "
                            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            start = trajectory[0][1:4]
            end = trajectory[-1][1:4]
            print(f"[headless] trajectory     : {tum_path}")
            print(f"[headless] start xyz (m)  : {start[0]:.3f} {start[1]:.3f} {start[2]:.3f}")
            print(f"[headless] end   xyz (m)  : {end[0]:.3f} {end[1]:.3f} {end[2]:.3f}")

    atexit.register(_report)

    # --- run the untouched example script ----------------------------------
    # chdir into the example dir so its relative "dataset/..." paths resolve,
    # and pass the absolute path so its os.path.dirname(__file__) also works.
    os.chdir(os.path.dirname(script_path))
    forwarded = args.script_args
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    sys.argv = [script_path] + forwarded
    runpy.run_path(script_path, run_name='__main__')


if __name__ == '__main__':
    main()
