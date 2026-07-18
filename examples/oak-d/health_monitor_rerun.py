# Health-conscious cuVSLAM + OAK-D Lite runner for diagnosing USB disconnects (X_LINK_ERROR).
#
# Combines health_check.py (device power-supply health check) with the rerun cuVSLAM example
# and adds live under-load monitoring:
#
#   1. Pre-flight power-supply health check (baseline, before the device is opened).
#   2. On-device SystemLogger: chip temperature / CPU / memory streamed while cuVSLAM runs.
#   3. Frame-cadence watchdog + device log callback with host timestamps.
#
# Everything is mirrored to a plaintext diagnostic log (~/cuvslam_health_diag.log) that
# survives the crash. Trajectory, camera images and the temp/cpu/mem scalar plots are
# recorded to a .rrd file (~/cuvslam_health.rrd) you can open later in the rerun viewer.
# No live web viewer is started.
#
# On disconnect it prints an evidence-based verdict: POWER/THERMAL (brownout) vs. a
# PHYSICAL USB connection (cable/connector) fault.
#
#   python3 cuVSLAM/examples/oak-d/health_monitor_rerun.py
#
import os
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam
import rerun as rr

FPS = 30
RESOLUTION = (640, 480)
WARMUP_FRAMES = 60
CM_TO_M = 100.0

POWER_CHECK_MS = 20000          # pre-flight power-supply check duration
SYS_LOG_HZ = 2                  # on-device SystemLogger rate
METRICS_PRINT_EVERY_S = 5.0     # console/log cadence for system metrics
WINDOW_S = 8.0                  # rolling-history window used for the disconnect verdict
TEMP_HOT_C = 80.0               # chip temp considered thermally stressed
GAP_STALL_FACTOR = 2.5          # inter-frame gap this many x the baseline == "slowing down"

DIAG_PATH = os.path.expanduser("~/cuvslam_health_diag.log")
RRD_PATH = os.path.expanduser("~/cuvslam_health.rrd")


# ----------------------------------------------------------------------------- logging
_diag_fh = open(DIAG_PATH, "w", buffering=1)  # line-buffered so it survives a hard crash


def log(msg="", to_console=True):
    line = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
    _diag_fh.write(line + "\n")
    _diag_fh.flush()
    if to_console:
        print(msg, flush=True)


def log_scalar(path, value):
    # rerun renamed Scalar -> Scalars across versions; try both, never let it crash the run.
    for name in ("Scalars", "Scalar"):
        fn = getattr(rr, name, None)
        if fn is None:
            continue
        try:
            rr.log(path, fn(value))
            return
        except Exception:
            continue


# ----------------------------------------------------------------------------- cuVSLAM helpers
def to_pose(ext):
    a = np.array(ext)
    return vslam.Pose(rotation=Rotation.from_matrix(a[:3, :3]).as_quat(),
                      translation=a[:3, 3] / CM_TO_M)


def make_cam(p):
    c = vslam.Camera()
    c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, p['distortion'])
    c.focal = (p['intrinsics'][0][0], p['intrinsics'][1][1])
    c.principal = (p['intrinsics'][0][2], p['intrinsics'][1][2])
    c.size = p['resolution']
    c.rig_from_camera = to_pose(p['extrinsics'])
    return c


# ----------------------------------------------------------------------------- 1. pre-flight power check
def preflight_health_check():
    """Runs the standalone health check (health_check.py) before the pipeline is opened.
    Returns (ok, power_warning, raw_text)."""
    deviceInfos = dai.Device.getAllConnectedDevices()
    if not deviceInfos:
        log("No DepthAI device found.")
        return False, False, ""

    info = deviceInfos[0]
    log("=" * 78)
    log(">> PRE-FLIGHT HEALTH CHECK")
    log(f"   Device: {info}")
    log(f"   Power-supply check duration: {POWER_CHECK_MS} ms")
    log("   Running (device must be idle)...")

    try:
        cfg = dai.HealthCheckConfig(powerSupplyCheckDuration=timedelta(milliseconds=POWER_CHECK_MS))
        start = time.monotonic()
        metrics = dai.Device.performHealthCheck(info, cfg)
        elapsed = int((time.monotonic() - start) * 1000)
    except Exception as e:
        log(f"   health check unavailable on this depthai build: {e}")
        return True, False, ""

    text = str(metrics)
    log(f"   completed in {elapsed} ms")
    for ln in text.splitlines():
        log(f"     {ln}")

    # Heuristic: flag any mention of power/voltage/current not reading "OK"/"true".
    lower = text.lower()
    power_warn = ("power" in lower or "voltage" in lower or "current" in lower) and \
                 any(bad in lower for bad in ("fail", "warn", "false", "insufficient", "brown", "under"))
    if power_warn:
        log("   !! pre-flight flagged a POWER-SUPPLY concern (see above)")
    log("=" * 78)
    return True, power_warn, text


# ----------------------------------------------------------------------------- 2. system metrics extraction
def _avg(field):
    if field is None:
        return None
    for attr in ("average", "value"):
        v = getattr(field, attr, None)
        if v is not None:
            return float(v)
    try:
        return float(field)
    except (TypeError, ValueError):
        return None


def parse_sysinfo(info):
    """Pull the diagnostically useful numbers out of a SystemInformation message, defensively."""
    d = {}
    d["temp_c"] = _avg(getattr(info, "chipTemperature", None))
    css = _avg(getattr(info, "leonCssCpuUsage", None))
    mss = _avg(getattr(info, "leonMssCpuUsage", None))
    d["cpu"] = max([x for x in (css, mss) if x is not None], default=None)
    ddr = getattr(info, "ddrMemoryUsage", None)
    if ddr is not None:
        used = getattr(ddr, "used", None)
        total = getattr(ddr, "total", None)
        d["ddr_pct"] = (100.0 * used / total) if (used and total) else None
    else:
        d["ddr_pct"] = None
    return d


# ----------------------------------------------------------------------------- main
def main():
    log(f">> diagnostic log: {DIAG_PATH}")
    ok, power_warn, _ = preflight_health_check()
    if not ok:
        return 1

    # --- rerun: record to a .rrd file (test_record_rerun.py) ---
    rr.init("cuVSLAM OAK-D health monitor")
    rr.save(RRD_PATH)
    log(f">> recording to {RRD_PATH}  (metrics under world/metrics/*)")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    device = dai.Device()

    # Capture firmware/XLink log messages with host timestamps so we can pinpoint the disconnect.
    log_events = deque(maxlen=200)

    def on_log(rec):
        try:
            payload = getattr(rec, "payload", str(rec))
            level = getattr(getattr(rec, "level", None), "name", "")
            log_events.append((time.monotonic(), level, payload))
            log(f"   [device:{level}] {payload}", to_console=False)
        except Exception:
            pass

    for setter, arg in (("setLogLevel", dai.LogLevel.WARN),
                        ("setLogOutputLevel", dai.LogLevel.WARN)):
        try:
            getattr(device, setter)(arg)
        except Exception:
            pass
    try:
        device.addLogCallback(on_log)
    except Exception as e:
        log(f"   (device log callback unavailable: {e})")

    calib = device.readCalibration()

    def cam_params(socket_id):
        return {
            'resolution': RESOLUTION,
            'intrinsics': calib.getCameraIntrinsics(socket_id, RESOLUTION[0], RESOLUTION[1]),
            'extrinsics': calib.getCameraExtrinsics(socket_id, dai.CameraBoardSocket.CAM_A),
            'distortion': calib.getDistortionCoefficients(socket_id)[:8],
        }

    cameras = [make_cam(cam_params(dai.CameraBoardSocket.CAM_B)),
               make_cam(cam_params(dai.CameraBoardSocket.CAM_C))]
    cfg = vslam.Tracker.OdometryConfig(async_sba=False, enable_final_landmarks_export=True,
                                       enable_observations_export=True, rectified_stereo_camera=False)
    tracker = vslam.Tracker(vslam.Rig(cameras), cfg)

    pipeline = dai.Pipeline(device)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))
    lout = left.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    rout = right.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    lout.link(sync.inputs["left"])
    rout.link(sync.inputs["right"])
    q = sync.out.createOutputQueue()

    # --- 3. on-device system logger (temp / cpu / mem under load) ---
    sys_q = None
    try:
        syslog = pipeline.create(dai.node.SystemLogger)
        syslog.setRate(SYS_LOG_HZ)
        sys_q = syslog.out.createOutputQueue()
        log(">> SystemLogger attached (temp/cpu/mem streaming)")
    except Exception as e:
        log(f">> SystemLogger unavailable on this build: {e} (temp/cpu will not be monitored)")

    pipeline.start()
    log(">> tracking; move the camera and apply your load...")

    # rolling history for the verdict:  (t, frame_gap, temp, cpu, ddr)
    history = deque()
    frame_intervals = deque(maxlen=60)
    last_frame_t = None
    last_metrics = {"temp_c": None, "cpu": None, "ddr_pct": None}
    peak_temp = None
    start_t = time.monotonic()
    last_print = start_t

    fid = 0
    trajectory = []
    disconnect_reason = None

    def snapshot():
        return {
            "uptime_s": time.monotonic() - start_t,
            "frames": max(0, fid - WARMUP_FRAMES),
            "last": dict(last_metrics),
            "peak_temp": peak_temp,
        }

    try:
        while pipeline.isRunning():
            # -- system metrics (independent of frame arrival) --
            if sys_q is not None:
                si = sys_q.tryGet()
                if si is not None:
                    m = parse_sysinfo(si)
                    last_metrics = m
                    if m["temp_c"] is not None:
                        peak_temp = m["temp_c"] if peak_temp is None else max(peak_temp, m["temp_c"])
                        log_scalar("world/metrics/chip_temp_c", m["temp_c"])
                    if m["cpu"] is not None:
                        log_scalar("world/metrics/cpu", m["cpu"])
                    if m["ddr_pct"] is not None:
                        log_scalar("world/metrics/ddr_pct", m["ddr_pct"])

            now = time.monotonic()
            if now - last_print >= METRICS_PRINT_EVERY_S:
                last_print = now
                t = last_metrics
                log(f"   [{now - start_t:6.1f}s] frames={max(0, fid - WARMUP_FRAMES):5d} "
                    f"temp={t['temp_c'] if t['temp_c'] is None else round(t['temp_c'], 1)}C "
                    f"cpu={t['cpu'] if t['cpu'] is None else round(t['cpu'], 2)} "
                    f"ddr={t['ddr_pct'] if t['ddr_pct'] is None else round(t['ddr_pct'], 0)}%")

            grp = q.tryGet()
            if grp is None:
                continue

            # -- frame cadence watchdog --
            gap = None
            if last_frame_t is not None:
                gap = now - last_frame_t
                frame_intervals.append(gap)
            last_frame_t = now
            history.append((now, gap, last_metrics["temp_c"], last_metrics["cpu"], last_metrics["ddr_pct"]))
            while history and now - history[0][0] > WINDOW_S:
                history.popleft()

            fid += 1
            ts = int(grp.getTimestamp().total_seconds() * 1e9)
            if fid <= WARMUP_FRAMES:
                continue

            limg = grp["left"].getFrame()
            est, _ = tracker.track(ts, (limg, grp["right"].getFrame()))
            pose = est.world_from_rig
            if pose is None:
                continue

            t = pose.pose.translation
            trajectory.append(t)
            rr.set_time("frame", sequence=fid)
            rr.log("world/rig", rr.Transform3D(translation=t, quaternion=pose.pose.rotation),
                   rr.Arrows3D(vectors=np.eye(3) * 0.1, colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
            rr.log("world/trajectory", rr.LineStrips3D([trajectory]))
            rr.log("world/camera/image", rr.Image(limg))
            if fid % 60 == 0:
                log(f"   logged {fid - WARMUP_FRAMES} tracked frames", to_console=True)

    except KeyboardInterrupt:
        disconnect_reason = "user Ctrl-C"
    except Exception as e:
        disconnect_reason = f"{type(e).__name__}: {e}"
        log("")
        log("!! " + "=" * 74)
        log(f"!! DEVICE COMMUNICATION LOST: {disconnect_reason}")
        log("!! " + "=" * 74)
    finally:
        for fn in ("flush", "disconnect"):
            try:
                getattr(rr, fn)()
            except Exception:
                pass
        if disconnect_reason != "user Ctrl-C":
            verdict(snapshot(), frame_intervals, history, log_events, power_warn)
        try:
            size_mb = os.path.getsize(RRD_PATH) / 1e6
            log(f">> saved {RRD_PATH}  ({size_mb:.1f} MB)")
        except OSError:
            pass
        log(f">> diagnostic log saved: {DIAG_PATH}")
        _diag_fh.close()
    return 0


# ----------------------------------------------------------------------------- verdict
def verdict(snap, frame_intervals, history, log_events, power_warn):
    log("")
    log(">> DISCONNECT DIAGNOSIS " + "-" * 54)
    log(f"   uptime before failure : {snap['uptime_s']:.1f} s")
    log(f"   tracked frames        : {snap['frames']}")
    log(f"   last chip temp        : {snap['last']['temp_c']} C")
    log(f"   peak chip temp        : {snap['peak_temp']} C")
    log(f"   last cpu / ddr        : {snap['last']['cpu']} / {snap['last']['ddr_pct']}%")

    evidence_power = []
    evidence_physical = []

    # -- pre-flight power flag --
    if power_warn:
        evidence_power.append("pre-flight health check flagged the power supply")

    # -- thermal --
    peak = snap["peak_temp"]
    if peak is not None and peak >= TEMP_HOT_C:
        evidence_power.append(f"chip temperature reached {peak:.1f}C (>= {TEMP_HOT_C}C, thermal/power stress)")

    # -- frame-cadence trend: were frames slowing down before the cut? --
    ivals = list(frame_intervals)
    if len(ivals) >= 10:
        baseline = np.median(ivals[:-5]) if len(ivals) > 5 else np.median(ivals)
        tail = np.median(ivals[-3:])
        if baseline > 0 and tail >= baseline * GAP_STALL_FACTOR:
            evidence_power.append(
                f"frame cadence was degrading before the cut "
                f"(last gaps ~{tail * 1000:.0f}ms vs baseline ~{baseline * 1000:.0f}ms) — "
                f"device was struggling, consistent with a brownout/thermal throttle")
        else:
            evidence_physical.append(
                f"frames arrived on-cadence right up to the cut "
                f"(last gaps ~{tail * 1000:.0f}ms vs baseline ~{baseline * 1000:.0f}ms) — "
                f"an instantaneous cut, consistent with a physical USB disconnect")
    else:
        evidence_physical.append("too few frames to judge cadence degradation")

    # -- reconnection behaviour from device logs --
    recon = [p for (_, _, p) in log_events if "reconnect" in p.lower()]
    reset = [p for (_, _, p) in log_events if any(k in p.lower() for k in ("reset", "reboot", "crash", "watchdog"))]
    if reset:
        evidence_power.append("firmware reported a reset/reboot/watchdog event — the device power-cycled, "
                              "strongly suggesting a brownout rather than a passive cable disconnect")
    if recon:
        log(f"   device attempted reconnect ({len(recon)}x) — transient link loss (loose cable or recovered brownout)")

    # -- render --
    log("")
    log("   EVIDENCE FOR POWER / BROWNOUT / THERMAL:")
    if evidence_power:
        for e in evidence_power:
            log(f"     - {e}")
    else:
        log("     (none)")

    log("   EVIDENCE FOR PHYSICAL USB DISCONNECT (cable/connector):")
    if evidence_physical:
        for e in evidence_physical:
            log(f"     - {e}")
    else:
        log("     (none)")

    log("")
    if len(evidence_power) > len(evidence_physical):
        log("   >>> VERDICT: most likely POWER-RELATED (brownout under load / thermal).")
        log("       Try: a powered USB hub or the barrel-jack/Y-cable, a shorter/quality USB3 cable,")
        log("       and reduce load (lower FPS/resolution) to confirm it stops recurring.")
    elif len(evidence_physical) > len(evidence_power):
        log("   >>> VERDICT: most likely a PHYSICAL USB CONNECTION fault (cable/connector).")
        log("       Try: reseat both ends, swap the USB cable, try a different port, check strain/movement.")
    else:
        log("   >>> VERDICT: INCONCLUSIVE — signals are mixed. Re-run and reproduce; compare this log across runs.")
    log("   " + "-" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
