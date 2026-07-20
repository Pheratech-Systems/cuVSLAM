#!/usr/bin/env python3
"""
Health-conscious PyCuVSLAM stereo VO -> PX4 EKF2 bridge (fused diagnostic build).

Fuses:
  * pycuvslam_vio_bridge.py  -- the real bridge: OAK-D Lite stereo -> PyCuVSLAM ->
    px4_msgs/VehicleOdometry on /fmu/in/vehicle_visual_odometry over uXRCE-DDS.
  * health_monitor_rerun.py  -- under-load diagnostics: pre-flight power check,
    on-device SystemLogger (temp/cpu/mem), frame-cadence watchdog, device log
    callback with host timestamps, a crash-surviving diagnostic log, and an
    evidence-based POWER/THERMAL-vs-PHYSICAL-USB verdict on disconnect.

IMPORTANT -- load fidelity:
  By DEFAULT this reproduces the *bridge's* real load: it does NOT log images to
  rerun and does NOT write a .rrd. The only additions over the plain bridge are
  lightweight telemetry (2 Hz SystemLogger, a log callback, a text diag file),
  whose overhead is negligible. So the USB bandwidth, OAK power draw and cuVSLAM
  compute match the production bridge exactly -- this is the tool to run when you
  want to catch a disconnect under the *actual* flight workload.

  --record adds the heavy rerun image logging + .rrd disk write (the record-run
  load profile). Use it only when you want that heavier stress, not for a faithful
  bridge repro.

Modes:
  (default)     publish to PX4 + monitor
  --dry-run     monitor only, do NOT publish (and does not require ROS/px4_msgs)
  --record      also record trajectory/images/metrics to ~/cuvslam_vio.rrd (adds load)
  --fps N       override capture FPS (raise to increase load)

Runtime env (publish mode needs ROS msgs + the cuvslam/depthai wheels importable):
  source /opt/ros/humble/setup.bash
  source ~/px4_oakd_ws/install/setup.bash          # provides px4_msgs
  source ~/cuvslam-venv/bin/activate               # --system-site-packages venv
  python3 vio_bridge_monitored.py

FRAMES / EKF2 params: see pycuvslam_vio_bridge.py header -- unchanged here.
"""
import argparse
import os
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# ROS is optional so --dry-run works on a machine without px4_msgs/rclpy.
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from px4_msgs.msg import VehicleOdometry
    HAVE_ROS = True
except Exception as _ros_err:          # noqa: BLE001 - report lazily in main()
    HAVE_ROS = False
    _ROS_IMPORT_ERROR = _ros_err

# ------------------------------ config ---------------------------------------
FPS = 10
RESOLUTION = (640, 480)          # OV7251 native VGA
WARMUP_FRAMES = 60
CM_TO_M = 100.0
OUT_TOPIC = "/fmu/in/vehicle_visual_odometry"

# EV measurement noise (variance). Tune later; conservative-ish start.
POS_VAR = (0.01, 0.01, 0.04)     # m^2  (x, y, z)  -- z looser
ORI_VAR = (0.01, 0.01, 0.05)     # rad^2 (roll, pitch, yaw) -- yaw looser

# cuVSLAM optical (RDF) world -> PX4 NED. Bench-verify (see FRAMES in bridge header).
R_OPT2NED = Rotation.from_matrix(np.array([[0, 0, 1],
                                           [1, 0, 0],
                                           [0, 1, 0]], dtype=float))
NAN3 = (float("nan"),) * 3

# --- diagnostics config ---
POWER_CHECK_MS = 20000           # pre-flight power-supply check duration
SYS_LOG_HZ = 2                   # on-device SystemLogger rate
METRICS_PRINT_EVERY_S = 5.0      # console/log cadence for system metrics
WINDOW_S = 8.0                   # rolling-history window for the disconnect verdict
TEMP_HOT_C = 80.0                # chip temp considered thermally stressed
GAP_STALL_FACTOR = 2.5           # inter-frame gap this many x baseline == "slowing down"

DIAG_PATH = os.path.expanduser("~/cuvslam_vio_diag.log")
RRD_PATH = os.path.expanduser("~/cuvslam_vio.rrd")
# -----------------------------------------------------------------------------

_diag_fh = open(DIAG_PATH, "w", buffering=1)  # line-buffered so it survives a hard crash
_rr = None                                    # rerun module, only imported with --record


def log(msg="", to_console=True):
    line = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
    _diag_fh.write(line + "\n")
    _diag_fh.flush()
    if to_console:
        print(msg, flush=True)


def log_scalar(path, value):
    # only meaningful with --record; rerun renamed Scalar -> Scalars across versions.
    if _rr is None:
        return
    for name in ("Scalars", "Scalar"):
        fn = getattr(_rr, name, None)
        if fn is None:
            continue
        try:
            _rr.log(path, fn(value))
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


# ----------------------------------------------------------------------------- PX4 bridge node
if HAVE_ROS:
    class PyCuvslamVioBridge(Node):
        def __init__(self):
            super().__init__("pycuvslam_vio_bridge")
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.pub = self.create_publisher(VehicleOdometry, OUT_TOPIC, qos)
            self.reset_counter = 0
            self._was_tracking = False
            self.get_logger().info(f"publishing VehicleOdometry -> {OUT_TOPIC}")

        def _now_us(self) -> int:
            return int(self.get_clock().now().nanoseconds // 1000)

        def publish_pose(self, pos_world, quat_world_xyzw, tracking_ok: bool):
            if tracking_ok and not self._was_tracking:
                self.reset_counter += 1
            self._was_tracking = tracking_ok
            if not tracking_ok:
                return

            p_ned = R_OPT2NED.apply(np.asarray(pos_world, dtype=float))
            q_world = Rotation.from_quat(np.asarray(quat_world_xyzw, dtype=float))  # [x,y,z,w]
            q_ned = R_OPT2NED * q_world * R_OPT2NED.inv()
            qx, qy, qz, qw = q_ned.as_quat()

            m = VehicleOdometry()
            ts = self._now_us()
            m.timestamp = ts
            m.timestamp_sample = ts
            m.pose_frame = VehicleOdometry.POSE_FRAME_NED
            m.position = [float(p_ned[0]), float(p_ned[1]), float(p_ned[2])]
            m.q = [float(qw), float(qx), float(qy), float(qz)]      # PX4 order [w,x,y,z]
            m.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
            m.velocity = list(NAN3)
            m.angular_velocity = list(NAN3)
            m.position_variance = [float(v) for v in POS_VAR]
            m.orientation_variance = [float(v) for v in ORI_VAR]
            m.velocity_variance = list(NAN3)
            m.reset_counter = self.reset_counter % 256
            m.quality = 1
            self.pub.publish(m)


# ----------------------------------------------------------------------------- 1. pre-flight power check
def preflight_health_check():
    """Standalone power-supply check before the pipeline opens.
    Returns (ok, power_warning)."""
    deviceInfos = dai.Device.getAllConnectedDevices()
    if not deviceInfos:
        log("No DepthAI device found.")
        return False, False

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
        return True, False

    text = str(metrics)
    log(f"   completed in {elapsed} ms")
    for ln in text.splitlines():
        log(f"     {ln}")

    lower = text.lower()
    power_warn = ("power" in lower or "voltage" in lower or "current" in lower) and \
                 any(bad in lower for bad in ("fail", "warn", "false", "insufficient", "brown", "under"))
    if power_warn:
        log("   !! pre-flight flagged a POWER-SUPPLY concern (see above)")
    log("=" * 78)
    return True, power_warn


# ----------------------------------------------------------------------------- 2. system metrics
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


# ----------------------------------------------------------------------------- pipeline
def build_pipeline_and_tracker(fps):
    device = dai.Device()

    # capture firmware/XLink log messages with host timestamps to pinpoint the disconnect
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
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / fps))
    lout = left.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    rout = right.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    lout.link(sync.inputs["left"])
    rout.link(sync.inputs["right"])
    q = sync.out.createOutputQueue()

    # on-device system logger (temp/cpu/mem under load) -- negligible extra load
    sys_q = None
    try:
        syslog = pipeline.create(dai.node.SystemLogger)
        syslog.setRate(SYS_LOG_HZ)
        sys_q = syslog.out.createOutputQueue()
        log(">> SystemLogger attached (temp/cpu/mem streaming)")
    except Exception as e:
        log(f">> SystemLogger unavailable on this build: {e} (temp/cpu will not be monitored)")

    return pipeline, tracker, q, sys_q, log_events


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Health-monitored PyCuVSLAM VIO bridge for PX4.")
    ap.add_argument("--dry-run", action="store_true",
                    help="monitor only; do NOT publish to PX4 (no ROS required)")
    ap.add_argument("--record", action="store_true",
                    help="also record trajectory/images/metrics to a .rrd (ADDS host load)")
    ap.add_argument("--fps", type=int, default=FPS, help=f"capture FPS (default {FPS})")
    args = ap.parse_args()

    global _rr
    publish = not args.dry_run

    log(f">> diagnostic log: {DIAG_PATH}")
    log(f">> mode: {'PUBLISH->PX4' if publish else 'DRY-RUN (no publish)'}"
        f"{' + RECORD .rrd' if args.record else ''}  fps={args.fps}")

    if publish and not HAVE_ROS:
        log(f"!! ROS/px4_msgs not importable: {_ROS_IMPORT_ERROR}")
        log("   source your ROS + px4 workspace, or run with --dry-run to monitor only.")
        return 1

    ok, power_warn = preflight_health_check()
    if not ok:
        return 1

    # optional rerun recording (heavier load path)
    if args.record:
        import rerun as rr
        _rr = rr
        rr.init("cuVSLAM VIO bridge")
        rr.save(RRD_PATH)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        log(f">> recording to {RRD_PATH}  (metrics under world/metrics/*)")

    node = None
    if publish:
        rclpy.init()
        node = PyCuvslamVioBridge()

    pipeline, tracker, q, sys_q, log_events = build_pipeline_and_tracker(args.fps)
    pipeline.start()
    log(">> tracking started; move the camera / apply your load...")

    # rolling history for the verdict
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
        while pipeline.isRunning() and (node is None or rclpy.ok()):
            if node is not None:
                rclpy.spin_once(node, timeout_sec=0.0)

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
            if last_frame_t is not None:
                frame_intervals.append(now - last_frame_t)
            last_frame_t = now

            fid += 1
            ts = int(grp.getTimestamp().total_seconds() * 1e9)
            if fid <= WARMUP_FRAMES:
                continue

            limg = grp["left"].getFrame()
            rimg = grp["right"].getFrame()
            est, _ = tracker.track(ts, (limg, rimg))
            pose = est.world_from_rig
            if pose is None:
                if node is not None:
                    node.publish_pose(None, None, tracking_ok=False)
                continue

            if node is not None:
                node.publish_pose(pose.pose.translation, pose.pose.rotation, tracking_ok=True)

            if _rr is not None:
                t = pose.pose.translation
                trajectory.append(t)
                _rr.set_time("frame", sequence=fid)
                _rr.log("world/rig", _rr.Transform3D(translation=t, quaternion=pose.pose.rotation),
                        _rr.Arrows3D(vectors=np.eye(3) * 0.1,
                                     colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
                _rr.log("world/trajectory", _rr.LineStrips3D([trajectory]))
                _rr.log("world/camera/image", _rr.Image(limg))

    except KeyboardInterrupt:
        disconnect_reason = "user Ctrl-C"
    except Exception as e:
        disconnect_reason = f"{type(e).__name__}: {e}"
        log("")
        log("!! " + "=" * 74)
        log(f"!! DEVICE COMMUNICATION LOST: {disconnect_reason}")
        log("!! " + "=" * 74)
    finally:
        if _rr is not None:
            for fn in ("flush", "disconnect"):
                try:
                    getattr(_rr, fn)()
                except Exception:
                    pass
            try:
                log(f">> saved {RRD_PATH}  ({os.path.getsize(RRD_PATH) / 1e6:.1f} MB)")
            except OSError:
                pass
        if node is not None:
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        if disconnect_reason is not None and disconnect_reason != "user Ctrl-C":
            verdict(snapshot(), frame_intervals, log_events, power_warn)
        log(f">> diagnostic log saved: {DIAG_PATH}")
        _diag_fh.close()
    return 0


# ----------------------------------------------------------------------------- verdict
def verdict(snap, frame_intervals, log_events, power_warn):
    log("")
    log(">> DISCONNECT DIAGNOSIS " + "-" * 54)
    log(f"   uptime before failure : {snap['uptime_s']:.1f} s")
    log(f"   tracked frames        : {snap['frames']}")
    log(f"   last chip temp        : {snap['last']['temp_c']} C")
    log(f"   peak chip temp        : {snap['peak_temp']} C")
    log(f"   last cpu / ddr        : {snap['last']['cpu']} / {snap['last']['ddr_pct']}%")

    evidence_power = []
    evidence_physical = []

    if power_warn:
        evidence_power.append("pre-flight health check flagged the power supply")

    peak = snap["peak_temp"]
    if peak is not None and peak >= TEMP_HOT_C:
        evidence_power.append(f"chip temperature reached {peak:.1f}C (>= {TEMP_HOT_C}C, thermal/power stress)")

    ivals = list(frame_intervals)
    if len(ivals) >= 10:
        baseline = np.median(ivals[:-5]) if len(ivals) > 5 else np.median(ivals)
        tail = np.median(ivals[-3:])
        if baseline > 0 and tail >= baseline * GAP_STALL_FACTOR:
            evidence_power.append(
                f"frame cadence was degrading before the cut "
                f"(last gaps ~{tail * 1000:.0f}ms vs baseline ~{baseline * 1000:.0f}ms) -- "
                f"device was struggling, consistent with a brownout/thermal throttle")
        else:
            evidence_physical.append(
                f"frames arrived on-cadence right up to the cut "
                f"(last gaps ~{tail * 1000:.0f}ms vs baseline ~{baseline * 1000:.0f}ms) -- "
                f"an instantaneous cut, consistent with a physical USB disconnect")
    else:
        evidence_physical.append("too few frames to judge cadence degradation")

    recon = [p for (_, _, p) in log_events if "reconnect" in p.lower()]
    reset = [p for (_, _, p) in log_events if any(k in p.lower() for k in ("reset", "reboot", "crash", "watchdog"))]
    if reset:
        evidence_power.append("firmware reported a reset/reboot/watchdog event -- the device power-cycled, "
                              "strongly suggesting a brownout rather than a passive cable disconnect")
    if recon:
        log(f"   device attempted reconnect ({len(recon)}x) -- transient link loss (loose cable or recovered brownout)")

    log("")
    log("   EVIDENCE FOR POWER / BROWNOUT / THERMAL:")
    for e in (evidence_power or ["(none)"]):
        log(f"     - {e}")
    log("   EVIDENCE FOR PHYSICAL USB DISCONNECT (cable/connector):")
    for e in (evidence_physical or ["(none)"]):
        log(f"     - {e}")

    log("")
    if len(evidence_power) > len(evidence_physical):
        log("   >>> VERDICT: most likely POWER-RELATED (brownout under load / thermal).")
        log("       Try: powered USB hub / barrel-jack, shorter quality USB3 cable, lower FPS/res.")
    elif len(evidence_physical) > len(evidence_power):
        log("   >>> VERDICT: most likely a PHYSICAL USB CONNECTION fault (cable/connector).")
        log("       Try: reseat both ends, swap the cable, different port, check strain/movement.")
    else:
        log("   >>> VERDICT: INCONCLUSIVE -- mixed signals. Re-run and compare this log across runs.")
    log("   " + "-" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
