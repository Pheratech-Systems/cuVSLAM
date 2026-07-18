# Headless cuVSLAM + OAK-D Lite -> records a .rrd file you can open in the rerun viewer.
# Run it, move the camera, press Ctrl-C to stop. Then copy the .rrd to your computer
# and drag it into https://app.rerun.io  (or run `rerun cuvslam_run.rrd` if installed).
import os
from datetime import timedelta

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam
import rerun as rr

FPS = 30
RESOLUTION = (640, 480)
WARMUP_FRAMES = 60
CM_TO_M = 100.0
OUT_PATH = os.path.expanduser("~/cuvslam_run.rrd")


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


# --- rerun: write to a file instead of a live viewer ---
rr.init("cuVSLAM OAK-D")
rr.save(OUT_PATH)
print(f">> recording to {OUT_PATH}  (Ctrl-C to stop)", flush=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

device = dai.Device()
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

pipeline.start()
print(">> tracking; move the camera...", flush=True)

fid = 0
trajectory = []
try:
    while pipeline.isRunning():
        grp = q.tryGet()
        if grp is None:
            continue
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
            print(f"   logged {fid - WARMUP_FRAMES} tracked frames", flush=True)
except KeyboardInterrupt:
    pass
finally:
    # data flushes to the .rrd on interpreter exit; call these only if present
    for fn in ("flush", "disconnect"):
        try:
            getattr(rr, fn)()
        except Exception:
            pass
    print(f"\n>> saved {OUT_PATH}  ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)", flush=True)
