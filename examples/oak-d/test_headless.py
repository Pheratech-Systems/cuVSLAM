# Headless cuVSLAM + OAK-D Lite diagnostic — no rerun, no cv2.
# Prints each bring-up stage so we can see exactly where it stalls.
import time
from datetime import timedelta

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

FPS = 30
RESOLUTION = (640, 480)          # OV7251 native VGA
WARMUP_FRAMES = 60
CM_TO_M = 100.0


def to_pose(ext):
    a = np.array(ext)
    R = a[:3, :3]
    t = a[:3, 3] / CM_TO_M
    return vslam.Pose(rotation=Rotation.from_matrix(R).as_quat(), translation=t)


def make_cam(p):
    c = vslam.Camera()
    c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, p['distortion'])
    c.focal = (p['intrinsics'][0][0], p['intrinsics'][1][1])
    c.principal = (p['intrinsics'][0][2], p['intrinsics'][1][2])
    c.size = p['resolution']
    c.rig_from_camera = to_pose(p['extrinsics'])
    return c


print(">> opening device...", flush=True)
device = dai.Device()
print(">> connected cameras:", [str(c) for c in device.getConnectedCameras()], flush=True)

calib = device.readCalibration()


def cam_params(socket):
    return {
        'resolution': RESOLUTION,
        'intrinsics': calib.getCameraIntrinsics(socket, RESOLUTION[0], RESOLUTION[1]),
        'extrinsics': calib.getCameraExtrinsics(socket, dai.CameraBoardSocket.CAM_A),
        'distortion': calib.getDistortionCoefficients(socket)[:8],
    }


left_p = cam_params(dai.CameraBoardSocket.CAM_B)
right_p = cam_params(dai.CameraBoardSocket.CAM_C)
ki = left_p['intrinsics']
print(f">> left fx={ki[0][0]:.1f} fy={ki[1][1]:.1f} cx={ki[0][2]:.1f} cy={ki[1][2]:.1f}", flush=True)

cameras = [make_cam(left_p), make_cam(right_p)]
cfg = vslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    enable_observations_export=True,
    rectified_stereo_camera=False,
)
tracker = vslam.Tracker(vslam.Rig(cameras), cfg)
print(">> tracker ready", flush=True)

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
print(">> pipeline started; waiting for frames (move the camera)...", flush=True)

fid = 0
last_report = time.time()
while pipeline.isRunning():
    grp = q.tryGet()
    if grp is None:
        if time.time() - last_report > 2.0:
            print("   ...no synced stereo frame yet", flush=True)
            last_report = time.time()
        continue

    fid += 1
    ts = int(grp.getTimestamp().total_seconds() * 1e9)
    if fid % 30 == 0:
        print(f">> received frame {fid} (ts={ts})", flush=True)

    if fid > WARMUP_FRAMES:
        limg = grp["left"].getFrame()
        rimg = grp["right"].getFrame()
        est, _ = tracker.track(ts, (limg, rimg))
        pose = est.world_from_rig
        if pose is None:
            print(f"   frame {fid}: TRACKING LOST", flush=True)
        else:
            t = pose.pose.translation
            print(f"   frame {fid}: pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})", flush=True)
