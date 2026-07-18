# Headless cuVSLAM + OAK-D Lite with a browser-viewable rerun stream.
import socket
import time
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
WEB_PORT = 9090


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


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


# --- rerun: serve to the network, no local window ---
rr.init("cuVSLAM OAK-D")
if hasattr(rr, "serve_web"):
    rr.serve_web(open_browser=False, web_port=WEB_PORT)
else:                                   # newer rerun split the API
    rr.serve_grpc()
    rr.serve_web_viewer(open_browser=False, web_port=WEB_PORT)
print(f">> OPEN THIS ON YOUR COMPUTER:  http://{lan_ip()}:{WEB_PORT}", flush=True)
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
