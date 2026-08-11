# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

# Headless cuVSLAM + RealSense stereo diagnostic — no rerun, no cv2.
# Prints each bring-up stage so we can see exactly where it stalls.
import time

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam
from camera_utils import get_rs_stereo_rig

RESOLUTION = (640, 360)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds
FRAME_TIMEOUT_MS = 2000


print(">> enumerating devices...", flush=True)
ctx = rs.context()
devices = list(ctx.query_devices())
if not devices:
    raise RuntimeError("No RealSense device connected")
for dev in devices:
    print(
        f">> found {dev.get_info(rs.camera_info.name)} "
        f"S/N={dev.get_info(rs.camera_info.serial_number)} "
        f"fw={dev.get_info(rs.camera_info.firmware_version)}",
        flush=True
    )

config = rs.config()
pipeline = rs.pipeline()
config.enable_stream(
    rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
)
config.enable_stream(
    rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
)

print(">> probing intrinsics/extrinsics...", flush=True)
pipeline.start(config)
frames = pipeline.wait_for_frames()
left_profile = frames[0].profile.as_video_stream_profile()
right_profile = frames[1].profile.as_video_stream_profile()
camera_params = {
    'left': {'intrinsics': left_profile.intrinsics},
    'right': {
        'intrinsics': right_profile.intrinsics,
        'extrinsics': right_profile.get_extrinsics_to(left_profile)
    }
}
pipeline.stop()

ki = camera_params['left']['intrinsics']
print(
    f">> left fx={ki.fx:.1f} fy={ki.fy:.1f} cx={ki.ppx:.1f} cy={ki.ppy:.1f} "
    f"size={ki.width}x{ki.height}",
    flush=True
)
baseline = camera_params['right']['extrinsics'].translation
print(f">> baseline (right->left) = {np.linalg.norm(baseline) * 1000:.1f} mm", flush=True)

cfg = vslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    enable_observations_export=True,
    rectified_stereo_camera=True
)
tracker = vslam.Tracker(get_rs_stereo_rig(camera_params), cfg)
print(">> tracker ready", flush=True)

# Disable the IR emitter so the dot pattern does not pollute stereo features
pipeline_profile = config.resolve(rs.pipeline_wrapper(pipeline))
depth_sensor = pipeline_profile.get_device().query_sensors()[0]
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    print(">> IR emitter disabled", flush=True)
else:
    print(">> IR emitter option not supported; leaving as is", flush=True)

pipeline.start(config)
print(">> pipeline started; waiting for frames (move the camera)...", flush=True)

frame_id = 0
prev_timestamp = None
last_report = time.time()

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        except RuntimeError:
            print("   ...no stereo frame yet (wait_for_frames timed out)", flush=True)
            last_report = time.time()
            continue

        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        if not left_frame or not right_frame:
            if time.time() - last_report > 2.0:
                print("   ...incomplete stereo pair", flush=True)
                last_report = time.time()
            continue

        frame_id += 1
        timestamp = int(left_frame.timestamp * 1e6)  # Convert to nanoseconds

        if prev_timestamp is not None:
            timestamp_diff = timestamp - prev_timestamp
            if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                print(
                    f"   frame {frame_id}: stream message drop: timestamp gap "
                    f"({timestamp_diff / 1e6:.2f} ms) exceeds threshold "
                    f"{IMAGE_JITTER_THRESHOLD_NS / 1e6:.2f} ms",
                    flush=True
                )
        prev_timestamp = timestamp

        if frame_id % 30 == 0:
            print(f">> received frame {frame_id} (ts={timestamp})", flush=True)

        if frame_id > WARMUP_FRAMES:
            images = (
                np.asanyarray(left_frame.get_data()),
                np.asanyarray(right_frame.get_data())
            )
            odom_pose_estimate, _ = tracker.track(timestamp, images)
            pose = odom_pose_estimate.world_from_rig
            if pose is None:
                print(f"   frame {frame_id}: TRACKING LOST", flush=True)
            else:
                t = pose.pose.translation
                print(
                    f"   frame {frame_id}: pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})",
                    flush=True
                )
finally:
    pipeline.stop()
