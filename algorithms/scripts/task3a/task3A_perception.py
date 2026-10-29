#!/usr/bin/env python3
"""Detect the Task 3A ores and the ore package, and publish them as transforms in ``base_link``."""

# Team ID:          [6131]
# Author List:      [Agrim Chauhan, Saksham Kaushal, Rayhan Khan, Manan Sood]
# Filename:         task3A_perception.py
# Task:             3A - StrataCobot (eYRC 2026-27)
# Functions:
#                   _rotate_points, __init__, _color_cb, _depth_cb, _info_cb, _find_regions,
#                   _point_in_base, _base_points, _assign_ids, _at_arm_cb, _cleared_cb, _find_package,
#                   _process, main
# Nodes:
#                   Name - [ task3a_perception ]
#                   Publishing Topics - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                       /camera/camera/aligned_depth_to_color/image_raw,
#                       /camera/camera/color/camera_info, /ebot_at_arm,
#                       /package_cleared, /tf, /tf_static, /clock ]

################### IMPORT MODULES #######################

import math
import sys

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool


BASE_FRAME = 'base_link'
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'
ORE_TYPES = ('azurite_ore', 'malachite_ore', 'vanadinite_ore')
# The four physical ores may have different kinds in different worlds. Limit
# created IDs by total object count, rather than assuming a fixed color mix.
MAX_TOTAL_ORES = 4
TRACK_RETENTION_S = 1800.0

# The ore package. Its top face carries a purple panel (diffuse 0.216, 0, 0.216 in
# the model) inside a black border; nothing else in the arm cell is purple.
PACKAGE_FRAME = 'ore_package'
PACKAGE_HSV = (np.array([135, 80, 20]), np.array([165, 255, 255]))
PACKAGE_MIN_AREA_PX = 60.0
# Where the deck of an eBot parked beside the arm can be, in base_link (m).
# Measured 28 Oct: top face centre at (0.651, 0.091, 0.255).
PACKAGE_ROI = ((0.25, 1.05), (-0.45, 0.45), (0.10, 0.45))
# Handover topics (latched), shared with task3A_planning / task3A_manipulation.
AT_ARM_TOPIC = '/ebot_at_arm'
CLEARED_TOPIC = '/package_cleared'

COLOR_RANGES = {
    'azurite_ore': (np.array([95, 150, 80]), np.array([115, 255, 255])),
    'malachite_ore': (np.array([65, 150, 100]), np.array([85, 255, 255])),
    'vanadinite_ore': (np.array([5, 230, 100]), np.array([15, 255, 255])),
}


def _rotate_points(points, quaternion):
    """Rotate an Nx3 array by a ROS xyzw quaternion."""
    q = np.array([quaternion.x, quaternion.y, quaternion.z], dtype=np.float64)
    w = float(quaternion.w)
    return points + 2.0 * np.cross(q, np.cross(q, points) + w * points)


class OrePerception(Node):
    def __init__(self):
        super().__init__(
            'task3a_perception',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)],
        )
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.color_msg = None
        self.depth_msg = None
        self.intrinsics = None
        # Per-kind 3-D tracks keep a name attached to the same ore between frames.
        self.tracks = {kind: {} for kind in ORE_TYPES}
        self.next_id = {kind: 1 for kind in ORE_TYPES}
        self.tracks_created = 0
        self.marker_areas = {}
        self.ebot_at_arm = False
        self.package_cleared = False
        self.package_logged = False
        self.package_miss_log_time = 0.0

        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Bool, AT_ARM_TOPIC, self._at_arm_cb, latched_qos)
        self.create_subscription(Bool, CLEARED_TOPIC, self._cleared_cb, latched_qos)

        self.create_subscription(
            Image, COLOR_TOPIC, self._color_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._info_cb, qos_profile_sensor_data)
        self.create_timer(0.15, self._process)
        self.get_logger().info('Task 3A perception ready; waiting for camera data.')

    def _color_cb(self, msg):
        self.color_msg = msg

    def _depth_cb(self, msg):
        self.depth_msg = msg

    def _info_cb(self, msg):
        if msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.intrinsics = (float(msg.k[0]), float(msg.k[4]),
                               float(msg.k[2]), float(msg.k[5]))

    def _at_arm_cb(self, msg):
        self.ebot_at_arm = bool(msg.data)

    def _cleared_cb(self, msg):
        self.package_cleared = bool(msg.data)

    def _find_package(self, image):
        """Return (point, yaw) of the package's top-face centre in base_link, or None.

        The purple panel is the package's top face. Its pixels give the 3-D centre
        through the depth image; the long axis of those points gives the yaw.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, PACKAGE_HSV[0], PACKAGE_HSV[1])
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_px = int(np.count_nonzero(mask))
        best = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < PACKAGE_MIN_AREA_PX:
                break
            points = self._base_points(
                contour, mask, self.depth_image, self.depth_msg.header.stamp,
                self.depth_msg.header.frame_id)
            if points is None or len(points) < 10:
                continue
            centre = np.median(points, axis=0)
            (x0, x1), (y0, y1), (z0, z1) = PACKAGE_ROI
            if not (x0 < centre[0] < x1 and y0 < centre[1] < y1 and z0 < centre[2] < z1):
                continue
            xy = points[:, :2] - centre[:2]
            # Principal axis of the panel = the package's long side.
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
            major = eigenvectors[:, int(np.argmax(eigenvalues))]
            yaw = math.atan2(major[1], major[0])
            best = (centre, yaw, cv2.contourArea(contour))
            break
        return best, purple_px

    def _find_regions(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        regions = []
        kernel = np.ones((3, 3), np.uint8)
        for kind, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 35.0:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] <= 0.0:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                regions.append((kind, (cx, cy), contour, mask))
        return regions

    def _point_in_base(self, contour, mask, depth, stamp, camera_frame):
        base_points = self._base_points(contour, mask, depth, stamp, camera_frame)
        if base_points is None:
            return None
        # The magnetic tool approaches from above. Estimate the center of the visible
        # upper face rather than using a single center-pixel ray, which can hit a side.
        top_cut = np.percentile(base_points[:, 2], 70.0)
        top = base_points[base_points[:, 2] >= top_cut]
        # The uppermost pixels can lie off-center on a faceted ore, so use the whole
        # color region for its horizontal center and only the upper face for height.
        estimate = np.array((
            np.median(base_points[:, 0]),
            np.median(base_points[:, 1]),
            np.median(top[:, 2]),
        ))
        return estimate

    def _base_points(self, contour, mask, depth, stamp, camera_frame):
        """3-D points (base_link, Nx3) of the pixels inside one colour region."""
        height, width = depth.shape[:2]
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        # An occluding wrist can cut a hole into a colour region. Never use its
        # depth (or background pixels) as part of the ore surface.
        region_mask = cv2.bitwise_and(region_mask, mask)
        ys, xs = np.nonzero(region_mask)
        if xs.size == 0:
            return None
        # Subsample large blobs while retaining enough pixels for a robust surface fit.
        stride = max(1, int(math.sqrt(xs.size / 700.0)))
        xs = xs[::stride]
        ys = ys[::stride]
        zs = depth[ys, xs].astype(np.float64)
        good = np.isfinite(zs) & (zs > 0.08) & (zs < 5.0)
        xs, ys, zs = xs[good], ys[good], zs[good]
        if zs.size < 6:
            # A small centre patch often has valid depth where the contour edge does not.
            cx = int(np.mean(xs)) if xs.size else width // 2
            cy = int(np.mean(ys)) if ys.size else height // 2
            return None

        fx, fy, cx, cy = self.intrinsics
        camera_points = np.column_stack(((xs - cx) * zs / fx,
                                         (ys - cy) * zs / fy, zs))
        try:
            stamp_time = rclpy.time.Time.from_msg(stamp)
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, camera_frame, stamp_time)
        except Exception:
            return None

        rotated = _rotate_points(camera_points, transform.transform.rotation)
        translation = transform.transform.translation
        return rotated + np.array([translation.x, translation.y, translation.z])

    def _assign_ids(self, regions):
        positions_by_kind = {kind: [] for kind in ORE_TYPES}
        for kind, pixel, contour, mask in regions:
            point = self._point_in_base(
                contour, mask, self.depth_image, self.depth_msg.header.stamp,
                self.depth_msg.header.frame_id)
            if (point is not None and -1.1 < point[0] < 0.35
                    and -1.1 < point[1] < -0.25
                    and -0.12 < point[2] < -0.02):
                positions_by_kind[kind].append((pixel, point, cv2.contourArea(contour)))

        result = []
        now = self.get_clock().now().nanoseconds / 1e9
        for kind, detections in positions_by_kind.items():
            detections.sort(key=lambda item: item[0][0])
            old = self.tracks[kind]
            candidates = []
            for pixel, point, area in detections:
                for track_id, (previous, last_seen) in old.items():
                    if now - last_seen > TRACK_RETENTION_S:
                        continue
                    distance = float(np.linalg.norm(point - previous))
                    # Do not rename a different ore when a track is occluded.
                    if distance < 0.10:
                        candidates.append((distance, track_id, pixel, point, area))
            candidates.sort(key=lambda item: item[0])
            assigned_ids = set()
            assigned_detections = set()
            current = {
                track_id: track_data for track_id, track_data in old.items()
                if now - track_data[1] <= TRACK_RETENTION_S
            }
            for _, track_id, pixel, point, area in candidates:
                # Pixel centers are unique for separate contours in a frame.
                if track_id in assigned_ids or pixel in assigned_detections:
                    continue
                assigned_ids.add(track_id)
                assigned_detections.add(pixel)
                reference = self.marker_areas[(kind, track_id)]
                if area < 0.85 * reference:
                    # A fragment's centroid is not the ore centre. Keep the track
                    # for reassociation, but publish no position from that fragment.
                    continue
                self.marker_areas[(kind, track_id)] = max(reference, area)
                current[track_id] = (point, now)
                result.append((kind, track_id, point))

            for pixel, point, area in detections:
                if pixel in assigned_detections:
                    continue
                if self.tracks_created >= MAX_TOTAL_ORES:
                    continue
                track_id = self.next_id[kind]
                self.next_id[kind] += 1
                self.tracks_created += 1
                current[track_id] = (point, now)
                self.marker_areas[(kind, track_id)] = area
                assigned_ids.add(track_id)
                assigned_detections.add(pixel)
                result.append((kind, track_id, point))
                self.get_logger().info(
                    f'New {kind}_{track_id} at pixel {pixel}, '
                    f'base=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}).')
            self.tracks[kind] = current
        return result

    def _process(self):
        if self.color_msg is None or self.depth_msg is None or self.intrinsics is None:
            return
        if abs((self.color_msg.header.stamp.sec - self.depth_msg.header.stamp.sec)
               + (self.color_msg.header.stamp.nanosec
                  - self.depth_msg.header.stamp.nanosec) / 1e9) > 0.15:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(self.color_msg, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(
                self.depth_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self.get_logger().warning(f'Camera conversion failed: {exc}')
            return
        if self.depth_image.ndim != 2:
            return

        regions = self._find_regions(image)
        detections = self._assign_ids(regions)
        transforms = []
        for kind, track_id, point in detections:
            msg = TransformStamped()
            msg.header.stamp = self.depth_msg.header.stamp
            msg.header.frame_id = BASE_FRAME
            msg.child_frame_id = f'{kind}_{track_id}'
            msg.transform.translation.x = float(point[0])
            msg.transform.translation.y = float(point[1])
            msg.transform.translation.z = float(point[2])
            msg.transform.rotation.w = 1.0
            transforms.append(msg)

        # The package is published only while it sits on the eBot beside the arm.
        if self.ebot_at_arm and not self.package_cleared:
            found, purple_px = self._find_package(image)
            now = self.get_clock().now().nanoseconds / 1e9
            if found is not None:
                centre, yaw, area = found
                msg = TransformStamped()
                msg.header.stamp = self.depth_msg.header.stamp
                msg.header.frame_id = BASE_FRAME
                msg.child_frame_id = PACKAGE_FRAME
                msg.transform.translation.x = float(centre[0])
                msg.transform.translation.y = float(centre[1])
                msg.transform.translation.z = float(centre[2])
                msg.transform.rotation.z = math.sin(yaw / 2.0)
                msg.transform.rotation.w = math.cos(yaw / 2.0)
                transforms.append(msg)
                if not self.package_logged:
                    self.get_logger().info(
                        f'Package detected: {area:.0f} px, base=({centre[0]:.3f}, '
                        f'{centre[1]:.3f}, {centre[2]:.3f}), yaw {math.degrees(yaw):.1f} deg.')
                    self.package_logged = True
            elif now - self.package_miss_log_time > 3.0:
                self.package_miss_log_time = now
                self.get_logger().warning(
                    f'eBot is at the arm but no package found; {purple_px} purple '
                    'pixels in view (arm occlusion, or the colour range needs tuning).')
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = OrePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
