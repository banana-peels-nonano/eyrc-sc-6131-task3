#!/usr/bin/env python3
"""Task 3A arm: sort the ores (as in 2A) and move the ore package from the eBot to the crusher."""

# Team ID:          [6131]
# Author List:      [Agrim Chauhan, Saksham Kaushal, Rayhan Khan, Manan Sood]
# Filename:         task3A_manipulation.py
# Task:             3A - StrataCobot (eYRC 2026-27)
# Functions:
#                   __init__, _now_seconds, _tcp_cb, _joints_cb, _status_cb, _force_cb, _position,
#                   _tool_alignment, _publish_twist, _publish_joints, _stop, _move_toward,
#                   _change_phase, _switch, _switch_result, _fresh_ore_transforms, _drop_target,
#                   _side_x, _reach_safe, _clamp_transit_waypoints, _start_next_ore, _target_transform_position, _refresh_pick_target,
#                   _begin_capture_check, _call_magnet, _service_result, _held, _fault,
#                   _at_arm_cb, _publish_cleared, _package_transform, _joint_move_toward,
#                   _begin_sweep, _on_sweep_circle,
#                   _start_package, _package_step, _step, main
# Nodes:
#                   Name - [ task3a_manipulation ]
#                   Publishing Topics - [ /delta_twist_cmds, /delta_joint_cmds, /package_cleared ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status,
#                       /ur7e/end_effector/force_magnitude, /ebot_at_arm, /tf, /tf_static,
#                       /clock ]
#                   Service Clients - [ /ur_arm_controller/switch_controller, /magnet ]

################### IMPORT MODULES #######################

import math
import sys

import rclpy
import tf2_ros
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import SetBool


BASE_FRAME = 'base_link'
SERVO_NS = '/ur_arm_controller'
TWIST_CONTROLLER = 'delta_twist_controller'
JOINT_CONTROLLER = 'delta_joint_controller'
ORE_KINDS = ('azurite_ore', 'malachite_ore', 'vanadinite_ore')
JOINT_NAMES = (
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
)

# Cargo rover world pose and scale come from eyantra_kepler_world/model.sdf and the
# cargo_rover GLB node. The three wells run across the rover's local X axis. The task's
# color order follows the left-to-right well order in that asset.
ARM_BASE_WORLD = (-0.597278, 1.8653, 0.8825)
CARGO_ROVER_WORLD = (-0.93719, 2.5537, 0.568798)
WELL_LOCAL_X = {
    # Bin order is viewed from the arm side, opposite the rover's forward view.
    'azurite_ore': 0.33,
    'malachite_ore': 0.04,
    'vanadinite_ore': -0.27,
}
# Offset the far-left well toward the arm so its drop pose remains reachable; this
# is along the rover's local depth axis and stays inside the well.
WELL_LOCAL_Z = {
    'azurite_ore': 0.0,
    'malachite_ore': 0.0,
    'vanadinite_ore': 0.16,
}
DROP_HEIGHT_BASE = 0.04

# This pose points the magnet's local +X face vertically down while keeping the
# arm above the conveyor, ready for Cartesian pick-and-place motion.
UNFOLD_POSE = (-2.0216, -2.1366, -1.5538, -1.0221, 1.5708, 1.5708)
JOINT_SPEED_LIMIT = 0.40     # rad/s; Task 3 limit is 0.60 (was 0.20 in 2A)
TWIST_SPEED_LIMIT = 0.20     # m/s; Task 3 limit is 0.25 (was 0.12 in 2A)
POSITION_GAIN = 1.5          # 1/s; tail time constant 0.67 s (was 0.75 -> 1.33 s)
ORIENTATION_GAIN = 1.5
ANGULAR_SPEED_LIMIT = 0.3
ORIENTATION_TOLERANCE = math.radians(0.5)
POSITION_TOLERANCE = 0.004
VIA_TOLERANCE = 0.03    # m; waypoints the arm only passes through
PICK_POSITION_TOLERANCE = 0.0005
FORCE_HELD_THRESHOLD_N = 0.50
FORCE_APPROACH_LIMIT_N = 15.0
TF_MAX_AGE_S = 0.8
EXPECTED_ORE_COUNT = 4
CARRY_CLEARANCE = 0.24
PICK_APPROACH_STANDOFF = 0.025
# Scan a narrow height range around a fixed camera estimate. Keep the image-derived
# height fixed across attempts so camera noise cannot cancel the commanded offset.
PICK_SEARCH_OFFSETS = (0.0, -0.001, -0.002, -0.003, 0.001, 0.002, 0.003)
PICK_XY_SEARCH_OFFSETS = (
    (0.0, 0.0),
    (0.0, 0.005), (0.0, 0.010), (0.0, 0.015), (0.0, 0.020),
    (0.005, 0.0), (0.010, 0.0), (-0.005, 0.0), (-0.010, 0.0),
    (0.0, -0.005),
)
FORCE_COLLISION_THRESHOLD_N = 5.0

# ---- Task 3A: the ore package ---------------------------------------------------
# Handover topics (latched), shared with task3A_planning and task3A_perception.
AT_ARM_TOPIC = '/ebot_at_arm'
CLEARED_TOPIC = '/package_cleared'
PACKAGE_FRAME = 'ore_package'
# Accept a package transform only where an eBot deck beside the arm can be.
# Measured 28 Oct: top face centre at (0.651, 0.091, 0.255) in base_link.
PACKAGE_ROI = ((0.25, 1.05), (-0.45, 0.45), (0.10, 0.45))
PACKAGE_HELD_N = 2.0            # the package reads about 2.94 N when held
PACKAGE_STANDOFF = 0.02         # m above the top face before the magnet goes on
PACKAGE_PRESS = 0.004           # m below the measured top face for the contact move
PACKAGE_CLEARED_RISE = 0.08     # m above the top face before the eBot may leave
PACKAGE_CARRY_Z = 0.42          # m; package bottom then ~0.34, over everything nearby
PACKAGE_MAX_TRIES = 3
# Hopper (crusher collision mesh, 27 Oct): opening x -1.123..-0.763, y -0.195..0.187
# and rim z +0.207 in base_link. Reach test (29 Oct): with the tool tilted 20 deg
# outward, the singularity warning starts at TCP x = -0.917 (z 0.33). Dropping at
# -0.88 puts the package centre near -0.893, about 4 cm inside the near rim with
# the long side along x, and keeps 3.7 cm from that warning.
HOPPER_DROP = (-0.88, 0.0, 0.34)
HOPPER_TILT = math.radians(20.0)
HOPPER_STAGING = (-0.62, 0.0, PACKAGE_CARRY_Z)
# Big heading changes are made by turning shoulder_pan alone ("sweeps"): the TCP
# then circles the base at constant radius and height, and we choose which way
# round so the pan joint stays inside its +-2 pi limit. Run 29 Oct: a Cartesian
# route wound the pan to -6.2822 rad and latched CRITICAL_JOINT_LIMIT (20).
PAN_LIMIT = 2.0 * math.pi - 0.35        # rad, stay this far inside the joint limit
PAN_PREFERRED = -math.pi                # 2A's ore work lives around here
SWEEP_SPEED = 0.55                      # rad/s, under the 0.60 rad/s Task 3 limit
# Ore transfers between the conveyor (-y) and the wells (+y) also go by sweep, at
# this radius: inside the hopper's near wall (base x -0.651) when passing behind
# the base, and far enough from the base axis that the pan rate stays low.
# Speed pack run (30 Oct): the Cartesian route took 6.8-15.9 s per transfer.
SWEEP_RADIUS = 0.55
HOPPER_HEADING = math.pi                # the hopper is straight behind the arm
# Ore phases that may be abandoned when the eBot arrives with the package: the
# magnet is still off, so nothing is held. The ore is simply re-selected later.
PREEMPTABLE_PHASES = ('prepare_pick', 'transit_to_pick', 'transit_route_to_pick',
                      'transit_final_to_pick', 'approach_pick')
# Reach envelope for transit waypoints, measured from the shoulder axis point.
# Reach test (29 Oct): tool down at z +0.33, the singularity warning starts 0.885 m
# from (0, 0, 0.1625). The 2A run in a 3A world latched 21 lifting azurite_ore_2
# (picked at radius 0.84) straight up. Transit waypoints now stay inside 0.80 m;
# only the pick itself (low, where the arm reaches further) may be outside.
SHOULDER = (0.0, 0.0, 0.1625)
TRANSIT_REACH = 0.80
# Where the arm turns to if it hides the package from the camera: the -y side.
LOOK_AWAY_HEADING = math.radians(-122.0)
SLOW_TWIST_SPEED = 0.05         # m/s for the last move into the drop pose
JOINT_TOLERANCE = 0.01          # rad


class Task2AManipulation(Node):
    def __init__(self):
        super().__init__(
            'task3a_manipulation',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)],
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.tcp_sub = self.create_subscription(
            PoseStamped, '/tcp_pose_raw', self._tcp_cb, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joints_cb, 20)
        self.status_sub = self.create_subscription(
            Int32, '/arm_status', self._status_cb, 10)
        self.force_sub = self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self._force_cb, 10)

        self.switch_client = self.create_client(
            SwitchController, f'{SERVO_NS}/switch_controller')
        self.magnet_client = self.create_client(SetBool, '/magnet')

        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Bool, AT_ARM_TOPIC, self._at_arm_cb, latched_qos)
        self.cleared_pub = self.create_publisher(Bool, CLEARED_TOPIC, latched_qos)

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.force_n = None
        self.force_time = None
        self.force_log_time = 0.0
        self.switch_future = None
        self.service_future = None
        self.phase_started = self._now_seconds()
        self.phase = 'wait_ready'
        self.active_controller = None
        self.target_name = None
        self.ore_position = None
        self.pick_surface_z = None
        self.pick_target_locked = False
        self.capture_reference_position = None
        self.pick_z_adjustment = PICK_SEARCH_OFFSETS[0]
        self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[0]
        self.pick_position = None
        self.pick_transit_position = None
        self.pick_route_position = None
        self.pick_side_position = None
        self.drop_position = None
        self.transit_position = None
        self.drop_route_position = None
        self.drop_side_position = None
        self.pick_attempts = 0
        self.pick_xy_attempts = 0
        self.tried_pick_heights = set()
        self.capture_contact_force = 0.0
        self.skip_pick_after_release = False
        self.magnet_enabled = False
        self.sorted_ores = set()
        self.failed_ores = set()
        self.fault_logged = False
        # Task 3A package job
        self.tool_axis_target = (0.0, 0.0, -1.0)
        self.switch_target = None
        self.ebot_at_arm = False
        self.package_done = False
        self.package_cleared_sent = False
        self.package_target = None      # locked top-face centre of the package
        self.package_tries = 0
        self.package_wait_logged = False
        self.package_look_logged = False
        self.package_find_started = 0.0
        self.package_looked_away = False
        self.sweep_joints = None
        self.sweep_next = None
        self.staging_joints = None      # joint angles recorded at HOPPER_STAGING
        self.package_phase_after_switch = None
        self.timer = self.create_timer(0.05, self._step)
        self.get_logger().info(
            'Task 3A manipulation ready; waiting for arm feedback and ore transforms.')

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _tcp_cb(self, msg):
        self.tcp_pose = msg.pose

    def _joints_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in JOINT_NAMES):
            self.joint_angles = [positions[name] for name in JOINT_NAMES]

    def _status_cb(self, msg):
        self.arm_status = int(msg.data)

    def _force_cb(self, msg):
        self.force_n = abs(float(msg.data))
        self.force_time = self._now_seconds()

    def _position(self):
        p = self.tcp_pose.position
        return (float(p.x), float(p.y), float(p.z))

    def _tool_alignment(self):
        # tcp_pose_raw is the TCP pose, not the flange. TCP +Z is the magnet's
        # outward normal (flange +X); align it with tool_axis_target, which is
        # base_link -Z (straight down) except at the hopper. Yaw is free.
        q = self.tcp_pose.orientation
        axis = (2.0 * (q.x * q.z + q.w * q.y),
                2.0 * (q.y * q.z - q.w * q.x),
                1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        d = self.tool_axis_target
        # The angular velocity a x d turns the tool axis a towards d.
        cross = (axis[1] * d[2] - axis[2] * d[1],
                 axis[2] * d[0] - axis[0] * d[2],
                 axis[0] * d[1] - axis[1] * d[0])
        magnitude = math.sqrt(sum(v * v for v in cross))
        angle = math.atan2(magnitude, sum(a * b for a, b in zip(axis, d)))
        if magnitude < 1e-9:
            return angle, (0.0, 0.0, 0.0)
        speed = min(ANGULAR_SPEED_LIMIT, ORIENTATION_GAIN * angle)
        return angle, tuple(speed * v / magnitude for v in cross)

    def _publish_twist(self, velocity, angular=(0.0, 0.0, 0.0)):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = velocity
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = angular
        self.twist_pub.publish(msg)

    def _publish_joints(self, velocities):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(JOINT_NAMES)
        msg.velocities = [float(value) for value in velocities]
        self.joint_pub.publish(msg)

    def _stop(self):
        if self.active_controller == TWIST_CONTROLLER:
            self._publish_twist((0.0, 0.0, 0.0))
        elif self.active_controller == JOINT_CONTROLLER:
            self._publish_joints((0.0,) * 6)

    def _move_toward(self, target, tolerance=POSITION_TOLERANCE,
                     max_speed=TWIST_SPEED_LIMIT):
        current = self._position()
        error = [float(t) - c for t, c in zip(target, current)]
        distance = math.sqrt(sum(value * value for value in error))
        tilt, angular = self._tool_alignment()
        if distance <= tolerance and tilt <= ORIENTATION_TOLERANCE:
            self._publish_twist((0.0, 0.0, 0.0))
            return True
        velocity = [POSITION_GAIN * value for value in error]
        # Correct tilt above the surface before a close approach.
        if tilt > math.radians(3.0):
            velocity = [0.0, 0.0, 0.0]
        speed = math.sqrt(sum(value * value for value in velocity))
        if speed > max_speed:
            scale = max_speed / speed
            velocity = [value * scale for value in velocity]
        self._publish_twist(velocity, angular)
        return False

    @staticmethod
    def _reach_safe(point):
        """Pull a transit waypoint towards the base axis until it is inside TRANSIT_REACH."""
        x, y, z = point
        dz = z - SHOULDER[2]
        radius = math.hypot(x, y)
        allowed = math.sqrt(max(0.0, TRANSIT_REACH ** 2 - dz * dz))
        if radius <= allowed or radius < 1e-6:
            return (x, y, z)
        scale = max(allowed, 0.30) / radius
        return (x * scale, y * scale, z)

    def _clamp_transit_waypoints(self):
        self.pick_transit_position = self._reach_safe(self.pick_transit_position)
        self.pick_route_position = self._reach_safe(self.pick_route_position)
        self.pick_side_position = self._reach_safe(self.pick_side_position)
        self.transit_position = self._reach_safe(self.transit_position)
        self.drop_route_position = self._reach_safe(self.drop_route_position)
        self.drop_side_position = self._reach_safe(self.drop_side_position)

    def _change_phase(self, phase):
        now = self._now_seconds()
        if phase != self.phase:
            # Sim-time trace of every phase, to see where the run's time goes.
            self.get_logger().info(
                f'[t={now:7.1f} s] {self.phase} -> {phase} '
                f'(spent {now - self.phase_started:5.1f} s)')
        self.phase = phase
        self.phase_started = now

    def _switch(self, activate, deactivate, next_phase):
        if not self.switch_client.service_is_ready():
            return False
        request = SwitchController.Request()
        request.activate_controllers = [activate]
        request.deactivate_controllers = [deactivate]
        request.strictness = SwitchController.Request.STRICT
        self.switch_future = self.switch_client.call_async(request)
        self.next_phase = next_phase
        self.switch_target = activate
        self._change_phase('wait_switch')
        self.get_logger().info(f'Switching arm control to {activate}.')
        return True

    def _switch_result(self):
        if self.switch_future is None or not self.switch_future.done():
            return None
        future = self.switch_future
        self.switch_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'Controller switch failed: {exc}')
            return False
        return bool(response and response.ok)

    def _fresh_ore_transforms(self):
        now_ns = self.get_clock().now().nanoseconds
        fresh = []
        for kind in ORE_KINDS:
            # Four objects total; allow additional IDs in case a color is duplicated.
            for ore_id in range(1, EXPECTED_ORE_COUNT + 3):
                frame = f'{kind}_{ore_id}'
                try:
                    tf_msg = self.tf_buffer.lookup_transform(
                        BASE_FRAME, frame, rclpy.time.Time())
                except Exception:
                    continue
                stamp_ns = (tf_msg.header.stamp.sec * 1_000_000_000
                            + tf_msg.header.stamp.nanosec)
                age = (now_ns - stamp_ns) / 1e9
                if age < -0.05 or age > TF_MAX_AGE_S:
                    continue
                p = tf_msg.transform.translation
                fresh.append((frame, kind, (p.x, p.y, p.z)))
        return fresh

    def _drop_target(self, kind):
        # The collision mesh places the well centers near local x=-0.27, 0.04,
        # and 0.33 m. The left well's reachable point is shifted along local z.
        x = CARGO_ROVER_WORLD[0] + WELL_LOCAL_X[kind] - ARM_BASE_WORLD[0]
        y = (CARGO_ROVER_WORLD[1] - ARM_BASE_WORLD[1]
             - WELL_LOCAL_Z[kind])
        return (x, y, DROP_HEIGHT_BASE)

    @staticmethod
    def _side_x(y):
        """Choose a reachable lateral waypoint that stays clear of the base axis."""
        # Keep radial clearance from the base axis while respecting the arm reach.
        max_x = math.sqrt(max(0.0, 0.80 * 0.80 - y * y))
        return -min(0.36, max(0.20, max_x))

    def _start_next_ore(self):
        candidates = [item for item in self._fresh_ore_transforms()
                      if item[0] not in self.sorted_ores | self.failed_ores
                      # Conveyor is on the negative-y side of base_link; the rover
                      # wells are on the positive-y side. Ignore ores already dropped.
                      and item[2][1] < -0.25]
        if not candidates:
            if len(self.sorted_ores | self.failed_ores) >= EXPECTED_ORE_COUNT:
                if not self.package_done:
                    # Ores finished first: wait for the eBot with the package.
                    if not self.package_wait_logged:
                        self.get_logger().info(
                            'Ores finished; waiting for the eBot to bring the package.')
                        self.package_wait_logged = True
                    self._publish_twist((0.0, 0.0, 0.0))
                    return
                self._change_phase('done')
                if self.failed_ores:
                    self.get_logger().error(
                        f'Incomplete run: {len(self.sorted_ores)}/4 released; '
                        f'failed pickups: {sorted(self.failed_ores)}.')
                else:
                    self.get_logger().info('All four ores have been picked and released in their containers.')
            return
        current = self._position()
        frame, kind, position = min(
            candidates,
            key=lambda item: math.dist(current, item[2]),
        )
        self.target_name = frame
        self.ore_position = tuple(float(v) for v in position)
        self.pick_surface_z = self.ore_position[2]
        self.pick_target_locked = False
        # The ore TF is on the top face. The spring-mounted magnet needs a small
        # stand-off so its face contacts the surface without driving into it.
        self.pick_z_adjustment = PICK_SEARCH_OFFSETS[0]
        self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[0]
        self.pick_attempts = 0
        self.pick_xy_attempts = 0
        self.tried_pick_heights = {round(self.pick_z_adjustment, 4)}
        self.skip_pick_after_release = False
        self.capture_reference_position = None
        pick_x, pick_y = self.ore_position[:2]
        pick_x += self.pick_xy_adjustment[0]
        pick_y += self.pick_xy_adjustment[1]
        horizontal_range = math.hypot(pick_x, pick_y)
        # Keep the TCP at the camera-measured ore center. The old 0.82 m clamp
        # moved outer conveyor ores inward by up to 4 cm, outside the seal window.
        safe_radius = 0.90
        if horizontal_range > safe_radius:
            scale = safe_radius / horizontal_range
            pick_x *= scale
            pick_y *= scale
        # The camera reports the ore top and TCP lies at the uncompressed face.
        self.pick_position = (
            pick_x, pick_y, self.ore_position[2] + self.pick_z_adjustment)
        pick_transit_z = max(
            self._position()[2], self.pick_position[2] + 0.13, CARRY_CLEARANCE)
        self.pick_transit_position = (
            self.pick_position[0], self.pick_position[1], pick_transit_z)
        # Route around the base axis: a straight rover-to-conveyor transfer crosses
        # the UR wrist singularity and can trigger a protective stop.
        side_x = self._side_x(pick_y)
        self.pick_route_position = (side_x, current[1], pick_transit_z)
        self.pick_side_position = (side_x, pick_y, pick_transit_z)
        self.drop_position = self._drop_target(kind)
        self.transit_position = (
            self.drop_position[0], self.drop_position[1],
            max(self.pick_position[2] + CARRY_CLEARANCE,
                self.drop_position[2] + CARRY_CLEARANCE))
        drop_side_x = self._side_x(self.drop_position[1])
        self.drop_route_position = (
            drop_side_x, pick_y, self.transit_position[2])
        self.drop_side_position = (
            drop_side_x, self.drop_position[1], self.transit_position[2])
        self._clamp_transit_waypoints()
        self.pick_attempts = 0
        self.get_logger().info(
            f'Picking {frame} at ({self.ore_position[0]:.3f}, '
            f'{self.ore_position[1]:.3f}, {self.ore_position[2]:.3f}); '
            f'dropping at ({self.drop_position[0]:.3f}, '
            f'{self.drop_position[1]:.3f}, {self.drop_position[2]:.3f}).')
        self._change_phase('prepare_pick')

    def _target_transform_position(self):
        if not self.target_name:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, self.target_name, rclpy.time.Time())
        except Exception:
            return None
        stamp = rclpy.time.Time.from_msg(transform.header.stamp).nanoseconds
        age = (self.get_clock().now().nanoseconds - stamp) / 1e9
        if age < -0.05 or age > TF_MAX_AGE_S:
            return None
        t = transform.transform.translation
        return (float(t.x), float(t.y), float(t.z))

    def _refresh_pick_target(self):
        """Update from visible camera detections, then retain the pre-occlusion pose."""
        point = self._target_transform_position()
        if (not self.pick_target_locked and point is not None
                and -1.1 < point[0] < 0.35 and -1.1 < point[1] < -0.25):
            # Keep the initial surface height stable across attempts.
            self.ore_position = (point[0], point[1], self.pick_surface_z)
        if self.ore_position is None:
            return
        # Keep applying the search offsets even when the arm briefly occludes the
        # ore and its latest TF expires; use the last camera position in that case.
        pick_x = self.ore_position[0] + self.pick_xy_adjustment[0]
        pick_y = self.ore_position[1] + self.pick_xy_adjustment[1]
        horizontal_range = math.hypot(pick_x, pick_y)
        if horizontal_range > 0.90:
            scale = 0.90 / horizontal_range
            pick_x *= scale
            pick_y *= scale
        self.pick_position = (
            pick_x, pick_y, self.pick_surface_z + self.pick_z_adjustment)
        transit_z = max(self._position()[2], self.pick_position[2] + 0.13,
                        CARRY_CLEARANCE)
        self.pick_transit_position = (pick_x, pick_y, transit_z)
        side_x = self._side_x(pick_y)
        self.pick_route_position = (side_x, self._position()[1], transit_z)
        self.pick_side_position = (side_x, pick_y, transit_z)
        self.transit_position = (
            self.drop_position[0], self.drop_position[1],
            max(self.pick_position[2] + CARRY_CLEARANCE,
                self.drop_position[2] + CARRY_CLEARANCE))
        drop_side_x = self._side_x(self.drop_position[1])
        self.drop_route_position = (
            drop_side_x, pick_y, self.transit_position[2])
        self.drop_side_position = (
            drop_side_x, self.drop_position[1], self.transit_position[2])
        self._clamp_transit_waypoints()

    def _begin_capture_check(self):
        self.capture_reference_position = (
            self._target_transform_position() or self.ore_position)
        self.capture_contact_force = self.force_n or 0.0
        p = self._position()
        self.get_logger().info(
            f'Checking whether {self.target_name} follows the magnet; '
            f'offset={self.pick_z_adjustment:+.3f} m, force={self.force_n}, '
            f'tilt={math.degrees(self._tool_alignment()[0]):.2f} deg, '
            f'tool=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), '
            f'target=({self.pick_position[0]:.3f}, {self.pick_position[1]:.3f}, '
            f'{self.pick_position[2]:.3f}).')
        self._change_phase('verify_capture_lift')

    def _call_magnet(self, enabled, wait_phase):
        if not self.magnet_client.service_is_ready():
            return False
        request = SetBool.Request()
        request.data = bool(enabled)
        self.service_future = self.magnet_client.call_async(request)
        self.service_enabled = bool(enabled)
        self._change_phase(wait_phase)
        return True

    def _service_result(self):
        if self.service_future is None or not self.service_future.done():
            return None
        future = self.service_future
        self.service_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'Magnet service call failed: {exc}')
            return False
        if response is None or not response.success:
            self.get_logger().warning(
                f'Magnet request failed: {getattr(response, "message", "no response")}')
            return False
        self.magnet_enabled = self.service_enabled
        return True

    def _held(self):
        return (self.force_n is not None and self.force_time is not None
                and self._now_seconds() - self.force_time < 0.5
                and self.force_n >= FORCE_HELD_THRESHOLD_N)

    def _fault(self, message):
        self._stop()
        if self.phase != 'fault':
            if self.tcp_pose is not None:
                p = self._position()
                self.get_logger().error(
                    f'{message} Phase={self.phase}, '
                    f'tool=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}).')
            else:
                self.get_logger().error(message)
        self._change_phase('fault')

    # ---- Task 3A: the ore package ---------------------------------------------------

    def _at_arm_cb(self, msg):
        if msg.data and not self.ebot_at_arm:
            self.get_logger().info('eBot reports it is parked beside the arm.')
        self.ebot_at_arm = bool(msg.data)

    def _publish_cleared(self):
        if self.package_cleared_sent:
            return
        msg = Bool()
        msg.data = True
        self.cleared_pub.publish(msg)
        self.package_cleared_sent = True
        self.get_logger().info('Package is off the deck; told the eBot it may leave.')

    def _package_transform(self):
        """Fresh ore_package position (top-face centre) in base_link, or None."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME, PACKAGE_FRAME, rclpy.time.Time())
        except Exception:
            return None
        stamp = rclpy.time.Time.from_msg(tf_msg.header.stamp).nanoseconds
        age = (self.get_clock().now().nanoseconds - stamp) / 1e9
        if age < -0.05 or age > TF_MAX_AGE_S:
            return None
        t = tf_msg.transform.translation
        point = (float(t.x), float(t.y), float(t.z))
        (x0, x1), (y0, y1), (z0, z1) = PACKAGE_ROI
        if not (x0 < point[0] < x1 and y0 < point[1] < y1 and z0 < point[2] < z1):
            return None
        return point

    def _joint_move_toward(self, target, max_speed=JOINT_SPEED_LIMIT):
        """One tick of a straight joint-space move. True when every joint is there."""
        errors = [t - c for t, c in zip(target, self.joint_angles)]
        largest = max(abs(value) for value in errors)
        if largest < JOINT_TOLERANCE:
            self._publish_joints((0.0,) * 6)
            return True
        speed = min(max_speed, 1.6 * largest)
        self._publish_joints([(error / largest) * speed for error in errors])
        return False

    def _begin_sweep(self, heading, next_phase):
        """Turn shoulder_pan alone so the TCP faces `heading` (rad, base_link).

        Rotating joint 1 by d rotates the whole arm about the base z axis, so the
        TCP heading changes by exactly d. Of the pan angles that give the wanted
        heading, take the one inside the joint limit nearest the current pan
        (ties go to the one nearer PAN_PREFERRED).
        """
        p = self._position()
        current_heading = math.atan2(p[1], p[0])
        pan = self.joint_angles[0]
        delta = math.atan2(math.sin(heading - current_heading),
                           math.cos(heading - current_heading))
        candidates = [pan + delta + k * 2.0 * math.pi for k in (-1, 0, 1)]
        candidates = [c for c in candidates if abs(c) <= PAN_LIMIT]
        target_pan = min(candidates, key=lambda c: (
            abs(c - pan) + 0.01 * abs(c - PAN_PREFERRED)))
        self.sweep_joints = [target_pan] + list(self.joint_angles[1:])
        self.sweep_next = next_phase
        self.get_logger().info(
            f'Sweeping shoulder_pan {pan:+.2f} -> {target_pan:+.2f} rad '
            f'(heading {math.degrees(heading):+.0f} deg).')
        self._change_phase('pkg_sweep_start')

    def _on_sweep_circle(self):
        """The point at SWEEP_RADIUS on the TCP's current heading, at its current height."""
        p = self._position()
        heading = math.atan2(p[1], p[0])
        return (SWEEP_RADIUS * math.cos(heading), SWEEP_RADIUS * math.sin(heading), p[2])

    def _start_package(self):
        self.package_target = None
        self.package_tries = 0
        self.package_find_started = self._now_seconds()
        self.package_looked_away = False
        self.tool_axis_target = (0.0, 0.0, -1.0)
        self.get_logger().info('Starting the package job.')
        self._change_phase('pkg_rise')

    def _package_step(self):
        phase = self.phase
        now = self._now_seconds()

        if phase == 'pkg_rise':
            # Straight up first, clear of the conveyor, the wells and the deck.
            p = self._position()
            if self._move_toward(self._reach_safe((p[0], p[1], max(p[2], PACKAGE_CARRY_Z)))):
                self._change_phase('pkg_find')
            return

        if phase == 'pkg_sweep_start':
            self._publish_twist((0.0, 0.0, 0.0))
            self._switch(JOINT_CONTROLLER, TWIST_CONTROLLER, 'pkg_sweep')
            return

        if phase == 'pkg_sweep':
            if self._joint_move_toward(self.sweep_joints, max_speed=SWEEP_SPEED):
                self._change_phase('pkg_sweep_end')
            return

        if phase == 'pkg_sweep_end':
            self._publish_joints((0.0,) * 6)
            self._switch(TWIST_CONTROLLER, JOINT_CONTROLLER, self.sweep_next)
            return

        if phase == 'pkg_find':
            point = self._package_transform()
            if point is not None:
                self.package_target = point
                self.get_logger().info(
                    f'Package top face at ({point[0]:.3f}, {point[1]:.3f}, '
                    f'{point[2]:.3f}); turning towards it.')
                self._begin_sweep(math.atan2(point[1], point[0]), 'pkg_above')
                return
            self._publish_twist((0.0, 0.0, 0.0))
            waited = now - self.package_find_started
            if waited > 3.0 and not self.package_looked_away:
                # Not visible from here: the arm may be between the camera and the
                # deck. Turn over the -y side, out of the camera's line of sight.
                self.package_looked_away = True
                if not self.package_look_logged:
                    self.get_logger().warning(
                        'No package transform yet; turning the arm out of the camera view.')
                    self.package_look_logged = True
                self._begin_sweep(LOOK_AWAY_HEADING, 'pkg_find')
                return
            if waited > 25.0:
                self._fault('No ore_package transform within 25 s of the eBot arriving.')
            return

        if phase == 'pkg_above':
            # Keep refining the target while the camera still sees the package.
            point = self._package_transform()
            if point is not None:
                self.package_target = point
            px, py, pz = self.package_target
            if self._move_toward((px, py, PACKAGE_CARRY_Z)):
                self._change_phase('pkg_descend')
            return

        if phase == 'pkg_descend':
            # The wrist hides the package from here on: the target stays locked.
            px, py, pz = self.package_target
            if self._move_toward((px, py, pz + PACKAGE_STANDOFF)):
                self._call_magnet(True, 'pkg_wait_magnet_on')
            return

        if phase == 'pkg_wait_magnet_on':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            if not result:
                self._fault('Could not enable the magnet for the package.')
                return
            self._change_phase('pkg_contact')
            return

        if phase == 'pkg_contact':
            px, py, pz = self.package_target
            press = PACKAGE_PRESS + 0.004 * self.package_tries
            force_fresh = (self.force_time is not None and now - self.force_time < 0.5)
            reached = self._move_toward(
                (px, py, pz - press), tolerance=PICK_POSITION_TOLERANCE * 4,
                max_speed=SLOW_TWIST_SPEED)
            hard = force_fresh and self.force_n >= FORCE_APPROACH_LIMIT_N
            if reached or hard or now - self.phase_started > 4.0:
                self._change_phase('pkg_test_lift')
            return

        if phase == 'pkg_test_lift':
            px, py, pz = self.package_target
            if self._move_toward((px, py, pz + 0.05)):
                self._change_phase('pkg_test_hold')
            return

        if phase == 'pkg_test_hold':
            self._publish_twist((0.0, 0.0, 0.0))
            if now - self.phase_started < 0.6:
                return
            force_fresh = (self.force_time is not None and now - self.force_time < 0.5)
            if force_fresh and self.force_n >= PACKAGE_HELD_N:
                self.get_logger().info(
                    f'Package held ({self.force_n:.2f} N); lifting it off the deck.')
                self._change_phase('pkg_lift')
                return
            self.package_tries += 1
            self.get_logger().warning(
                f'Package not held (force {self.force_n}); attempt '
                f'{self.package_tries}/{PACKAGE_MAX_TRIES}.')
            if self.package_tries >= PACKAGE_MAX_TRIES:
                self._fault('The package could not be picked.')
                return
            self._call_magnet(False, 'pkg_wait_retry_release')
            return

        if phase == 'pkg_wait_retry_release':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            self._change_phase('pkg_descend')
            return

        if phase == 'pkg_lift':
            px, py, pz = self.package_target
            if self._position()[2] >= pz + PACKAGE_CLEARED_RISE and self._held():
                self._publish_cleared()
            if self._move_toward((px, py, PACKAGE_CARRY_Z)):
                self._publish_cleared()
                self._begin_sweep(HOPPER_HEADING, 'pkg_staging')
            return

        if phase == 'pkg_staging':
            if self._move_toward(HOPPER_STAGING):
                # Tilt the magnet face outward, towards the hopper, in place.
                self.tool_axis_target = (
                    -math.sin(HOPPER_TILT), 0.0, -math.cos(HOPPER_TILT))
                self._change_phase('pkg_tilt')
            return

        if phase == 'pkg_tilt':
            if self._move_toward(HOPPER_STAGING):
                # Remember this configuration: the way out is a joint move back to it.
                self.staging_joints = list(self.joint_angles)
                self._change_phase('pkg_to_drop')
            return

        if phase == 'pkg_to_drop':
            if self.arm_status is not None and 10 <= self.arm_status <= 19:
                self.get_logger().warning(
                    f'Arm warning {self.arm_status} on the way to the hopper; '
                    'releasing here.')
                self._call_magnet(False, 'pkg_wait_release')
                return
            if self._move_toward(HOPPER_DROP, max_speed=SLOW_TWIST_SPEED):
                self._call_magnet(False, 'pkg_wait_release')
            return

        if phase == 'pkg_wait_release':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            if not result:
                self._fault('Could not turn off the magnet over the hopper.')
                return
            self._change_phase('pkg_verify_release')
            return

        if phase == 'pkg_verify_release':
            self._publish_twist((0.0, 0.0, 0.0))
            force_fresh = (self.force_time is not None and now - self.force_time < 0.5)
            if force_fresh and self.force_n < FORCE_HELD_THRESHOLD_N:
                self.get_logger().info('Package released over the crusher hopper.')
                self._switch(JOINT_CONTROLLER, TWIST_CONTROLLER, 'pkg_exit')
                return
            if now - self.phase_started > 1.5:
                self._fault('Payload force did not clear after releasing the package.')
            return

        if phase == 'pkg_exit':
            # Leave the hopper in joint space. A straight Cartesian move back from
            # near full stretch is the direction the arm can barely produce
            # (the reach test latched 21 doing exactly that).
            if self._joint_move_toward(self.staging_joints):
                self.tool_axis_target = (0.0, 0.0, -1.0)
                self.package_done = True
                self.get_logger().info('Package job done.')
                self._switch(TWIST_CONTROLLER, JOINT_CONTROLLER, 'select')
            return

        self._fault(f'Unknown package phase: {phase}')

    def _step(self):
        if self.arm_status is not None and 20 <= self.arm_status <= 23:
            self._fault(
                f'Protective stop reported on /arm_status: {self.arm_status}.')
            return

        if self.phase == 'fault':
            self._stop()
            return
        if self.phase == 'done':
            if self.active_controller == TWIST_CONTROLLER:
                self._publish_twist((0.0, 0.0, 0.0))
            return

        if (self.ebot_at_arm and not self.package_done
                and self.phase in PREEMPTABLE_PHASES):
            # The eBot waits idle until the package is off its deck; serve it now.
            self.get_logger().info(
                f'eBot is waiting: leaving {self.target_name} for later.')
            self.target_name = None
            self.pick_target_locked = False
            self._start_package()
            return

        if self.phase == 'wait_ready':
            if (self.tcp_pose is None or self.joint_angles is None
                    or self.arm_status is None):
                return
            if self.switch_client.service_is_ready():
                self._switch(JOINT_CONTROLLER, TWIST_CONTROLLER, 'unfold')
            return

        if self.phase == 'wait_switch':
            result = self._switch_result()
            if result is None:
                return
            if not result:
                self._fault('The arm controller switch was rejected.')
                return
            self.active_controller = self.switch_target
            self._change_phase(self.next_phase)
            return

        if self.phase == 'unfold':
            errors = [target - current for target, current
                      in zip(UNFOLD_POSE, self.joint_angles)]
            largest = max(abs(value) for value in errors)
            if largest < 0.015:
                self._publish_joints((0.0,) * 6)
                self._change_phase('settle')
                return
            speed = min(JOINT_SPEED_LIMIT, 1.6 * largest)
            self._publish_joints([
                (error / largest) * speed for error in errors])
            return

        if self.phase == 'settle':
            self._publish_joints((0.0,) * 6)
            if self._now_seconds() - self.phase_started >= 0.6:
                if not self._switch(TWIST_CONTROLLER, JOINT_CONTROLLER, 'select'):
                    self._publish_twist((0.0, 0.0, 0.0))
            return

        if self.phase == 'select':
            # The package goes first as soon as the eBot is parked with it, so the
            # eBot's drive home overlaps with the remaining ore sorting.
            if self.ebot_at_arm and not self.package_done:
                self._start_package()
                return
            self._start_next_ore()
            return

        if self.phase.startswith('pkg_'):
            self._package_step()
            return

        if self.phase == 'prepare_pick':
            self._refresh_pick_target()
            current = self._position()
            safe_z = max(current[2], self.pick_transit_position[2])
            if self._move_toward(self._reach_safe((current[0], current[1], safe_z)), tolerance=VIA_TOLERANCE):
                self._change_phase('transit_to_pick')
            return

        if self.phase == 'transit_to_pick':
            # Radially to SWEEP_RADIUS on the current heading, then sweep the pan
            # round to the ore's heading.
            if self._move_toward(self._on_sweep_circle(), tolerance=VIA_TOLERANCE):
                self._refresh_pick_target()
                self._begin_sweep(
                    math.atan2(self.pick_position[1], self.pick_position[0]),
                    'transit_final_to_pick')
            return

        if self.phase == 'transit_route_to_pick':
            self._refresh_pick_target()
            if self._move_toward(self.pick_side_position, tolerance=VIA_TOLERANCE):
                self._change_phase('transit_final_to_pick')
            return

        if self.phase == 'transit_final_to_pick':
            self._refresh_pick_target()
            if self._move_toward(self.pick_transit_position, tolerance=VIA_TOLERANCE):
                # The wrist hides the marker during descent. Its last complete
                # camera observation remains the pickup target until we lift away.
                self.pick_target_locked = True
                self._change_phase('approach_pick')
            return

        if self.phase == 'approach_pick':
            self._refresh_pick_target()
            target = (self.pick_position[0], self.pick_position[1],
                      self.pick_position[2] + PICK_APPROACH_STANDOFF)
            if self._move_toward(target):
                self._change_phase('enable_magnet')
            return

        if self.phase == 'move_to_pick':
            self._refresh_pick_target()
            force_fresh = (self.force_time is not None
                           and self._now_seconds() - self.force_time < 0.5)
            if force_fresh and self.force_n >= FORCE_APPROACH_LIMIT_N:
                self.get_logger().warning(
                    f'{self.target_name} approach reached '
                    f'{self.force_n:.1f} N; checking the pickup before descending further.')
                self._begin_capture_check()
                return
            if self._move_toward(
                    self.pick_position, tolerance=PICK_POSITION_TOLERANCE):
                self._change_phase('verify_pick')
            return

        if self.phase == 'enable_magnet':
            self._publish_twist((0.0, 0.0, 0.0))
            if not self._call_magnet(True, 'wait_magnet_on'):
                return
            return

        if self.phase == 'wait_magnet_on':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            if not result:
                self._fault('Could not enable the magnet.')
                return
            self._change_phase('move_to_pick')
            return

        if self.phase == 'verify_pick':
            # Hold the pre-occlusion target while checking for payload force.
            self._refresh_pick_target()
            aligned = self._move_toward(
                self.pick_position, tolerance=PICK_POSITION_TOLERANCE)
            if self._now_seconds() - self.force_log_time >= 0.5:
                self.force_log_time = self._now_seconds()
                pose = self._position()
                self.get_logger().info(
                    f'Pickup check {self.target_name}: force={self.force_n}, '
                    f'tool=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}), '
                    f'target=({self.pick_position[0]:.3f}, '
                    f'{self.pick_position[1]:.3f}, {self.pick_position[2]:.3f}).')
            if self._held():
                self._begin_capture_check()
                return
            elapsed = self._now_seconds() - self.phase_started
            if elapsed > 0.45 and aligned:
                self._begin_capture_check()
            elif elapsed > 1.0:
                self._begin_capture_check()
            return

        if self.phase == 'verify_capture_lift':
            target = (self.pick_position[0], self.pick_position[1],
                      self.pick_position[2] + 0.05)
            if self._move_toward(target):
                self._change_phase('wait_capture_check')
            return

        if self.phase == 'wait_capture_check':
            self._publish_twist((0.0, 0.0, 0.0))
            if self._now_seconds() - self.phase_started < 0.6:
                return
            current_ore = self._target_transform_position()
            reference = self.capture_reference_position
            ore_followed = (current_ore is not None and reference is not None
                            and current_ore[2] - reference[2] >= 0.025)
            force_held_after_lift = self._held()
            if ore_followed or force_held_after_lift:
                self.get_logger().info(
                    f'{self.target_name} pickup confirmed after lift '
                    f'(ore_followed={ore_followed}, force={self.force_n}); carrying it.')
                self.capture_reference_position = None
                self._change_phase('lift')
                return
            if self.pick_xy_attempts < len(PICK_XY_SEARCH_OFFSETS) - 1:
                self.pick_xy_attempts += 1
                self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[
                    self.pick_xy_attempts]
                self.get_logger().warning(
                    f'{self.target_name} made contact without sealing; '
                    f'trying lateral pickup offset '
                    f'({self.pick_xy_adjustment[0]:+.3f}, '
                    f'{self.pick_xy_adjustment[1]:+.3f}) m.')
                self._call_magnet(False, 'wait_retry_release')
                return
            if self.capture_contact_force > FORCE_COLLISION_THRESHOLD_N:
                next_heights = [height for height in PICK_SEARCH_OFFSETS
                                if height > self.pick_z_adjustment + 0.0005
                                and round(height, 4) not in self.tried_pick_heights]
                reason = (f'high contact force '
                          f'({self.capture_contact_force:.1f} N); raising the next approach')
                # A high-force miss can be a side collision or an off-center seal.
                # After exhausting the lateral scan, move to the next height while
                # keeping that scan exhausted so we do not repeat it at every level.
                self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[0]
            else:
                self.pick_xy_attempts = 0
                self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[0]
                next_heights = [height for height in PICK_SEARCH_OFFSETS
                                if round(height, 4) not in self.tried_pick_heights]
                reason = 'no sealed pickup; changing approach height'
            if not next_heights:
                self.get_logger().warning(
                    f'{self.target_name} was not captured after the height scan; '
                    'skipping it and continuing with the remaining ores.')
                self.failed_ores.add(self.target_name)
                self.skip_pick_after_release = True
                self._call_magnet(False, 'wait_retry_release')
                return
            self.pick_z_adjustment = next_heights[0]
            self.tried_pick_heights.add(round(self.pick_z_adjustment, 4))
            self.pick_attempts = len(self.tried_pick_heights) - 1
            if self.capture_contact_force <= FORCE_COLLISION_THRESHOLD_N:
                self.pick_xy_attempts = 0
                self.pick_xy_adjustment = PICK_XY_SEARCH_OFFSETS[0]
            self.get_logger().warning(
                f'{self.target_name} did not follow the tool; {reason}, '
                f'to {self.pick_z_adjustment:+.3f} m.')
            self._call_magnet(False, 'wait_retry_release')
            return

        if self.phase == 'wait_retry_release':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            if not result:
                self._fault('Could not reset the magnet for a pickup retry.')
                return
            if self.skip_pick_after_release:
                self.skip_pick_after_release = False
                self._change_phase('skip_failed_pick')
            else:
                self._change_phase('retry_preposition')
            return

        if self.phase == 'retry_preposition':
            self._refresh_pick_target()
            retry_target = (self.pick_position[0], self.pick_position[1],
                            self.pick_position[2] + PICK_APPROACH_STANDOFF)
            if math.dist(self._position(), retry_target) > 0.16:
                self._change_phase('prepare_pick')
            elif self._move_toward(retry_target):
                self._change_phase('enable_magnet')
            return

        if self.phase == 'skip_failed_pick':
            current = self._position()
            if self._move_toward(
                    self._reach_safe((current[0], current[1], CARRY_CLEARANCE))):
                self.target_name = None
                self._change_phase('select')
            return

        if self.phase == 'lift':
            # Up and inwards together: straight up from a far ore runs into
            # full stretch.
            target = self._reach_safe((self.pick_position[0], self.pick_position[1],
                                       self.transit_position[2]))
            if self._move_toward(target, tolerance=VIA_TOLERANCE):
                self._change_phase('transit_to_drop_route')
            return

        if self.phase == 'transit_to_drop_route':
            if self._move_toward(self._on_sweep_circle(), tolerance=VIA_TOLERANCE):
                self._begin_sweep(
                    math.atan2(self.drop_position[1], self.drop_position[0]),
                    'transit_final_to_drop')
            return

        if self.phase == 'transit_to_drop':
            if self._move_toward(self.drop_side_position, tolerance=VIA_TOLERANCE):
                self._change_phase('transit_final_to_drop')
            return

        if self.phase == 'transit_final_to_drop':
            if self._move_toward(self.transit_position, tolerance=VIA_TOLERANCE):
                self._change_phase('approach_drop')
            return

        if self.phase == 'approach_drop':
            if self._move_toward(self.drop_position):
                self._change_phase('call_magnet_off')
            return

        if self.phase == 'call_magnet_off':
            self._publish_twist((0.0, 0.0, 0.0))
            if not self._call_magnet(False, 'wait_magnet_off'):
                return
            return

        if self.phase == 'wait_magnet_off':
            self._publish_twist((0.0, 0.0, 0.0))
            result = self._service_result()
            if result is None:
                return
            if not result:
                self._fault('Could not turn off the magnet at the container.')
                return
            self._change_phase('verify_drop')
            return

        if self.phase == 'verify_drop':
            self._publish_twist((0.0, 0.0, 0.0))
            force_fresh = (self.force_time is not None
                           and self._now_seconds() - self.force_time < 0.5)
            if force_fresh and self.force_n < FORCE_HELD_THRESHOLD_N:
                self.sorted_ores.add(self.target_name)
                self.get_logger().info(
                    f'{self.target_name} released in its container '
                    f'({len(self.sorted_ores)}/{EXPECTED_ORE_COUNT}).')
                self.target_name = None
                self._change_phase('retreat')
                return
            if self._now_seconds() - self.phase_started > 1.5:
                self._fault(f'Payload force did not clear after releasing {self.target_name}.')
            return

        if self.phase == 'retreat':
            if self._move_toward(self.transit_position, tolerance=VIA_TOLERANCE):
                self._change_phase('select')
            return

        self._fault(f'Unknown manipulation phase: {self.phase}')


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = Task2AManipulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
