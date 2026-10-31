#!/usr/bin/env python3
"""Follow successive Task 3B eBot paths using odometry and publish /cmd_vel."""

# Team ID:          [6131]
# Author List:      [Agrim Chauhan, Saksham Kaushal, Rayhan Khan, Manan Sood]
# Filename:         task3B_navigation.py
# Task:             3B - StrataCobot (eYRC 2026-27)
# Functions:
#                   _wrap_angle, _quaternion_yaw, _clamp, __init__, pathcb, odomcb,
#                   _publish_velocity, _lookahead_target, stop_rover, _final_direction,
#                   _publish_done, process_navigation, main
# Nodes:
#                   Name - [ task3b_navigation ]
#                   Publishing Topics - [ /cmd_vel, /ebot_nav_done ]
#                   Subscribing Topics - [ /ebot_path, /odom, /clock ]

################### IMPORT MODULES #######################

import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


PATH_TOPIC = '/ebot_path'
ODOM_TOPIC = '/odom'
CMD_TOPIC = '/cmd_vel'
MAP_FRAME = 'map'

MAX_LINEAR_MPS = 0.25      # m/s; eBot limit 0.5 (was 0.20 in 2B)
MAX_ANGULAR_RPS = 0.80     # rad/s; eBot limit 1.0 (was 0.65 in 2B)
LOOKAHEAD_DISTANCE_M = 0.15  # keeps the steering loop damped (zeta ~0.5) at 0.25 m/s
DONE_TOPIC = '/ebot_nav_done'   # ours: True once the eBot has stopped on a path's end

# End of a path, 3B version. With noisy /odom (3B run 31 Oct: the reported turn
# rate alone wanders +-0.08 rad/s), "steer at the end point" made the eBot circle
# 0.07-0.14 m from it: the direction to a point that close swings with every
# centimetre of noise and of turning drift. Now, inside END_ZONE_M, the eBot drives
# along the path's final direction until it has reached the end along that line,
# then turns to the final yaw and stops. Being stopped is judged from what this node
# commands, not from the noisy odometry speeds.
END_ZONE_M = 0.35
END_AIM_BEYOND_M = 0.30          # steer at a point this far past the end, on the final line
ALONG_TOLERANCE_M = 0.02         # "reached" once less than this remains along the line
LOCK_MAX_DISTANCE_M = 0.20       # never lock farther than this from the end
FINAL_POSITION_RELEASE_M = 0.30  # unlock (and drive again) only beyond this
FINAL_YAW_TOLERANCE_RAD = 0.04   # enter "aligned"
FINAL_YAW_RELEASE_RAD = 0.08     # leave "aligned" (hysteresis against yaw noise)
YAW_FILTER_ALPHA = 0.2           # low-pass on /odom yaw, per message (~50 Hz)
STABLE_CYCLES = 10               # 0.5 s of commanding zero while aligned = done
TURN_LATCH_START_ERROR_RAD = 2.75
TURN_LATCH_EXIT_ERROR_RAD = 2.45


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _quaternion_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


class EbotNavigation(Node):
    """Track each newly published path and stop at its required final pose."""

    def __init__(self):
        super().__init__(
            'task3b_navigation',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)],
        )

        path_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_sub = self.create_subscription(
            Path, PATH_TOPIC, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odomcb, 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.done_pub = self.create_publisher(Bool, DONE_TOPIC, path_qos)
        self.yaw_filtered = None
        self.yaw_aligned = False

        self.route = None
        self.route_signature = None
        self.current_waypoint = 0
        self.odom = None
        self.route_complete = False
        self.final_position_locked = False
        self.stable_cycles = 0
        self.turn_target_index = None
        self.turn_direction = 0

        self.create_timer(0.05, self.process_navigation)
        self.get_logger().info('Task 3B navigation ready; waiting for /ebot_path.')

    def pathcb(self, data):
        if data.header.frame_id != MAP_FRAME:
            self.get_logger().warning(
                f'Ignoring path in unexpected frame {data.header.frame_id!r}.')
            return
        if not data.poses:
            self.get_logger().warning('Ignoring an empty /ebot_path message.')
            return

        route = []
        for stamped_pose in data.poses:
            pose = stamped_pose.pose
            route.append((
                float(pose.position.x),
                float(pose.position.y),
                _quaternion_yaw(pose.orientation),
            ))
        signature = tuple(
            (round(x, 4), round(y, 4), round(yaw, 4))
            for x, y, yaw in route
        )
        if signature == self.route_signature:
            return

        self.route = route
        self.route_signature = signature
        self.current_waypoint = 0
        if self.odom is not None:
            x, y = self.odom[:2]
            nearest_waypoint = min(
                range(len(route)),
                key=lambda index: math.hypot(route[index][0] - x, route[index][1] - y),
            )
            self.current_waypoint = min(nearest_waypoint, len(route) - 2)
        self.route_complete = False
        self.final_position_locked = False
        self.stable_cycles = 0
        self.yaw_aligned = False
        self.turn_target_index = None
        self.turn_direction = 0
        self._publish_done(False)
        self.get_logger().info(
            f'Received new path with {len(route)} poses in {data.header.frame_id}; '
            f'starting at waypoint {self.current_waypoint + 1}.')

    def odomcb(self, data):
        position = data.pose.pose.position
        twist = data.twist.twist
        yaw = _quaternion_yaw(data.pose.pose.orientation)
        if self.yaw_filtered is None:
            self.yaw_filtered = yaw
        else:
            self.yaw_filtered = _wrap_angle(
                self.yaw_filtered + YAW_FILTER_ALPHA * _wrap_angle(yaw - self.yaw_filtered))
        self.odom = (
            float(position.x),
            float(position.y),
            yaw,
            math.hypot(twist.linear.x, twist.linear.y),
            float(twist.angular.z),
        )

    def _publish_done(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.done_pub.publish(msg)

    def _final_direction(self):
        """Unit vector of the path's last non-degenerate segment."""
        ex, ey, _ = self.route[-1]
        for index in range(len(self.route) - 2, -1, -1):
            px, py, _ = self.route[index]
            length = math.hypot(ex - px, ey - py)
            if length > 1e-6:
                return (ex - px) / length, (ey - py) / length
        _, _, end_yaw = self.route[-1]
        return math.cos(end_yaw), math.sin(end_yaw)

    def _publish_velocity(self, linear, angular):
        command = Twist()
        command.linear.x = _clamp(linear, -MAX_LINEAR_MPS, MAX_LINEAR_MPS)
        command.angular.z = _clamp(angular, -MAX_ANGULAR_RPS, MAX_ANGULAR_RPS)
        self.cmd_pub.publish(command)

    def _lookahead_target(self, x, y):
        """Return a point ahead on the route, advancing past passed segments."""
        last_index = len(self.route) - 1

        while self.current_waypoint < last_index - 1:
            ax, ay, _ = self.route[self.current_waypoint]
            bx, by, _ = self.route[self.current_waypoint + 1]
            dx = bx - ax
            dy = by - ay
            segment_sq = dx * dx + dy * dy
            if segment_sq <= 1e-12:
                self.current_waypoint += 1
                continue

            projection = ((x - ax) * dx + (y - ay) * dy) / segment_sq
            if projection < 1.0:
                break
            self.current_waypoint += 1

        segment_index = self.current_waypoint
        ax, ay, _ = self.route[segment_index]
        bx, by, _ = self.route[segment_index + 1]
        dx = bx - ax
        dy = by - ay
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-9:
            return bx, by

        projection = _clamp(((x - ax) * dx + (y - ay) * dy)
                            / (segment_length * segment_length), 0.0, 1.0)
        start_x = ax + projection * dx
        start_y = ay + projection * dy
        remaining = LOOKAHEAD_DISTANCE_M

        for index in range(segment_index, last_index):
            end_x, end_y, _ = self.route[index + 1]
            if index == segment_index:
                leg_x, leg_y = start_x, start_y
            else:
                leg_x, leg_y, _ = self.route[index]
            leg_length = math.hypot(end_x - leg_x, end_y - leg_y)
            if leg_length <= 1e-9:
                continue
            if remaining <= leg_length:
                ratio = remaining / leg_length
                return (
                    leg_x + ratio * (end_x - leg_x),
                    leg_y + ratio * (end_y - leg_y),
                )
            remaining -= leg_length

        return self.route[-1][0], self.route[-1][1]

    def stop_rover(self):
        self._publish_velocity(0.0, 0.0)

    def process_navigation(self):
        if self.route is None or self.odom is None:
            self.stop_rover()
            return
        if self.route_complete:
            self.stop_rover()
            return

        x, y, yaw, linear_speed, angular_speed = self.odom
        last_index = len(self.route) - 1
        end_x, end_y, end_yaw = self.route[-1]
        end_distance = math.hypot(end_x - x, end_y - y)

        if self.final_position_locked and end_distance > FINAL_POSITION_RELEASE_M:
            self.final_position_locked = False
            self.get_logger().warning(
                f'Drifted {end_distance:.2f} m from the path end; driving back.')

        if self.final_position_locked:
            yaw_error = _wrap_angle(end_yaw - self.yaw_filtered)
            if abs(yaw_error) <= FINAL_YAW_TOLERANCE_RAD:
                self.yaw_aligned = True
            elif abs(yaw_error) > FINAL_YAW_RELEASE_RAD:
                self.yaw_aligned = False
            if self.yaw_aligned:
                self.stop_rover()
                self.stable_cycles += 1
            else:
                self.stable_cycles = 0
                self._publish_velocity(0.0, 1.2 * yaw_error)
            if self.stable_cycles >= STABLE_CYCLES:
                self.route_complete = True
                self._publish_done(True)
                self.get_logger().info(
                    f'Path complete; stopped at ({x:.3f}, {y:.3f}), '
                    f'yaw {self.yaw_filtered:.3f} (odom).')
            return

        if end_distance <= END_ZONE_M:
            ux, uy = self._final_direction()
            along = (end_x - x) * ux + (end_y - y) * uy
            if along <= ALONG_TOLERANCE_M and end_distance <= LOCK_MAX_DISTANCE_M:
                self.final_position_locked = True
                self.stable_cycles = 0
                self.yaw_aligned = False
                self.stop_rover()
                return
            if along > ALONG_TOLERANCE_M:
                # Drive along the final line, steering at a point beyond the end so
                # the heading stays steady however close the end gets.
                aim_x = end_x + END_AIM_BEYOND_M * ux
                aim_y = end_y + END_AIM_BEYOND_M * uy
                heading_error = _wrap_angle(math.atan2(aim_y - y, aim_x - x) - yaw)
                angular = 1.8 * heading_error
                if abs(heading_error) > 0.65:
                    linear = 0.0
                else:
                    linear = _clamp(0.75 * along, 0.03, MAX_LINEAR_MPS)
                    linear *= max(0.0, math.cos(heading_error))
                self._publish_velocity(linear, angular)
                return
            # Past the end along the line but too far off to the side: fall through
            # to ordinary path following, which will bring it round.

        target_x, target_y = self._lookahead_target(x, y)
        dx = target_x - x
        dy = target_y - y

        if self.current_waypoint != self.turn_target_index:
            self.turn_target_index = self.current_waypoint
            self.turn_direction = 0

        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - yaw)
        angular = 1.8 * heading_error

        # At a 180-degree reversal, the wrapped error jumps between +pi and
        # -pi. Hold one turn direction until the eBot has clearly moved away
        # from that boundary; otherwise small odometry changes can make it
        # alternate directions forever at a waypoint.
        if abs(heading_error) >= TURN_LATCH_START_ERROR_RAD:
            if self.turn_direction == 0:
                self.turn_direction = 1 if heading_error > 0.0 else -1
            angular = self.turn_direction * MAX_ANGULAR_RPS
            linear = 0.0
            self._publish_velocity(linear, angular)
            return

        if self.turn_direction != 0:
            if abs(heading_error) > TURN_LATCH_EXIT_ERROR_RAD:
                angular = self.turn_direction * MAX_ANGULAR_RPS
                self._publish_velocity(0.0, angular)
                return
            self.turn_direction = 0

        # Rotate before advancing when the target direction differs greatly.
        if abs(heading_error) > 0.65:
            linear = 0.0
        else:
            # Slow down for the end of the path, not for the carrot: the carrot is
            # always ~LOOKAHEAD ahead, so 0.75 * distance capped the 2B eBot at
            # about 0.09-0.13 m/s however high MAX_LINEAR_MPS was set.
            linear = min(MAX_LINEAR_MPS, 0.75 * end_distance)
            linear *= max(0.0, math.cos(heading_error))

        self._publish_velocity(linear, angular)


def main():
    rclpy.init(args=sys.argv)
    node = EbotNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.stop_rover()
            except Exception:
                pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
