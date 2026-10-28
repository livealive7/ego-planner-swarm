"""Single entry point for the whole Isaac Sim <-> ego-planner stack for one
drone: run this file alone, nothing else.

Starts, in order:
  1. world_to_map_tf   -- identity static transform bridging Isaac's own
                            tf tree (rooted at 'map') with the 'world'
                            frame_id used by odom_bridge_node/grid_map --
                            same physical origin, just named differently
                            by different Isaac OmniGraph nodes. Without
                            it, anything stamped 'world' (odom, occupancy
                            grid) can't be viewed relative to anything
                            under Isaac's own 'map' tree (e.g. in
                            Foxglove) since there's no path between them.
  2. odom_bridge_node   -- synthesizes nav_msgs/Odometry from Isaac Sim's
                            separate state/pose + state/twist_inertial,
                            published as /drone_<id>_<odometry_topic>.
  3. cmd_vel_mavlink_bridge (backend:=ardupilot, default) or
     cmd_vel_px4_bridge (backend:=px4) -- forwards cmd_vel (Twist, body
                            frame) to the drone's ArduPilot or PX4 SITL
                            over MAVLink. Pick with the 'backend' launch
                            argument.
  4. depth_restamp_node -- relays Isaac's real depth topic, overwriting
                            header.stamp with this node's own clock.
  5. ego_planner_node   -- the planner + grid_map, wired to the topics
                            above, producing
                            /drone_<id>_grid/grid_map/occupancy_inflate.
  6. traj_server        -- ego_planner's own executable; turns the
                            planner's B-spline into a 100Hz stream of
                            PositionCommand setpoints on /position_cmd.
                            Publishes nothing until a trajectory exists
                            (send a goal on /move_base_simple/goal --
                            flight_type is MANUAL_TARGET).
  7. poscmd_to_cmdvel_node -- converts /position_cmd into the body-frame
                            Twist cmd_vel_mavlink_bridge expects.
  8. foxglove_bridge    -- optional, for live visualization (enable_foxglove).

Without 6 and 7, ego_planner_node computing a trajectory has no effect
on the real vehicle at all -- planning/bspline has no consumer and
nothing ever reaches cmd_vel.

If world_to_map_tf's node starts after something long-running (like
foxglove_bridge) has already finished its own initial DDS discovery on
a machine already running many other ROS2 processes (Isaac Sim,
ArduPilot, ...), that other node can occasionally miss discovering it --
confirmed via `ros2 topic info -v /tf_static` showing only 1 publisher
where 2 were expected. Restarting the ros2 daemon fixed the CLI's own
view of the graph but not an already-running node's; the actual fix was
restarting that node so it re-discovers everyone. Starting everything
together in one launch (as this file does) avoids the issue in the
first place.

Grid map wiring notes (why it's built this way):

Isaac Sim's vehicle-state publishers (state/pose, state/twist) and its
Render/Replicator camera publisher (camera/depth) stamp messages using
two unrelated clocks -- one looked like wall-clock unix epoch seconds,
the other like small sim-elapsed seconds, off by roughly 1.8e9 seconds.
grid_map's message_filters::ApproximateTime sync of grid_map/depth +
grid_map/odom can never match a pair across a gap that large, so
occupancy_inflate stayed permanently empty no matter what was actually
in the camera's view -- confirmed live: depth values decoded to a sane
~2.2m and the drone's pose was well inside the map bounds, yet width
was always 0. odom_bridge_node and depth_restamp_node both re-stamp
their output with their own node's now(), putting grid_map/odom and
grid_map/depth back in the same clock domain (off only by real
processing latency, sub-ms to a few ms) so the synchronizer can actually
pair them.

grid_map/depth is remapped to depth_restamp_node's output (relative,
resolves under the ego_planner_node's own namespace) rather than
directly to /drone<id>/camera/depth.

pose_type is ODOMETRY (2), not POSE_STAMPED: grid_map/pose was tried first
and doesn't work here -- grid_map.cpp subscribes to it with RELIABLE QoS
(rclcpp::QoS(25) default), but Isaac Sim's own /drone<id>/state/pose is
published BEST_EFFORT (same publisher as state/twist), so the two are
QoS-incompatible and depthPoseCallback never fires.
grid_map/odom doesn't have this problem: odom_bridge_node's Odometry
publisher and grid_map's odom_sub_ both default to RELIABLE.

In ODOMETRY mode, grid_map applies a hardcoded camera-to-body rotation
(grid_map.cpp's default cam2body_) before using the pose. That matrix is
the standard ROS optical-frame (X-right/Y-down/Z-forward) -> body-frame
(X-forward/Y-left/Z-up) rotation (REP 103/104), not something specific to
ego-planner's own simulated camera rig, so it should already be correct
for any camera publishing images in the normal ROS convention. If Isaac
Sim's camera has a non-negligible translation offset from the body
origin, that part still isn't modeled; publish it on
/vins_estimator/extrinsic (nav_msgs/Odometry) if it turns out to matter.

cx/cy/fx/fy default to the real intrinsics read from
/drone0/camera/color/camera_info (fx=fy=228.50367736816406, cx=160.0,
cy=120.0, 320x240). There is no separate depth camera_info topic; the
depth image is also 320x240 (confirmed via /drone0/camera/depth), so
reusing the color camera's intrinsics assumes both come from the same
underlying camera prim in Isaac Sim. /drone0/camera/depth's encoding is
32FC1 (meters), which grid_map converts using k_depth_scaling_factor --
1000.0 is correct for that. If you add a second drone with a differently
configured camera, override cx/cy/fx/fy via launch arguments.

320x240 (not 640x480) is deliberate: 12_multi_backend_standalone.py's
camera used to default to 640x480, and running it that way (vs. the
pegasus.multi_backend_example *extension*, whose GUI button spawns the
same drone at 320x240) was measured pinning Isaac Sim's own process at
~19 CPU cores vs. ~10 at 320x240 on this box -- enough to starve
ArduCopter's SITL process down to <1% CPU, which in turn starved its
GPS backend below AP_GPS::is_healthy()'s ~4.65Hz threshold
(libraries/AP_GPS/AP_GPS.cpp) and blocked arming entirely. Lowering the
standalone script's own --cam-res default to match closed most but not
all of that gap; the rest needs the ArduCopter process's own scheduling
priority raised (needs root - chrt/renice), which this launch file has
no way to do for you.

Each ArduPilot SITL instance started via sim_vehicle.py listens on
14550 + 10*instance by default, so mavlink_connection auto-derives from
drone_id unless overridden.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    drone_id_str = perform_substitutions(context, [LaunchConfiguration('drone_id')])
    drone_id = int(drone_id_str)
    namespace = f'drone{drone_id_str}'

    backend = perform_substitutions(context, [LaunchConfiguration('backend')]).strip().lower()
    if backend not in ('ardupilot', 'px4'):
        raise ValueError(f"backend must be 'ardupilot' or 'px4', got {backend!r}")

    odometry_topic = perform_substitutions(
        context, [LaunchConfiguration('odometry_topic')])
    odom_output_topic = f'/drone_{drone_id_str}_{odometry_topic}'

    twist_topic = perform_substitutions(context, [LaunchConfiguration('twist_topic')])

    cmd_vel_topic = perform_substitutions(context, [LaunchConfiguration('cmd_vel_topic')])
    mavlink_connection = perform_substitutions(
        context, [LaunchConfiguration('mavlink_connection')])
    if not mavlink_connection:
        if backend == 'ardupilot':
            # NOT port 14550: Pegasus's own ArduPilotMavlinkBackend hardcodes
            # udpin:127.0.0.1:14550+10*id as its HIL sensor/actuator channel to
            # ArduPilot (see ardupilot_mavlink_backend.py's connection_baseport
            # default). run_ardupilot.sh's own --out is deliberately on 14551
            # instead, as a dedicated companion-computer link.
            #
            # pymavlink's bare 'udp:' prefix defaults to input=True (bind/listen),
            # not connect-out. Against 14550 that used to make this node silently
            # re-bind() Pegasus's own HIL socket (SO_REUSEADDR lets a second bind
            # to the identical address:port succeed with no error), splitting
            # delivery of Pegasus's HIL feed between the two sockets -- confirmed
            # by bind()-ing that combination by hand while Pegasus already held
            # it. That's why arming/hover looked fine (nothing sent on this port
            # yet) but things broke exactly once cmd_vel started flowing after a
            # goal. Explicit 'udpout:' makes this node the connecting client to
            # run_ardupilot.sh's --out=udpin:0.0.0.0:14551 link instead.
            mavlink_connection = f'udpout:127.0.0.1:{14551 + 10 * drone_id}'
        else:
            # PX4 SITL sends its offboard/companion-computer MAVLink stream to
            # UDP 14540 + 10*instance by default (14550+10*instance is the
            # separate GCS link) -- see cmd_vel_px4_bridge.py's own default
            # and PX4MavlinkBackendConfig.connection_baseport (4560, the HIL
            # sim link, is a different port again). 'udpin:' here because PX4
            # connects out to us, we don't connect out to it.
            mavlink_connection = f'udpin:0.0.0.0:{14540 + 10 * drone_id}'

    venv_site_packages = perform_substitutions(
        context, [LaunchConfiguration('venv_site_packages')])
    current_pythonpath = perform_substitutions(
        context, [EnvironmentVariable('PYTHONPATH', default_value='')])
    merged_pythonpath = os.pathsep.join(
        p for p in [current_pythonpath, venv_site_packages] if p)

    # Isaac's own tf tree is rooted at 'map' (drone_base_link, camera, etc.
    # all hang off it), but odom_bridge_node/grid_map publish frame_id
    # 'world'. Bridge the two with an identity static transform -- they're
    # the same physical origin, just named differently by different Isaac
    # OmniGraph nodes.
    use_sim_time = LaunchConfiguration('use_sim_time')

    world_map_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_map_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    odom_node = Node(
        package='isaac_odom_bridge',
        executable='odom_bridge_node',
        name='isaac_odom_bridge',
        namespace=namespace,
        parameters=[{
            'pose_topic': 'state/pose',
            'twist_topic': twist_topic,
            'output_topic': odom_output_topic,
            'world_frame_id': 'world',
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # Both cmd_vel_mavlink_bridge (ArduPilot) and cmd_vel_px4_bridge (PX4)
    # import pymavlink, which only lives in the ArduPilot venv.
    # colcon-generated console_scripts always shebang the system python3,
    # which can't see the venv's site-packages, so inject it directly into
    # the node's process environment instead of requiring everyone to
    # `export PYTHONPATH=...` by hand.
    if backend == 'ardupilot':
        cmd_vel_node = Node(
            package='isaac_odom_bridge',
            executable='cmd_vel_mavlink_bridge',
            name='cmd_vel_mavlink_bridge',
            namespace=namespace,
            parameters=[{
                'connection_string': mavlink_connection,
                'cmd_vel_topic': cmd_vel_topic,
                'use_sim_time': use_sim_time,
            }],
            additional_env={'PYTHONPATH': merged_pythonpath},
            output='screen',
        )
    else:
        cmd_vel_node = Node(
            package='isaac_odom_bridge',
            executable='cmd_vel_px4_bridge',
            name='cmd_vel_px4_bridge',
            namespace=namespace,
            parameters=[{
                'connection_string': mavlink_connection,
                'cmd_vel_topic': cmd_vel_topic,
                'takeoff_height': float(perform_substitutions(
                    context, [LaunchConfiguration('takeoff_height')])),
                'odom_topic': odom_output_topic,
                'use_sim_time': use_sim_time,
            }],
            additional_env={'PYTHONPATH': merged_pythonpath},
            output='screen',
        )

    depth_topic_absolute = f'/drone{drone_id_str}/camera/depth'
    depth_restamped_topic = f'/drone{drone_id_str}/camera/depth_restamped'

    depth_restamp_node = Node(
        package='isaac_odom_bridge',
        executable='depth_restamp_node',
        name='depth_restamp_node',
        namespace=namespace,
        parameters=[{
            'input_topic': depth_topic_absolute,
            'output_topic': depth_restamped_topic,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # traj_server converts the planner's B-spline trajectory into
    # per-tick PositionCommand setpoints (100Hz, via its own wall timer).
    # Its position_cmd publisher is hardcoded to the absolute topic
    # /position_cmd in traj_server.cpp -- not remappable to a per-drone
    # name, fine for a single drone.
    traj_server_node = Node(
        package='ego_planner',
        executable='traj_server',
        name=f'drone_{drone_id_str}_traj_server',
        output='screen',
        remappings=[
            ('planning/bspline', f'drone_{drone_id_str}_planning/bspline'),
        ],
        parameters=[
            {'traj_server/time_forward': 1.0},
            {'use_sim_time': use_sim_time},
        ],
    )

    # Converts /position_cmd (world-frame pos/vel/acc/yaw) into the
    # body-frame Twist cmd_vel_mavlink_bridge expects. See the module's
    # own docstring for why this needs odom (for current yaw) and why
    # yaw uses feedforward + a P term rather than a straight yaw_dot
    # passthrough.
    poscmd_to_cmdvel_node = Node(
        package='isaac_odom_bridge',
        executable='poscmd_to_cmdvel_node',
        name='poscmd_to_cmdvel_node',
        namespace=namespace,
        parameters=[{
            'position_cmd_topic': '/position_cmd',
            'odom_topic': odom_output_topic,
            'cmd_vel_topic': cmd_vel_topic,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    ego_planner_node = Node(
        package='ego_planner',
        executable='ego_planner_node',
        name=f'drone_{drone_id_str}_ego_planner_node',
        output='screen',
        remappings=[
            ('odom_world', odom_output_topic),
            ('planning/bspline', f'drone_{drone_id_str}_planning/bspline'),
            ('planning/data_display', f'drone_{drone_id_str}_planning/data_display'),
            ('planning/broadcast_bspline_from_planner', '/broadcast_bspline'),
            ('planning/broadcast_bspline_to_planner', '/broadcast_bspline'),

            ('goal_point', f'drone_{drone_id_str}_plan_vis/goal_point'),
            ('global_list', f'drone_{drone_id_str}_plan_vis/global_list'),
            ('init_list', f'drone_{drone_id_str}_plan_vis/init_list'),
            ('optimal_list', f'drone_{drone_id_str}_plan_vis/optimal_list'),
            ('a_star_list', f'drone_{drone_id_str}_plan_vis/a_star_list'),

            ('grid_map/odom', odom_output_topic),
            ('grid_map/depth', depth_restamped_topic),
            ('grid_map/occupancy_inflate',
             f'drone_{drone_id_str}_grid/grid_map/occupancy_inflate'),
        ],
        parameters=[
            {'fsm/flight_type': 1},
            {'fsm/thresh_replan_time': 0.5},
            {'fsm/thresh_no_replan_meter': 1.0},
            {'fsm/planning_horizon': LaunchConfiguration('planning_horizon')},
            {'fsm/planning_horizen_time': 3.0},
            {'fsm/emergency_time': 1.0},
            {'fsm/realworld_experiment': False},
            {'fsm/fail_safe': True},

            {'fsm/waypoint_num': 0},

            {'grid_map/resolution': 0.1},
            {'grid_map/map_size_x': LaunchConfiguration('map_size_x_')},
            {'grid_map/map_size_y': LaunchConfiguration('map_size_y_')},
            {'grid_map/map_size_z': LaunchConfiguration('map_size_z_')},
            {'grid_map/local_update_range_x': 5.5},
            {'grid_map/local_update_range_y': 5.5},
            {'grid_map/local_update_range_z': 4.5},
            # Iris prop-tip radius is ~0.4m, but 0.4-0.5 was confirmed to make
            # the planner treat the drone's own position as inside an
            # obstacle in at least one corridor in this map (massive
            # replan-fail-loop: 133k "First 3 control points in obstacles"
            # warnings / 76k replan attempts / only 55 successes in one
            # 150s flight). Also note grid_map.cpp's inf_step = ceil(this /
            # resolution) rounds UP to whole 0.1m grid cells, so 0.45 and 0.5
            # were actually identical (both -> 5 cells -> 0.5m) -- to get a
            # real change, use a value that lands on a different cell count.
            {'grid_map/obstacles_inflation': 0.5},
            {'grid_map/local_map_margin': 10},
            {'grid_map/ground_height': -0.01},
            # camera parameters -- see the module docstring
            {'grid_map/cx': LaunchConfiguration('cx')},
            {'grid_map/cy': LaunchConfiguration('cy')},
            {'grid_map/fx': LaunchConfiguration('fx')},
            {'grid_map/fy': LaunchConfiguration('fy')},
            # depth filter
            {'grid_map/use_depth_filter': True},
            {'grid_map/depth_filter_tolerance': 0.15},
            {'grid_map/depth_filter_maxdist': 5.0},
            {'grid_map/depth_filter_mindist': 0.2},
            {'grid_map/depth_filter_margin': 2},
            {'grid_map/k_depth_scaling_factor': LaunchConfiguration('k_depth_scaling_factor')},
            {'grid_map/skip_pixel': 2},
            # local fusion
            {'grid_map/p_hit': 0.75},
            {'grid_map/p_miss': 0.35},
            {'grid_map/p_min': 0.12},
            {'grid_map/p_max': 0.90},
            {'grid_map/p_occ': 0.70},
            {'grid_map/min_ray_length': 0.1},
            {'grid_map/max_ray_length': 4.5},

            # real ceiling in the Full Warehouse scene is ~9.3m; keep the
            # virtual ceiling well below it so nothing plans right up to
            # the physical limit
            {'grid_map/virtual_ceil_height': 8.5},
            {'grid_map/visualization_truncate_height': 8.5},
            {'grid_map/show_occ_time': False},
            # ODOMETRY: grid_map/odom (real, RELIABLE topic) + the
            # standard optical->body cam2body_ rotation. See the module
            # docstring for why POSE_STAMPED (1) doesn't work here.
            {'grid_map/pose_type': 2},
            {'grid_map/frame_id': 'world'},

            {'manager/max_vel': LaunchConfiguration('max_vel')},
            {'manager/max_acc': LaunchConfiguration('max_acc')},
            {'manager/max_jerk': 4.0},
            {'manager/control_points_distance': 0.4},
            {'manager/feasibility_tolerance': 0.05},
            {'manager/planning_horizon': LaunchConfiguration('planning_horizon')},
            {'manager/use_distinctive_trajs': True},
            {'manager/drone_id': drone_id},

            {'optimization/lambda_smooth': 1.0},
            {'optimization/lambda_collision': 0.5},
            {'optimization/lambda_feasibility': 0.1},
            {'optimization/lambda_fitness': 1.0},
            {'optimization/dist0': 0.5},
            {'optimization/swarm_clearance': 0.5},
            {'optimization/max_vel': LaunchConfiguration('max_vel')},
            {'optimization/max_acc': LaunchConfiguration('max_acc')},

            {'bspline/limit_vel': LaunchConfiguration('max_vel')},
            {'bspline/limit_acc': LaunchConfiguration('max_acc')},
            {'bspline/limit_ratio': 1.1},

            {'prediction/obj_num': 0},
            {'prediction/lambda': 1.0},
            {'prediction/predict_rate': 1.0},

            {'use_sim_time': use_sim_time},
        ],
    )

    foxglove_bridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('foxglove_bridge'),
                'launch', 'foxglove_bridge_launch.xml')),
        launch_arguments={'port': LaunchConfiguration('foxglove_port')}.items(),
        condition=IfCondition(LaunchConfiguration('enable_foxglove')),
    )

    return [
        world_map_tf_node, odom_node, cmd_vel_node, depth_restamp_node,
        ego_planner_node, traj_server_node, poscmd_to_cmdvel_node,
        foxglove_bridge,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('drone_id', default_value='0'),

        DeclareLaunchArgument(
            'takeoff_height', default_value='2.0',
            description='px4 backend only: after auto arm + OFFBOARD, climb '
                        'straight up to this odom z (m) before ego-planner '
                        'takes over. 0 disables.'),
        DeclareLaunchArgument(
            'backend', default_value='ardupilot',
            description="'ardupilot' or 'px4' -- selects which cmd_vel "
                        "bridge node (cmd_vel_mavlink_bridge vs "
                        "cmd_vel_px4_bridge) is launched and the default "
                        "MAVLink port scheme used when 'mavlink_connection' "
                        "is left empty."),

        # All nodes below use this instead of wall-clock time, reading it from /clock
        # (published by Isaac's ROS2Backend when pub_clock is enabled -- see
        # ros2_backend.py). Needed because the sim's real-time factor can be well
        # below 1 (heavy rendering, camera sensors): traj_server's t_cur, grid_map's
        # depth+odom time-sync, and every other now()-based computation in this graph
        # were previously measured against wall-clock time while the vehicle itself
        # only advanced at simulation speed, so the planner's replan starting point
        # ran further and further ahead of where the vehicle actually was -- by several
        # meters after under a minute of flight -- eventually placing it past obstacles
        # the vehicle hadn't reached (or even seen) yet. Set to false to fall back to
        # each node's own wall clock (e.g. if /clock isn't being published).
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # odom_bridge_node
        DeclareLaunchArgument(
            'twist_topic', default_value='state/twist_inertial',
            description="'state/twist_inertial' (world frame) or "
                        "'state/twist' (body frame) -- confirm which "
                        "is which against /droneN/sensors/gps_twist "
                        "before flying"),
        DeclareLaunchArgument(
            'odometry_topic', default_value='visual_slam/odom',
            description='odom_bridge_node publishes here, and ego_planner '
                        'is remapped to read odom_world/grid_map/odom '
                        'from the same composed topic.'),

        # cmd_vel_mavlink_bridge
        DeclareLaunchArgument(
            'cmd_vel_topic', default_value='cmd_vel',
            description="Relative topic name (resolves under /droneN). "
                        "Pass an absolute name like '/cmd_vel' to match a "
                        "single-vehicle setup that isn't namespaced yet."),
        DeclareLaunchArgument(
            'mavlink_connection', default_value='',
            description='e.g. udp:127.0.0.1:14550. Leave empty to '
                        'auto-derive from drone_id as 14550 + 10*drone_id '
                        '(sim_vehicle.py default per-instance port).'),
        DeclareLaunchArgument(
            'venv_site_packages',
            default_value='/home/peterson/venv-ardupilot/lib/python3.12/site-packages',
            description='Prepended to PYTHONPATH for cmd_vel_mavlink_bridge '
                        'only, so pymavlink is importable despite the '
                        "colcon-generated script's system-python shebang."),

        # ego_planner_node / grid_map
        DeclareLaunchArgument('map_size_x_', default_value='42.0'),
        DeclareLaunchArgument('map_size_y_', default_value='82.0'),
        DeclareLaunchArgument('map_size_z_', default_value='10.5'),
        DeclareLaunchArgument('cx', default_value='160.0',
                               description='From /drone0/camera/color/camera_info '
                                           '(320x240 -- see this file\'s module '
                                           'docstring for why not 640x480).'),
        DeclareLaunchArgument('cy', default_value='120.0'),
        DeclareLaunchArgument('fx', default_value='228.50367736816406'),
        DeclareLaunchArgument('fy', default_value='228.50367736816406'),
        DeclareLaunchArgument('k_depth_scaling_factor', default_value='1000.0'),
        DeclareLaunchArgument('max_vel', default_value='1.0'),
        DeclareLaunchArgument('max_acc', default_value='2.0'),
        DeclareLaunchArgument('planning_horizon', default_value='7.5'),

        # foxglove_bridge
        DeclareLaunchArgument(
            'enable_foxglove', default_value='true',
            description='Also start foxglove_bridge for live visualization '
                        'in Foxglove Studio.'),
        DeclareLaunchArgument('foxglove_port', default_value='8765'),

        OpaqueFunction(function=launch_setup),
    ])
