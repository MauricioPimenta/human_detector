from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events.matchers import matches_action
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
import glob
import os


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    mediapipe_venv = os.path.expanduser("~/venvs/mediapipe")

    venv_bin = os.path.join(mediapipe_venv, "bin")
    venv_site_packages = sorted(
        glob.glob(os.path.join(mediapipe_venv, "lib", "python*", "site-packages"))
    )
    pythonpath_entries = venv_site_packages[:]
    if os.environ.get("PYTHONPATH"):
        pythonpath_entries.append(os.environ["PYTHONPATH"])

    # Launcher Arguments
    use_sim_arg = DeclareLaunchArgument("use_sim_time", default_value="True", description="Use sim time.")
    namespace_arg = DeclareLaunchArgument("namespace", default_value="a200_0000", description="Namespace for the human detector node.")


    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    human_detector = LifecycleNode(
        package="human_detector",
        executable="human_detector",
        name="human_detector",
        output="screen",
        namespace=namespace,
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
        remappings=remappings,
        additional_env={
            "VIRTUAL_ENV": mediapipe_venv,
            "PATH": os.pathsep.join([venv_bin, os.environ.get("PATH", "")]),
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        },
    )

    # 1) Configure AFTER the process starts
    configure_on_start = RegisterEventHandler(
        OnProcessStart(
            target_action=human_detector,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(human_detector),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    )
                )
            ],
        )
    )
    # 2) Activate AFTER it successfully reaches "inactive" (i.e., configured)
    activate_on_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=human_detector,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(human_detector),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )



    return LaunchDescription(
        [
            use_sim_arg,
            namespace_arg,
            human_detector,
            configure_on_start,
            activate_on_inactive,
        ]
    )
