from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.events.matchers import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Launcher Arguments
    use_sim_arg = DeclareLaunchArgument("use_sim_time", default_value="True", description="Use sim time.")
    namespace_arg = DeclareLaunchArgument("namespace", default_value="", description="Namespace for the human detector node.")


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
        remappings=remappings
    )

    move_human_detector_to_configure_state_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(human_detector),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )



    return LaunchDescription(
        [
            use_sim_arg,
            namespace_arg,
            human_detector,
            move_human_detector_to_configure_state_event,
        ]
    )
