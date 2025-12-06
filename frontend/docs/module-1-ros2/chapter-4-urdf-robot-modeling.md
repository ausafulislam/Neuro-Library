---
title: URDF Robot Modeling
description: Learn about Unified Robot Description Format (URDF) for modeling robots in ROS 2
sidebar_position: 4
---

# URDF Robot Modeling

## Learning Objectives

- Understand the Unified Robot Description Format (URDF) for robot modeling
- Create URDF files to describe robot kinematics and dynamics
- Use Xacro to simplify complex URDF definitions
- Visualize robot models in RViz
- Integrate robot models with ROS 2 systems

## Prerequisites

- Understanding of ROS 2 nodes and topics (Chapters 1-2)
- Basic knowledge of 3D geometry and kinematics
- ROS 2 environment setup completed

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. It defines the physical and visual properties of a robot, including its links, joints, and how they connect to form a kinematic chain.

### URDF Structure

A URDF file contains:
- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links that allow motion
- **Visual**: Visual representation for display
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## Basic URDF Elements

### Links

A link represents a rigid body in the robot. It can have visual, collision, and inertial properties:

```xml
<link name="base_link">
  <visual>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="10"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

### Joints

A joint connects two links and defines how they can move relative to each other:

```xml
<joint name="base_to_wheel" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0 0.2 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>
```

Joint types include:
- **fixed**: No movement allowed
- **continuous**: Continuous rotation (like a wheel)
- **revolute**: Limited rotation (like a servo)
- **prismatic**: Linear motion (like a linear actuator)
- **floating**: 6 degrees of freedom
- **planar**: Planar motion

## Complete URDF Example

Here's a simple robot with a base and two wheels:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 -0.05" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 -0.05" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

## Using Xacro for Complex Models

Xacro (XML Macros) allows you to use variables, macros, and expressions in URDF files to reduce repetition:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Properties -->
  <xacro:property name="base_width" value="0.5"/>
  <xacro:property name="base_length" value="0.3"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_y_offset" value="0.2"/>

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix *origin">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="origin"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="left">
    <origin xyz="0 ${wheel_y_offset} -0.05" rpy="1.5707 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="right">
    <origin xyz="0 -${wheel_y_offset} -0.05" rpy="1.5707 0 0"/>
  </xacro:wheel>

</robot>
```

## Robot State Publishing

To visualize a robot model in RViz, you need to publish the robot state using the `robot_state_publisher` package:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to publish state
        self.timer = self.create_timer(0.1, self.publish_state)

        # Initialize joint positions
        self.joint_positions = {'left_wheel_joint': 0.0, 'right_wheel_joint': 0.0}

    def publish_state(self):
        # Create joint state message
        msg = JointState()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint positions (for example, simulate rotation)
        self.joint_positions['left_wheel_joint'] += 0.1
        self.joint_positions['right_wheel_joint'] += 0.1

        # Publish joint states
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launching Robot Models

Create a launch file to bring up your robot model:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    urdf_model_path = LaunchConfiguration('model')
    declare_urdf_model_path = DeclareLaunchArgument(
        name='model',
        default_value=PathJoinSubstitution([FindPackageShare('my_robot_description'), 'urdf', 'robot.xacro']),
        description='Absolute path to robot urdf file'
    )

    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_model_path.get_path()).read()}]
    )

    # Joint State Publisher Node (for GUI)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_gui': True}]
    )

    return LaunchDescription([
        declare_urdf_model_path,
        robot_state_publisher_node,
        joint_state_publisher_node
    ])
```

## Visualizing in RViz

To visualize your robot model in RViz:

1. Launch your robot state publisher
2. Open RViz: `rviz2`
3. Add a RobotModel display
4. Set the Robot Description parameter to 'robot_description'
5. Set the TF Prefix if needed

## Best Practices for URDF

1. **Use Xacro**: For complex robots, use Xacro to avoid repetition
2. **Proper Naming**: Use consistent, descriptive names for links and joints
3. **Inertial Properties**: Include accurate inertial properties for simulation
4. **Kinematic Chains**: Ensure your joints form proper kinematic chains
5. **Collision vs Visual**: Use simpler geometry for collision models than visual models
6. **Units**: Use meters for length and kilograms for mass

## Common URDF Issues

- **Self-collision**: Links colliding with themselves during motion
- **Kinematic loops**: Closed loops in kinematic chains (use transmission elements)
- **Invalid transforms**: Joint transforms that result in unreachable configurations
- **Inertial errors**: Incorrect inertial properties causing simulation instability

## Exercises

1. **Simple Robot**: Create a URDF for a simple differential drive robot with a base and two wheels
2. **Xacro Conversion**: Convert a simple URDF to Xacro format to demonstrate macro usage
3. **Complex Robot**: Create a URDF for a robot arm with multiple joints
4. **Visualization**: Launch your robot model in RViz and verify it displays correctly

## Summary

In this chapter, we've explored URDF (Unified Robot Description Format), which is essential for representing robots in ROS 2. We've learned how to create links and joints, use Xacro to simplify complex models, and visualize robots in RViz. URDF is fundamental for robot simulation, visualization, and kinematic analysis.

## References

- [URDF Documentation](http://wiki.ros.org/urdf)
- [Xacro Documentation](http://wiki.ros.org/xacro)
- [Robot State Publisher](http://wiki.ros.org/robot_state_publisher)
- [TF2 Documentation](http://wiki.ros.org/tf2)

## Next Steps

[← Previous Chapter: Services, Actions, and Parameters](./chapter-3-services-actions-parameters) | [Next Chapter: Launch Files and Package Management →](./chapter-5-launch-files-package-management)