---
title: Launch Files and Package Management
description: Learn about ROS 2 launch files and package management for organizing robot systems
sidebar_position: 5
---

# Launch Files and Package Management

## Learning Objectives

- Understand ROS 2 package structure and organization
- Create and manage ROS 2 packages using colcon
- Write launch files to start multiple nodes at once
- Use parameters and configuration files effectively
- Implement best practices for system organization

## Prerequisites

- Understanding of ROS 2 nodes and topics (Chapters 1-2)
- Basic knowledge of ROS 2 communication patterns (Chapter 3)
- Basic command-line skills

## ROS 2 Package Structure

A ROS 2 package is the basic building block of a ROS 2 system. It contains nodes, libraries, and other resources organized in a standard structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.py                # Build configuration for Python
├── setup.cfg               # Installation configuration
├── src/                    # Source code (C++)
│   └── my_robot_package/
│       └── my_node.cpp
├── my_robot_package/       # Python modules
│   ├── __init__.py
│   └── my_node.py
├── launch/                 # Launch files
│   └── robot.launch.py
├── config/                 # Configuration files
│   └── params.yaml
├── urdf/                   # Robot description files
│   └── robot.urdf
└── test/                   # Test files
    └── test_my_node.py
```

### package.xml

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>My robot package for the Neuro Library textbook</description>
  <maintainer email="student@neuro-library.org">Student</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### CMakeLists.txt (for C++ packages)

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Create executables
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp std_msgs)

# Install executables
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Creating and Building Packages

### Creating a Python Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package
```

### Creating a C++ Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake my_robot_package
```

### Building Packages

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

## Launch Files

Launch files allow you to start multiple nodes and configure your system with a single command. ROS 2 uses Python for launch files.

### Basic Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node',
            output='screen'
        )
    ])
```

### Launch File with Parameters

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    robot_name_launch_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    return LaunchDescription([
        robot_name_launch_arg,
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'robot_name': LaunchConfiguration('robot_name')},
                {'max_velocity': 1.0}
            ],
            output='screen'
        )
    ])
```

### Launch File with Multiple Nodes

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Robot controller node
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'max_velocity': 1.0},
                {'wheel_diameter': 0.1}
            ],
            output='screen'
        ),

        # Sensor processor node
        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'sensor_range': 10.0},
                {'update_rate': 10.0}
            ],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description': open('/path/to/robot.urdf').read()}
            ],
            output='screen'
        )
    ])
```

### Launch File with Conditions

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    start_rviz = DeclareLaunchArgument(
        'start_rviz',
        default_value='true',
        description='Start RViz'
    )

    return LaunchDescription([
        use_sim_time,
        start_rviz,

        # Robot controller
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # RViz (only if start_rviz is true)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            output='screen'
        )
    ])
```

## Parameter Files (YAML)

Parameters can be organized in YAML files for easier management:

```yaml
# config/robot_params.yaml
my_robot:
  ros__parameters:
    robot_name: "turtlebot4"
    max_velocity: 0.5
    wheel_diameter: 0.1
    sensor_range: 5.0
    update_rate: 10.0

    # Nested parameter group
    navigation:
      goal_tolerance: 0.1
      max_linear_vel: 0.5
      max_angular_vel: 1.0
```

Loading parameters from YAML in launch files:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the parameter file
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[config],
            output='screen'
        )
    ])
```

## Colcon Build System

Colcon is the build system used in ROS 2. It's more flexible than the previous catkin system.

### Common Colcon Commands

```bash
# Build all packages
colcon build

# Build specific packages
colcon build --packages-select my_robot_package

# Build with verbose output
colcon build --event-handlers console_direct+

# Build with specific CMake arguments
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Run tests
colcon test
colcon test-result --all

# Clean build directory
rm -rf build/ install/ log/
```

## Package Management Best Practices

### Naming Conventions

- Use lowercase with underscores: `my_robot_driver`
- Use descriptive names that indicate purpose
- Avoid generic names like `robot` or `node`

### Directory Structure

- Keep related functionality in the same package
- Use separate packages for distinct components
- Organize launch files by use case (simulation, real robot, testing)

### Dependency Management

- Declare all dependencies in `package.xml`
- Use the minimum required versions
- Separate build, exec, and test dependencies

## Advanced Launch Concepts

### Including Other Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Include another launch file
    other_launch_file = os.path.join(
        get_package_share_directory('other_package'),
        'launch',
        'other_launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(other_launch_file)
        )
    ])
```

### Launch Actions

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import LogInfo, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    robot_node = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller'
    )

    return LaunchDescription([
        # Log a message when the launch starts
        LogInfo(msg='Starting robot system...'),

        # Start the robot node
        robot_node,

        # Register an event handler
        RegisterEventHandler(
            OnProcessStart(
                target_action=robot_node,
                on_start=[
                    LogInfo(msg='Robot controller started successfully!')
                ]
            )
        )
    ])
```

## Command-Line Tools

### Package Tools

```bash
# List all packages
ros2 pkg list

# Show package information
ros2 pkg info my_robot_package

# Find a package path
ros2 pkg prefix my_robot_package

# Create a new package
ros2 pkg create --build-type ament_python new_package
```

### Launch Tools

```bash
# List available launch files in a package
ros2 launch my_robot_package

# Run a launch file
ros2 launch my_robot_package robot.launch.py

# Run with arguments
ros2 launch my_robot_package robot.launch.py robot_name:=my_robot
```

## Exercises

1. **Package Creation**: Create a new ROS 2 package with a simple node that publishes a message
2. **Launch File**: Create a launch file that starts multiple nodes with different parameters
3. **Parameter Configuration**: Create a YAML parameter file and use it in a launch file
4. **Package Dependencies**: Create two packages where one depends on the other and demonstrates inter-package communication

## Summary

In this chapter, we've covered ROS 2 package management and launch files, which are essential for organizing and deploying robot systems. We've learned how to create packages, organize code and resources, write launch files to start complex systems, and manage parameters effectively. These skills are crucial for building maintainable and scalable robotic applications.

## References

- [ROS 2 Package Documentation](https://docs.ros.org/en/humble/How-To-Guides/Creating-Your-First-ROS2-Package.html)
- [ROS 2 Launch Documentation](https://docs.ros.org/en/humble/How-To-Guides/Launch-system.html)
- [Colcon Build System](https://colcon.readthedocs.io/en/released/)
- [ROS 2 Parameter System](https://docs.ros.org/en/humble/How-To-Guides/Using-Parameters-in-a-classic-ROS2-node.html)

## Next Steps

[← Previous Chapter: URDF Robot Modeling](./chapter-4-urdf-robot-modeling) | [Next Module: Digital Twin →](../module-2-digital-twin/chapter-1-introduction-gazebo-unity)