---
title: Introduction to Gazebo and Unity
description: Learn about digital twin simulation using Gazebo and Unity for Physical AI & Humanoid Robotics
sidebar_position: 1
---

# Introduction to Gazebo and Unity

## Learning Objectives

- Understand the concept of digital twins in robotics
- Learn the fundamentals of Gazebo simulation environment
- Explore Unity as an alternative simulation platform
- Compare Gazebo and Unity for different use cases
- Set up basic simulation environments

## Prerequisites

- Understanding of ROS 2 fundamentals (Module 1)
- Basic knowledge of robot modeling concepts
- ROS 2 environment setup completed

## What is a Digital Twin?

A digital twin is a virtual representation of a physical robot or system that runs in parallel with the real system. In robotics, digital twins serve several critical purposes:

1. **Testing and Validation**: Verify algorithms and behaviors in simulation before deployment
2. **Training**: Train machine learning models in safe, controlled environments
3. **Development**: Develop and debug robot software without hardware access
4. **Optimization**: Optimize robot performance through simulation analysis

### Benefits of Digital Twins

- **Safety**: Test dangerous scenarios without risk to hardware or humans
- **Cost-Effectiveness**: Reduce hardware wear and tear
- **Speed**: Run simulations faster than real-time
- **Reproducibility**: Create consistent testing conditions
- **Scalability**: Test multiple robots simultaneously

## Introduction to Gazebo

Gazebo is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in the robotics community and integrates well with ROS.

### Key Features of Gazebo

- **Physics Engine**: Uses ODE, Bullet, or DART for accurate physics simulation
- **Sensor Simulation**: Supports cameras, LiDAR, IMU, GPS, and other sensors
- **Realistic Rendering**: High-quality 3D graphics with dynamic lighting
- **Plugin Architecture**: Extensible through custom plugins
- **ROS Integration**: Direct integration with ROS/ROS 2 through gazebo_ros_pkgs

### Gazebo Architecture

Gazebo consists of several components:

- **Server (gzserver)**: Runs the simulation and physics engine
- **Client (gzclient)**: Provides the graphical user interface
- **Plugins**: Extend functionality for sensors, controllers, and communication
- **Models**: Represent robots, objects, and environments

### Basic Gazebo Simulation

Here's a simple world file that creates a basic environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Introduction to Unity

Unity is a powerful game engine that has been adapted for robotics simulation. It offers high-fidelity graphics and physics simulation, making it suitable for applications requiring photorealistic rendering or complex 3D environments.

### Key Features of Unity for Robotics

- **High-Fidelity Graphics**: Photorealistic rendering capabilities
- **Physics Engine**: Built-in physics simulation with PhysX
- **XR Support**: Excellent support for VR/AR applications
- **C# Scripting**: Uses C# for custom behaviors and logic
- **Asset Store**: Large library of 3D models and components
- **Cross-Platform**: Deploy to multiple platforms

### Unity Robotics Simulation

Unity Robotics provides several tools and packages:

- **Unity Robotics Hub**: Centralized access to robotics packages
- **Unity Robot Templates**: Pre-built robot models and environments
- **ROS#**: Bridge for ROS/ROS 2 communication
- **ML-Agents**: Machine learning framework for robot training
- **Synthetic Data Generation**: Tools for creating training datasets

## Setting up Gazebo with ROS 2

### Installation

```bash
# Install Gazebo Garden (recommended version for ROS 2 Humble)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Install additional Gazebo components
sudo apt install gz-garden
```

### Basic Gazebo Integration

To launch Gazebo with ROS 2, you can use the gazebo_ros_pkgs:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Get Gazebo launch directory
    gazebo_launch_dir = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch'
    )

    return LaunchDescription([
        # Launch Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_launch_dir, 'gzserver.launch.py')
            ),
            launch_arguments={'world': os.path.join(get_package_share_directory('my_robot_gazebo'), 'worlds', 'simple.world')}.items()
        ),

        # Launch Gazebo client (GUI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_launch_dir, 'gzclient.launch.py')
            )
        )
    ])
```

### Spawning Robots in Gazebo

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get URDF file
    urdf_file = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'my_robot.urdf'
    )

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'my_robot'
            ],
            output='screen'
        )
    ])
```

## Setting up Unity for Robotics

### Installation Requirements

- Unity Hub (recommended)
- Unity Editor (2021.3 LTS or newer)
- Unity Robotics packages

### Unity ROS Integration

Unity can communicate with ROS 2 using the ROS# package or the newer Unity Robotics package:

```csharp
using UnityEngine;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopic = "/cmd_vel";

    void Start()
    {
        // Get the ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(robotTopic);
    }

    void Update()
    {
        // Example: Send velocity command
        if (Input.GetKeyDown(KeyCode.Space))
        {
            var twist = new TwistMsg();
            twist.linear.x = 1.0f;
            twist.angular.z = 0.5f;

            ros.Publish(robotTopic, twist);
        }
    }
}
```

## Comparing Gazebo and Unity

| Aspect | Gazebo | Unity |
|--------|--------|-------|
| **Physics Accuracy** | High | Good |
| **Graphics Quality** | Good | Excellent |
| **ROS Integration** | Native | Requires bridge |
| **Learning Curve** | Moderate | Steep (C#) |
| **Performance** | Good for robotics | Excellent for graphics |
| **Cost** | Free | Free for personal/academic |
| **Use Cases** | Control, navigation, planning | Perception, training, visualization |

## Best Practices for Simulation

### Model Accuracy

- Use accurate physical properties (mass, inertia, friction)
- Include realistic sensor noise and limitations
- Validate simulation against real robot behavior

### Performance Optimization

- Simplify collision geometry where possible
- Use appropriate update rates
- Limit the number of active objects

### Simulation Fidelity

- Understand the limitations of your simulation
- Account for the "reality gap" in perception tasks
- Use domain randomization to improve transfer learning

## Exercises

1. **Gazebo Installation**: Install Gazebo and run a simple simulation with a robot model
2. **World Creation**: Create a custom world file with obstacles and test robot navigation
3. **Sensor Integration**: Add a camera or LiDAR sensor to your robot model in simulation
4. **Comparison Study**: Compare the same robot behavior in both Gazebo and Unity environments

## Summary

In this chapter, we've introduced digital twin simulation using Gazebo and Unity. We've covered the fundamentals of both platforms, their strengths and weaknesses, and how to integrate them with ROS 2. Simulation is a crucial tool in robotics development, allowing for safe, cost-effective testing and development of robotic systems.

## References

- [Gazebo Documentation](http://gazebosim.org/)
- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- [ROS 2 with Gazebo](https://github.com/ros-simulation/gazebo_ros_pkgs)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

## Next Steps

[← Previous Module: ROS 2](../module-1-ros2/chapter-5-launch-files-package-management) | [Next Chapter: Physics Simulation →](./chapter-2-physics-simulation)