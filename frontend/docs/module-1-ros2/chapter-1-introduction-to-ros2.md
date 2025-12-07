---
title: Introduction to ROS 2
description: Learn the fundamentals of Robot Operating System 2 (ROS 2) for humanoid robotics applications.
keywords: [ros2, robotics, middleware, introduction, nodes, topics]
sidebar_position: 1
---

# Introduction to ROS 2

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the fundamental concepts of ROS 2 and its role in robotics
- Understand the architecture and design principles of ROS 2
- Set up a basic ROS 2 workspace and environment
- Create and run your first ROS 2 nodes
- Identify the key differences between ROS 1 and ROS 2

## Prerequisites

- Basic Python or C++ programming knowledge
- Familiarity with Linux command line
- Understanding of basic robotics concepts (optional but helpful)

## What is ROS 2?

Robot Operating System 2 (ROS 2) is not an actual operating system, but rather a flexible framework for writing robot software. It provides libraries, tools, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms and configurations.

ROS 2 is the successor to ROS 1, designed to address the limitations of its predecessor while maintaining the core principles that made ROS 1 successful in the robotics community. The key improvements in ROS 2 include:

- **Real-time support**: Deterministic behavior for time-critical applications
- **Multi-robot systems**: Better support for coordinating multiple robots
- **Production deployment**: More robust for commercial applications
- **Quality of Service (QoS)**: Configurable reliability and performance parameters
- **Security**: Built-in security features for safe robot operation

## ROS 2 Architecture

ROS 2 is built on a distributed architecture that allows nodes to communicate across different machines and processes. The core components include:

### Nodes
A node is a process that performs computation. Nodes are organized into packages that can be shared and reused. In ROS 2, nodes are designed to be more independent and robust than in ROS 1.

### Topics
Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic. This creates a many-to-many relationship where multiple publishers and subscribers can exist for the same topic.

### Services
Services provide a request/response communication pattern. A service client sends a request and waits for a response from a service server. This is useful for operations that need a specific result.

### Actions
Actions are a more complex communication pattern that includes goals, feedback, and results. They're designed for long-running tasks that may take time to complete and need to provide ongoing feedback.

### Parameters
Parameters are configuration values that can be changed at runtime. They're stored in a centralized parameter server and can be accessed by any node.

## Setting Up Your ROS 2 Environment

### Installation

For this textbook, we recommend using Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill, which is an LTS (Long Term Support) version with extended support.

```bash
# Add the ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep2
sudo apt install -y python3-colcon-common-extensions
```

### Environment Setup

Add the following line to your `~/.bashrc` file:

```bash
source /opt/ros/humble/setup.bash
```

Then source your environment:

```bash
source ~/.bashrc
```

## Creating Your First ROS 2 Package

Let's create a simple package to demonstrate the basic concepts:

```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Create a new package
cd src
ros2 pkg create --build-type ament_python beginner_tutorials
```

This creates a basic Python package structure with the necessary files for a ROS 2 package.

## ROS 2 vs ROS 1: Key Differences

### Middleware
- **ROS 1**: Uses custom TCPROS/UDPROS transport
- **ROS 2**: Uses DDS (Data Distribution Service) as the middleware, providing more flexibility and better real-time support

### Build System
- **ROS 1**: Uses catkin build system
- **ROS 2**: Uses colcon build system, which is more flexible and supports multiple build systems

### Lifecycle Management
- **ROS 1**: Nodes start and run until terminated
- **ROS 2**: Nodes can have explicit lifecycle states (unconfigured, inactive, active, finalized)

### Quality of Service (QoS)
- **ROS 1**: No QoS settings
- **ROS 2**: Configurable QoS profiles for reliability, durability, and performance

## Best Practices for ROS 2 Development

1. **Use meaningful names**: Choose clear, descriptive names for your nodes, topics, and services
2. **Follow naming conventions**: Use forward slashes to separate namespaces (e.g., `/arm/joint_states`)
3. **Handle errors gracefully**: Implement proper error handling and logging
4. **Use launch files**: Organize your system startup with launch files
5. **Parameterize your nodes**: Use parameters instead of hardcoded values
6. **Document your interfaces**: Clearly document your topics, services, and actions

## Exercises

1. **Environment Setup**: Install ROS 2 Humble on your development machine and verify the installation by running `ros2 topic list`
2. **Package Creation**: Create a new ROS 2 package called `my_first_robot` with Python support
3. **Node Exploration**: Use `ros2 node list` and `ros2 node info <node_name>` to explore nodes in your system

## References

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Design](https://design.ros2.org/)
