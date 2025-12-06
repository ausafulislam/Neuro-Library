---
title: Nodes and Topics
description: Learn about ROS 2 nodes and topics for robot communication and coordination.
keywords: [ros2, nodes, topics, communication, messaging]
sidebar_position: 2
---

# Nodes and Topics

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and manage ROS 2 nodes in both Python and C++
- Understand the publish-subscribe communication pattern
- Design effective topic architectures for robot systems
- Implement robust message passing between nodes
- Use tools to monitor and debug topic communication

## Prerequisites

- Completion of Chapter 1: Introduction to ROS 2
- Basic Python or C++ programming knowledge
- Understanding of object-oriented programming concepts

## Understanding Nodes

In ROS 2, a node is the fundamental building block of a robot application. A node is a process that performs computation and communicates with other nodes through topics, services, and actions.

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes several states:
- **Unconfigured**: The node is created but not yet configured
- **Inactive**: The node is configured but not yet activated
- **Active**: The node is running and performing its tasks
- **Finalized**: The node has been shut down and cleaned up

This lifecycle allows for more robust and predictable system behavior, especially in complex robotic applications.

### Creating a Node in Python

Here's a basic example of creating a ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Creating a Node in C++

Here's the equivalent node in C++:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello World: " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclpy::init(argc, argv);
  rclpy::spin(std::make_shared<MinimalPublisher>());
  rclpy::shutdown();
  return 0;
}
```

## Topics: The Publish-Subscribe Pattern

Topics are the primary method of communication in ROS 2. They use a publish-subscribe pattern where nodes can publish messages to a topic and other nodes can subscribe to that topic to receive the messages.

### Topic Characteristics

- **Asynchronous**: Publishers and subscribers don't need to run at the same time
- **Many-to-many**: Multiple publishers can publish to the same topic, and multiple subscribers can subscribe to the same topic
- **Typed**: Each topic has a specific message type that all publishers and subscribers must use
- **Named**: Topics have hierarchical names (e.g., `/arm/joint_states`)

### Quality of Service (QoS) Settings

ROS 2 provides Quality of Service (QoS) settings that allow you to configure the reliability, durability, and performance characteristics of topic communication:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Use it when creating a publisher
publisher = self.create_publisher(String, 'topic', qos_profile)
```

Common QoS policies include:
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local (whether messages persist for late-joining subscribers)
- **History**: Keep all vs. keep last N messages
- **Deadline**: Maximum time between messages
- **Lifespan**: How long messages are kept in the system

## Creating a Subscriber Node

Here's an example of a subscriber node that receives messages:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()
```

## Topic Tools and Monitoring

ROS 2 provides several command-line tools for monitoring and debugging topic communication:

### List Topics
```bash
ros2 topic list
```

### Echo Topic Messages
```bash
ros2 topic echo /topic std_msgs/msg/String
```

### Publish to a Topic
```bash
ros2 topic pub /topic std_msgs/msg/String "data: 'Hello'"
```

### Get Topic Information
```bash
ros2 topic info /topic
```

## Best Practices for Topic Design

### Naming Conventions
- Use forward slashes to separate namespaces: `/arm/joint_states`
- Use descriptive names that clearly indicate the content
- Group related topics under common namespaces

### Message Design
- Keep messages small and efficient
- Use appropriate data types for your application
- Consider bandwidth limitations in distributed systems
- Version your message types appropriately

### Performance Considerations
- Use appropriate QoS settings for your use case
- Limit the frequency of high-bandwidth topics
- Use latching for static information that late-joining nodes need
- Consider using services for request-response communication when appropriate

## Example: Sensor Data Pipeline

Here's a practical example of using nodes and topics for a sensor data pipeline:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped


class SensorProcessor(Node):

    def __init__(self):
        super().__init__('sensor_processor')

        # Subscribe to raw laser scan data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.scan_callback,
            10)

        # Publish processed obstacle data
        self.obstacle_publisher = self.create_publisher(
            PointStamped,
            '/obstacles',
            10)

    def scan_callback(self, msg):
        # Process the laser scan to detect obstacles
        # This is a simplified example
        for i, range_val in enumerate(msg.ranges):
            if range_val < 1.0 and range_val > msg.range_min:
                # Create obstacle message
                obstacle_msg = PointStamped()
                obstacle_msg.header = msg.header
                # Convert polar to Cartesian coordinates
                angle = msg.angle_min + i * msg.angle_increment
                obstacle_msg.point.x = range_val * math.cos(angle)
                obstacle_msg.point.y = range_val * math.sin(angle)
                obstacle_msg.point.z = 0.0

                # Publish obstacle
                self.obstacle_publisher.publish(obstacle_msg)
```

## Exercises

1. **Basic Publisher-Subscriber**: Create a publisher node that publishes the current time and a subscriber node that prints the received time messages.

2. **Sensor Pipeline**: Implement a node that subscribes to a simulated IMU topic and publishes filtered orientation data.

3. **Topic Monitoring**: Use `ros2 topic` commands to monitor the communication between your nodes and analyze the message frequency and size.

## References

- [ROS 2 Node Concepts](https://docs.ros.org/en/humble/Concepts/About-ROS-2-Nodes.html)
- [ROS 2 Topics and Messages](https://docs.ros.org/en/humble/Concepts/About-ROS-2-Topics.html)
- [Quality of Service in ROS 2](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service.html)

## Next Steps

[← Previous Chapter: Introduction to ROS 2](./chapter-1-introduction-to-ros2) | [Next Chapter: Services, Actions, and Parameters →](./chapter-3-services-actions-parameters)