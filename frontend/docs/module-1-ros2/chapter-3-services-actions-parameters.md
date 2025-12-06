---
title: Services, Actions, and Parameters
description: Learn about ROS 2 services, actions, and parameters for advanced robot communication
sidebar_position: 3
---

# Services, Actions, and Parameters

## Learning Objectives

- Understand the differences between topics, services, and actions
- Create and use ROS 2 services for request-response communication
- Implement actions for long-running tasks with feedback
- Use parameters for configuration management
- Apply appropriate communication patterns based on use case

## Prerequisites

- Understanding of nodes and topics (Chapters 1 and 2)
- Basic Python or C++ programming knowledge
- ROS 2 environment setup completed

## Services: Request-Response Communication

Services provide a request-response communication pattern in ROS 2. Unlike topics which are asynchronous, services are synchronous - the client sends a request and waits for a response from the service server.

### Service Architecture

In the service pattern:
- **Service Client**: Sends a request and waits for a response
- **Service Server**: Receives requests and sends responses
- **Service Type**: Defines the request and response message structure

### Creating a Service Definition

Services are defined using `.srv` files that specify the request and response structure:

```
# Request (before the --- separator)
string name
int32 age
---
# Response (after the --- separator)
bool success
string message
```

This would be saved as `PersonInfo.srv` in the `srv` directory of a package.

### Service Server Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions: Long-Running Tasks with Feedback

Actions are used for long-running tasks that require feedback and the ability to cancel. They combine aspects of both services and topics.

### Action Architecture

An action includes three components:
- **Goal**: The request to start a long-running task
- **Feedback**: Ongoing updates about the task progress
- **Result**: The final outcome of the task

### Action Definition

Actions are defined using `.action` files:

```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] sequence
```

### Action Server Implementation

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()

if __name__ == '__main__':
    main()
```

## Parameters: Configuration Management

Parameters provide a way to configure nodes at runtime. They are key-value pairs that can be set before or during node execution.

### Using Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('wheel_diameter', 0.1)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.wheel_diameter = self.get_parameter('wheel_diameter').value

        self.get_logger().info(f'Robot: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Wheel diameter: {self.wheel_diameter}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

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

### Setting Parameters at Launch

Parameters can be set in launch files:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='parameter_node',
            parameters=[
                {'robot_name': 'turtlebot4'},
                {'max_velocity': 2.0},
                {'wheel_diameter': 0.15},
            ]
        )
    ])
```

## When to Use Each Communication Pattern

### Topics vs Services vs Actions

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| Topics | Continuous data streams | Asynchronous, many-to-many |
| Services | Request-response | Synchronous, blocking |
| Actions | Long-running tasks | Goal-feedback-result |

### Examples of Appropriate Use

- **Topics**: Sensor data, robot state, logging
- **Services**: Calibration, configuration changes, simple computations
- **Actions**: Navigation, manipulation tasks, trajectory execution

## Command-Line Tools

### Service Tools

```bash
# List services
ros2 service list

# Get service information
ros2 service info <service_name>

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

### Action Tools

```bash
# List actions
ros2 action list

# Get action information
ros2 action info <action_name>

# Send a goal
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}"
```

### Parameter Tools

```bash
# List parameters for a node
ros2 param list <node_name>

# Get a parameter value
ros2 param get <node_name> <param_name>

# Set a parameter value
ros2 param set <node_name> <param_name> <value>
```

## Best Practices

1. **Service Design**: Use services for operations that have a clear request-response pattern
2. **Action Design**: Use actions for tasks that take significant time and need progress feedback
3. **Parameter Validation**: Always validate parameter values within acceptable ranges
4. **Error Handling**: Implement proper error handling for all communication patterns
5. **Documentation**: Clearly document the expected request/response formats

## Exercises

1. **Service Implementation**: Create a service that converts temperatures between Celsius and Fahrenheit
2. **Action Implementation**: Implement an action that moves a robot to a specified location with feedback on progress
3. **Parameter Management**: Create a node that uses parameters to configure its behavior and test changing parameters at runtime
4. **Communication Pattern Selection**: For a given robot scenario, identify which communication pattern (topic, service, or action) is most appropriate

## Summary

In this chapter, we've explored the three main communication patterns in ROS 2: services for request-response communication, actions for long-running tasks with feedback, and parameters for configuration management. Understanding when and how to use each pattern is crucial for designing effective robotic systems.

## References

- [ROS 2 Services Documentation](https://docs.ros.org/en/humble/Concepts/About-ROS-2-Services.html)
- [ROS 2 Actions Documentation](https://docs.ros.org/en/humble/Concepts/About-ROS-2-Actions.html)
- [ROS 2 Parameters Documentation](https://docs.ros.org/en/humble/Concepts/About-ROS-2-Parameters.html)

## Next Steps

[← Previous Chapter: Nodes and Topics](./chapter-2-nodes-and-topics) | [Next Chapter: URDF Robot Modeling →](./chapter-4-urdf-robot-modeling)