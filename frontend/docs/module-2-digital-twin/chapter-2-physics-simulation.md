---
title: Physics Simulation
description: Learn about physics simulation in Gazebo and Unity for realistic robot behavior in digital twins
sidebar_position: 2
---

# Physics Simulation

## Learning Objectives

- Understand the fundamentals of physics simulation in robotics
- Learn how physics engines model real-world forces and interactions
- Configure physics parameters for accurate simulation
- Implement realistic robot dynamics in simulation
- Validate simulation physics against real-world behavior

## Prerequisites

- Understanding of digital twin concepts (Chapter 1)
- Basic knowledge of Newtonian mechanics
- ROS 2 and simulation environment setup

## Physics Simulation Fundamentals

Physics simulation in robotics involves modeling the fundamental forces and interactions that govern robot behavior in the real world. This includes:

- **Rigid Body Dynamics**: Movement and interaction of solid objects
- **Collision Detection**: Determining when objects make contact
- **Contact Response**: Calculating forces and reactions when objects touch
- **Constraints**: Limiting motion through joints and connections

### Key Physics Concepts

1. **Newton's Laws of Motion**:
   - First Law: Objects at rest stay at rest, objects in motion stay in motion
   - Second Law: F = ma (Force equals mass times acceleration)
   - Third Law: Every action has an equal and opposite reaction

2. **Energy Conservation**: Kinetic and potential energy transformations

3. **Momentum**: Linear and angular momentum conservation

## Physics Engines in Simulation

### Overview of Physics Engines

Different simulation environments use different physics engines, each with specific strengths:

- **ODE (Open Dynamics Engine)**: Used in older versions of Gazebo
- **Bullet**: Fast, stable, good for real-time applications
- **DART**: Advanced features, good for humanoid robots
- **PhysX**: NVIDIA's engine, used in Unity

### Choosing the Right Physics Engine

| Engine | Strengths | Weaknesses | Best Use Cases |
|--------|-----------|------------|----------------|
| ODE | Stable, well-tested | Limited features | Basic simulations |
| Bullet | Fast, stable | Less accurate for complex contacts | Real-time applications |
| DART | Advanced constraints, humanoid-friendly | More complex | Humanoid robots, complex mechanisms |
| PhysX | High fidelity, GPU acceleration | Proprietary | High-quality graphics applications |

## Physics Configuration in Gazebo

### World Physics Configuration

Physics parameters for a Gazebo world are defined in the SDF file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physics_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Model Physics Configuration

Physics properties for individual models are defined in their URDF/SDF:

```xml
<link name="wheel_link">
  <!-- Visual properties -->
  <visual>
    <geometry>
      <cylinder radius="0.1" length="0.05"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <geometry>
      <cylinder radius="0.1" length="0.05"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0"/>
    <inertia
      ixx="0.001" ixy="0.0" ixz="0.0"
      iyy="0.001" iyz="0.0"
      izz="0.002"/>
  </inertial>
</link>

<!-- Joint with dynamics -->
<joint name="wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0 0.2 -0.05" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>
```

### Surface Properties

Surface properties define how objects interact when they contact each other:

```xml
<collision name="wheel_collision">
  <geometry>
    <cylinder radius="0.1" length="0.05"/>
  </geometry>

  <!-- Surface properties -->
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
        <fdir1>0 0 1</fdir1>
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1e+13</kp>
        <kd>1.0</kd>
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

## Physics Configuration in Unity

### Rigidbody Components

In Unity, physics properties are managed through Rigidbody components:

```csharp
using UnityEngine;

public class RobotPart : MonoBehaviour
{
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Set mass
        rb.mass = 1.0f;

        // Set drag and angular drag
        rb.drag = 0.1f;
        rb.angularDrag = 0.05f;

        // Configure other properties
        rb.useGravity = true;
        rb.isKinematic = false;  // Set to true to disable physics simulation

        // Freeze certain axes if needed
        rb.constraints = RigidbodyConstraints.FreezeRotationX |
                        RigidbodyConstraints.FreezeRotationZ;
    }
}
```

### Physic Materials

Unity uses Physic Materials to define surface properties:

```csharp
// Create a PhysicMaterial in code
PhysicMaterial wheelMaterial = new PhysicMaterial();
wheelMaterial.staticFriction = 0.8f;
wheelMaterial.dynamicFriction = 0.5f;
wheelMaterial.bounciness = 0.1f;
wheelMaterial.frictionCombine = PhysicMaterialCombine.Maximum;
wheelMaterial.bounceCombine = PhysicMaterialCombine.Average;

// Apply to a collider
Collider wheelCollider = GetComponent<Collider>();
wheelCollider.material = wheelMaterial;
```

### Joint Components

Unity provides various joint components for connecting rigid bodies:

```csharp
using UnityEngine;

public class RobotJoint : MonoBehaviour
{
    public ConfigurableJoint joint;

    void Start()
    {
        joint = GetComponent<ConfigurableJoint>();

        // Configure joint limits
        SoftJointLimit limit = new SoftJointLimit();
        limit.limit = 45f;  // 45 degrees
        joint.lowAngularXLimit = limit;
        joint.highAngularXLimit = limit;

        // Configure spring/damper
        joint.xDrive = new JointDrive
        {
            mode = JointDriveMode.Position,
            positionSpring = 10000f,
            positionDamper = 100f
        };
    }
}
```

## Tuning Physics Parameters

### Simulation Accuracy vs Performance

Physics simulation involves a trade-off between accuracy and performance:

- **Smaller time steps**: More accurate but slower
- **Higher solver iterations**: More accurate contacts but slower
- **More complex collision geometry**: More accurate but slower

### Parameter Tuning Guidelines

1. **Time Step Size**: Start with 0.001s, increase if stable
2. **Solver Iterations**: Start with 10-20, increase if contacts are unstable
3. **Real-time Factor**: Set based on required simulation speed
4. **Contact Parameters**: Adjust ERP and CFM for stable contacts

### Validation Techniques

To validate physics simulation:

1. **Compare with analytical solutions** for simple cases
2. **Test with real robot** when available
3. **Run multiple simulations** to check consistency
4. **Monitor energy conservation** in closed systems

## Advanced Physics Concepts

### Soft Body Simulation

For flexible parts or soft robots:

```xml
<!-- In Gazebo, this requires special plugins or finite element methods -->
<!-- Example with a simplified approach using multiple rigid bodies -->
<model name="soft_arm">
  <link name="segment_1">
    <inertial><mass value="0.1"/><inertia .../></inertial>
    <visual><geometry><cylinder radius="0.02" length="0.1"/></geometry></visual>
  </link>
  <link name="segment_2">
    <inertial><mass value="0.1"/><inertia .../></inertial>
    <visual><geometry><cylinder radius="0.02" length="0.1"/></geometry></visual>
  </link>
  <!-- Connect with flexible joint -->
  <joint name="flex_joint" type="revolute">
    <parent link="segment_1"/>
    <child link="segment_2"/>
    <dynamics damping="5.0" friction="1.0"/>
  </joint>
</model>
```

### Fluid Simulation

For underwater or aerial robots, fluid dynamics can be approximated:

```xml
<!-- Simplified fluid drag in URDF -->
<transmission name="wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- In simulation, add drag forces through plugins -->
<gazebo>
  <plugin name="hydrodynamics" filename="libhydrodynamics.so">
    <linear_damping>0.1</linear_damping>
    <angular_damping>0.05</angular_damping>
  </plugin>
</gazebo>
```

## Best Practices for Physics Simulation

### Model Accuracy

1. **Use realistic masses and inertias**: Calculate from actual dimensions and materials
2. **Include friction**: Real robots experience friction at all contact points
3. **Model compliance**: Consider flexibility in joints and structures
4. **Validate parameters**: Test with simple scenarios before complex ones

### Performance Optimization

1. **Simplify collision geometry**: Use boxes instead of complex meshes for collision
2. **Adjust update rates**: Match physics rate to controller requirements
3. **Use appropriate solvers**: Choose based on simulation needs
4. **Limit active objects**: Deactivate physics for distant objects

### Simulation Fidelity

1. **Understand limitations**: No simulation perfectly matches reality
2. **Account for the "reality gap"**: Plan for differences between sim and real
3. **Domain randomization**: Vary parameters to improve robustness
4. **Systematic validation**: Test simulation outputs against known behaviors

## Exercises

1. **Physics Parameter Tuning**: Create a simple pendulum simulation and tune parameters for stable behavior
2. **Collision Validation**: Compare simulation results with analytical solutions for a falling object
3. **Friction Modeling**: Implement a wheeled robot with realistic friction parameters
4. **Energy Analysis**: Create a simulation that demonstrates energy conservation

## Summary

Physics simulation is fundamental to creating realistic digital twins for robotics. By properly configuring physics parameters, understanding the trade-offs between accuracy and performance, and validating simulation behavior, you can create simulation environments that effectively support robot development and testing. The choice of physics engine and parameters should align with your specific application requirements.

## References

- [Gazebo Physics Documentation](http://gazebosim.org/tutorials?tut=physics)
- [Unity Physics Manual](https://docs.unity3d.com/Manual/PhysicsSection.html)
- [ROS 2 Gazebo Integration](https://github.com/ros-simulation/gazebo_ros_pkgs)
- [Physics-Based Robot Simulation](https://arxiv.org/abs/2008.12528)
