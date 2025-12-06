---
title: Reinforcement Learning
description: Learn about reinforcement learning for robotics using NVIDIA Isaac and GPU acceleration
sidebar_position: 5
---

# Reinforcement Learning

## Learning Objectives

- Understand reinforcement learning fundamentals for robotics
- Learn about GPU-accelerated RL training and inference
- Explore Isaac Lab for robot learning research
- Implement RL algorithms for robotic tasks
- Evaluate RL performance and transfer to real robots

## Prerequisites

- Understanding of AI-powered perception (Chapter 4)
- Basic knowledge of machine learning concepts
- ROS 2 environment setup completed

## Introduction to Reinforcement Learning in Robotics

Reinforcement Learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In robotics, RL enables robots to learn complex behaviors and skills through trial and error.

### Key RL Concepts

- **Agent**: The learning entity (the robot)
- **Environment**: The world the agent interacts with
- **State**: Current situation or configuration
- **Action**: Decision made by the agent
- **Reward**: Feedback signal for the agent's actions
- **Policy**: Strategy for selecting actions

### RL in Robotics Applications

1. **Manipulation**: Learning grasping and manipulation skills
2. **Locomotion**: Learning walking and movement patterns
3. **Navigation**: Learning path planning and obstacle avoidance
4. **Control**: Learning optimal control policies
5. **Human-Robot Interaction**: Learning social behaviors

## RL Algorithms for Robotics

### Deep Q-Network (DQN)

DQN combines Q-learning with deep neural networks:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr

        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Actor-Critic Methods

Actor-critic methods combine policy-based and value-based approaches:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor (policy) network
        self.actor = nn.Linear(hidden_size, action_size)

        # Critic (value) network
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.shared_layers(state)

        # Actor output (logits for action probabilities)
        action_logits = self.actor(features)

        # Critic output (state value)
        state_value = self.critic(features)

        return action_logits, state_value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size

        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action_and_value(self, state):
        """Get action and state value from the model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits, state_value = self.model(state_tensor)

        # Sample action from policy
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()

        return action, state_value.item(), action_probs.squeeze().detach().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        """Update the model using actor-critic algorithm"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)

        # Get current action logits and state values
        action_logits, state_values = self.model(states_tensor)

        # Calculate next state values for target computation
        _, next_state_values = self.model(next_states_tensor)
        next_state_values = next_state_values.detach()

        # Calculate target values
        target_values = rewards_tensor + 0.99 * next_state_values.squeeze() * (~dones_tensor)

        # Calculate advantages
        advantages = target_values - state_values.squeeze()

        # Actor loss (policy gradient)
        action_probs = F.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        log_probs = torch.log(selected_action_probs + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = F.mse_loss(state_values.squeeze(), target_values)

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### Proximal Policy Optimization (PPO)

PPO is a popular policy gradient method that ensures stable updates:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPOActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value

    def get_action(self, state):
        """Sample action from policy"""
        action_logits, _ = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        return action, log_prob

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        action_logits, state_values = self.forward(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        return log_probs.squeeze(), state_values.squeeze(), entropy

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, clip_epsilon=0.2, epochs=4):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

        self.model = PPOActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update PPO policy"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)

        for _ in range(self.epochs):
            # Get new action probabilities and values
            new_log_probs, state_values, entropy = self.model.evaluate_actions(
                states_tensor, actions_tensor)

            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # Calculate surrogate objectives
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor

            # Actor loss (PPO objective)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(state_values, returns_tensor)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
```

## GPU-Accelerated RL with Isaac Lab

### Isaac Lab Overview

Isaac Lab provides a comprehensive framework for robot learning research with GPU acceleration:

```python
import omni
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.envs.mdp import observations, rewards, terminations
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for the cartpole scene."""

    # Define the robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn_func_name="spawn_cartpole",
        init_state={
            "joint_pos": {"slider_to_cart": 0.0, "cart_to_pole": 0.0},
            "joint_vel": {"slider_to_cart": 0.0, "cart_to_pole": 0.0},
        },
        actuators={
            "cart_actuator": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"],
                effort_limit=400.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=10.0,
            )
        },
    )

    # Define contact sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.005,
        history_length=3,
        track_contact_forces=True,
        visualize=False,
    )

class CartpoleEnv(ManagerBasedRLEnv):
    """Simple cartpole environment for RL training."""

    def __init__(self, cfg):
        super().__init__(cfg)

        # Initialize episode counters
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _get_observations(self) -> dict:
        """Get observations for the environment."""
        # Get joint positions and velocities
        joint_pos = self.scene["robot"].data.joint_pos_w
        joint_vel = self.scene["robot"].data.joint_vel_w

        # Return observations
        return {
            "policy": torch.cat([joint_pos, joint_vel], dim=1)
        }

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the environment."""
        # Calculate pole angle (second joint is the pole)
        pole_angle = self.scene["robot"].data.joint_pos_w[:, 1]
        pole_vel = self.scene["robot"].data.joint_vel_w[:, 1]

        # Reward for keeping pole upright
        reward = torch.exp(-pole_angle**2) - 0.1 * pole_vel**2

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get dones and time outs for the environment."""
        # Terminate if pole angle is too large
        pole_angle = self.scene["robot"].data.joint_pos_w[:, 1]
        terminated = torch.abs(pole_angle) > 1.0

        # Time out after max episode length
        time_out = self.episode_length_buf >= 500
        truncated = time_out

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with IDs in env_ids."""
        # Randomize initial joint positions and velocities
        joint_pos = self.scene["robot"].data.default_joint_pos[env_ids].clone()
        joint_pos[:, 1] += torch.randn_like(joint_pos[:, 1]) * 0.1  # Add small random angle

        joint_vel = torch.zeros_like(self.scene["robot"].data.joint_vel_w[env_ids])

        # Set new joint states
        self.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

        # Reset episode counters
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0.0
```

### GPU-Accelerated Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time

class GPUAcceleratedRLTrainer:
    def __init__(self, agent, env, device='cuda'):
        self.agent = agent
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir='runs/rl_experiment')

        # Training parameters
        self.max_episodes = 1000
        self.max_steps_per_episode = 1000
        self.update_frequency = 1000  # Update every 1000 steps

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=100000)

        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def train(self):
        """Main training loop with GPU acceleration"""
        total_steps = 0

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.max_steps_per_episode):
                # Convert state to tensor and move to GPU
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get action from agent
                action = self.agent.act(state_tensor)

                # Take action in environment
                next_state, reward, done, info = self.env.step(action)

                # Store experience in replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))

                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # Train agent periodically
                if len(self.replay_buffer) > 32 and total_steps % self.update_frequency == 0:
                    self.train_agent_batch()

                if done:
                    break

            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()

            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Log to tensorboard
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)

                self.writer.add_scalar('Episode/Reward', avg_reward, episode)
                self.writer.add_scalar('Episode/Length', avg_length, episode)
                self.writer.add_scalar('Training/Steps', total_steps, episode)

                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.2f}, Total Steps = {total_steps}")

            # Save model periodically
            if episode % 100 == 0:
                self.save_model(f"model_checkpoint_{episode}.pth")

    def train_agent_batch(self):
        """Train agent on a batch of experiences"""
        # Sample batch from replay buffer
        batch_size = 32
        if len(self.replay_buffer) < batch_size:
            return

        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Extract batch components
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)

        # Train the agent
        self.agent.replay(states, actions, rewards, next_states, dones)

    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.agent.q_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards)
        }, path)

    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
```

## Isaac Lab for Advanced RL

### Environment Configuration

```python
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ManagerTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer import OffsetCfg
from omni.isaac.lab.utils import configclass

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reaching task environment."""

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5)

    # Define the robot (e.g., Franka Emika Panda)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn_func_name="spawn_franka",
        init_state={
            "joint_pos": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
            },
        },
    )

    # Define target object
    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn_func_name="spawn_sphere",
        init_state={
            "pos": [0.5, 0.0, 0.5],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        spawn_kwargs={"radius": 0.05, "color": (0.9, 0.2, 0.2)},
    )

    # Frame transformer for end-effector position
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FrameTransformerCfg.OffsetCfg(
            prim_path="/Visuals/EEFrameTransformer",
            mark_size=0.02,
            line_width=3,
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                    rot=[0.707, 0.0, 0.707, 0.0],
                ),
            )
        ],
    )

    def __post_init__(self):
        """Post initialization."""
        # Set the physics simulation step
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 2

        # Update the scene
        self.scene.num_envs = 1024  # Number of parallel environments
        self.scene.env_spacing = 2.5
```

### Curriculum Learning

```python
class CurriculumLearning:
    """Implementation of curriculum learning for RL"""

    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0, threshold=0.8):
        self.difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.threshold = threshold
        self.performance_history = deque(maxlen=100)

    def update_difficulty(self, current_performance):
        """Update task difficulty based on performance"""
        self.performance_history.append(current_performance)

        # Calculate average performance
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0

        # Adjust difficulty based on performance
        if avg_performance > self.threshold and self.difficulty < self.max_difficulty:
            # Increase difficulty if performing well
            self.difficulty = min(self.difficulty * 1.1, self.max_difficulty)
        elif avg_performance < self.threshold * 0.7 and self.difficulty > 0.1:
            # Decrease difficulty if struggling
            self.difficulty = max(self.difficulty * 0.9, 0.1)

        return self.difficulty

    def get_task_parameters(self):
        """Get current task parameters based on difficulty"""
        # Example: vary target distance based on difficulty
        min_distance = 0.1
        max_distance = 1.0
        current_distance = min_distance + self.difficulty * (max_distance - min_distance)

        return {
            'target_distance': current_distance,
            'success_threshold': 0.05 / self.difficulty,  # Tighter tolerance for higher difficulty
            'time_limit': int(100 * (2 - self.difficulty))  # Shorter time limit for higher difficulty
        }
```

## Domain Randomization and Transfer Learning

### Domain Randomization

```python
import random

class DomainRandomization:
    """Apply domain randomization to improve sim-to-real transfer"""

    def __init__(self, randomization_params):
        self.params = randomization_params

    def randomize_environment(self, env):
        """Randomize environment parameters"""
        # Randomize object properties
        if 'object_mass' in self.params:
            mass_range = self.params['object_mass']
            new_mass = random.uniform(mass_range[0], mass_range[1])
            env.set_object_mass(new_mass)

        # Randomize friction coefficients
        if 'friction' in self.params:
            friction_range = self.params['friction']
            new_friction = random.uniform(friction_range[0], friction_range[1])
            env.set_friction(new_friction)

        # Randomize visual properties
        if 'lighting' in self.params:
            lighting_range = self.params['lighting']
            new_lighting = random.uniform(lighting_range[0], lighting_range[1])
            env.set_lighting_intensity(new_lighting)

        # Randomize camera properties
        if 'camera_noise' in self.params:
            noise_level = random.uniform(0, self.params['camera_noise'])
            env.set_camera_noise(noise_level)

# Example randomization parameters
randomization_params = {
    'object_mass': [0.5, 2.0],  # Random mass between 0.5 and 2.0 kg
    'friction': [0.1, 1.0],     # Random friction coefficient
    'lighting': [0.5, 2.0],     # Random lighting intensity
    'camera_noise': 0.1         # Maximum camera noise
}

domain_rand = DomainRandomization(randomization_params)
```

### Transfer Learning

```python
class TransferLearning:
    """Transfer learning from simulation to real robot"""

    def __init__(self, sim_model_path, real_robot_interface):
        self.sim_model = self.load_model(sim_model_path)
        self.real_robot = real_robot_interface

    def adapt_model(self, real_data):
        """Adapt simulation model to real robot using limited real data"""
        # Fine-tune model with real data
        real_states, real_actions = real_data

        # Convert real data to appropriate format
        real_states_tensor = torch.FloatTensor(real_states)
        real_actions_tensor = torch.LongTensor(real_actions)

        # Fine-tune the policy network
        optimizer = torch.optim.Adam(self.sim_model.parameters(), lr=1e-5)

        for epoch in range(10):  # Small number of epochs to avoid overfitting
            optimizer.zero_grad()

            # Get predictions from model
            action_logits, _ = self.sim_model(real_states_tensor)
            loss = nn.CrossEntropyLoss()(action_logits, real_actions_tensor)

            loss.backward()
            optimizer.step()

            print(f"Fine-tuning epoch {epoch}, loss: {loss.item():.4f}")

    def collect_real_data(self, num_episodes=10):
        """Collect data from real robot for adaptation"""
        real_data = []

        for episode in range(num_episodes):
            state = self.real_robot.reset()
            episode_data = []

            while not self.real_robot.is_done():
                # Get action from current policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = self.sim_model.get_action_and_value(state_tensor)

                # Execute action on real robot
                next_state, reward, done, info = self.real_robot.step(action)

                # Store transition
                episode_data.append((state, action, reward, next_state, done))
                state = next_state

            real_data.extend(episode_data)

        return real_data
```

## Performance Evaluation and Metrics

### RL Performance Metrics

```python
class RLEvaluationMetrics:
    """Comprehensive evaluation metrics for RL agents"""

    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'convergence_speed': [],
            'sample_efficiency': [],
            'stability': []
        }

    def evaluate_agent(self, agent, env, num_episodes=100):
        """Evaluate agent performance across multiple metrics"""
        total_reward = 0
        total_steps = 0
        successes = 0

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                # Get action from agent
                if hasattr(agent, 'act'):
                    action = agent.act(state)
                else:
                    # For Isaac Lab agents
                    action = agent.compute_single_action(state)

                # Take step in environment
                state, reward, done, info = env.step(action)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Check for success (task-specific)
                if info.get('success', False):
                    successes += 1
                    break

            total_reward += episode_reward

            # Store episode metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_steps)

        # Calculate aggregate metrics
        avg_reward = total_reward / num_episodes
        success_rate = successes / num_episodes
        avg_length = np.mean(self.metrics['episode_lengths'])

        # Calculate additional metrics
        self.metrics['success_rates'].append(success_rate)

        # Stability metric (variance of rewards)
        reward_variance = np.var(self.metrics['episode_rewards'])
        self.metrics['stability'].append(1.0 / (1.0 + reward_variance))  # Lower variance = higher stability

        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length,
            'stability': self.metrics['stability'][-1]
        }

    def calculate_sample_efficiency(self, rewards_over_time):
        """Calculate how efficiently the agent learns"""
        # Calculate area under the learning curve
        total_area = np.trapz(rewards_over_time)
        max_possible_area = len(rewards_over_time) * np.max(rewards_over_time)

        efficiency = total_area / max_possible_area if max_possible_area > 0 else 0
        return efficiency

    def get_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not self.metrics['episode_rewards']:
            return "No evaluation data available"

        report = {
            'avg_reward': np.mean(self.metrics['episode_rewards']),
            'std_reward': np.std(self.metrics['episode_rewards']),
            'min_reward': np.min(self.metrics['episode_rewards']),
            'max_reward': np.max(self.metrics['episode_rewards']),
            'avg_length': np.mean(self.metrics['episode_lengths']),
            'success_rate': np.mean(self.metrics['success_rates']) if self.metrics['success_rates'] else 0,
            'stability': np.mean(self.metrics['stability']) if self.metrics['stability'] else 0
        }

        return report
```

## Integration with ROS 2

### RL Node Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
import torch
import numpy as np

class RLControlNode(Node):
    def __init__(self):
        super().__init__('rl_control_node')

        # Initialize RL agent
        self.rl_agent = None
        self.load_rl_model()

        # Subscribers for sensor data
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/ee_pose', self.pose_callback, 10)

        # Publishers for commands and status
        self.action_pub = self.create_publisher(JointState, '/rl_actions', 10)
        self.reward_pub = self.create_publisher(Float32, '/rl_reward', 10)
        self.status_pub = self.create_publisher(Bool, '/rl_running', 10)

        # Services
        self.reset_srv = self.create_service(Empty, 'rl_reset', self.reset_callback)
        self.train_srv = self.create_service(Empty, 'rl_train', self.train_callback)

        # Robot state
        self.current_joint_positions = None
        self.current_pose = None
        self.rl_running = False

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    def load_rl_model(self):
        """Load pre-trained RL model"""
        try:
            # Load the trained model
            model_path = self.get_parameter_or('model_path', 'rl_model.pth').value
            self.rl_agent = torch.load(model_path)
            self.rl_agent.eval()  # Set to evaluation mode
            self.get_logger().info(f'RL model loaded from {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load RL model: {e}')

    def joint_state_callback(self, msg):
        """Update joint state from robot"""
        self.current_joint_positions = np.array(msg.position)

    def pose_callback(self, msg):
        """Update end-effector pose"""
        self.current_pose = np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])

    def control_loop(self):
        """Main control loop for RL agent"""
        if not self.rl_running or self.rl_agent is None:
            return

        if self.current_joint_positions is None or self.current_pose is None:
            return

        # Construct state from sensor data
        state = self.construct_state()

        # Get action from RL agent
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.rl_agent.act(state_tensor).cpu().numpy()

        # Publish action
        action_msg = JointState()
        action_msg.position = action.tolist()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        self.action_pub.publish(action_msg)

        # Calculate and publish reward
        reward = self.calculate_reward()
        reward_msg = Float32()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)

    def construct_state(self):
        """Construct state vector from sensor data"""
        # Combine joint positions, velocities, and pose
        # This is a simplified example - actual state construction depends on task
        state = np.concatenate([
            self.current_joint_positions,
            self.current_pose[:3]  # Just position, not orientation for simplicity
        ])
        return state

    def calculate_reward(self):
        """Calculate reward based on current state"""
        # Example: reward for reaching a target position
        target_pos = np.array([0.5, 0.0, 0.5])  # Example target
        current_pos = self.current_pose[:3]

        distance = np.linalg.norm(target_pos - current_pos)
        reward = np.exp(-distance)  # Higher reward for closer to target

        return reward

    def reset_callback(self, request, response):
        """Reset the RL environment"""
        # Reset robot to initial position
        # Implementation depends on specific robot
        self.get_logger().info('RL environment reset')
        return response

    def train_callback(self, request, response):
        """Start/stop RL training"""
        self.rl_running = not self.rl_running
        status_msg = Bool()
        status_msg.data = self.rl_running
        self.status_pub.publish(status_msg)

        status = "started" if self.rl_running else "stopped"
        self.get_logger().info(f'RL training {status}')
        return response

def main(args=None):
    rclpy.init(args=args)
    rl_node = RLControlNode()

    try:
        rclpy.spin(rl_node)
    except KeyboardInterrupt:
        pass
    finally:
        rl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for RL in Robotics

### Training Strategies

1. **Simulation-to-Real Transfer**: Use domain randomization and sim-to-real techniques
2. **Hierarchical RL**: Break complex tasks into simpler sub-tasks
3. **Multi-task Learning**: Train on multiple related tasks simultaneously
4. **Curriculum Learning**: Start with simple tasks and gradually increase difficulty

### Hardware Considerations

1. **GPU Utilization**: Maximize GPU usage with batched training
2. **Memory Management**: Efficiently manage GPU memory for large models
3. **Parallel Environments**: Use multiple parallel environments for sample efficiency
4. **Mixed Precision**: Use FP16 training where possible for speed

### Safety and Reliability

1. **Constraint Satisfaction**: Ensure actions satisfy physical constraints
2. **Safety Filters**: Implement safety filters to prevent dangerous actions
3. **Robustness Testing**: Test agent behavior under various conditions
4. **Fallback Policies**: Implement traditional control methods as fallback

## Exercises

1. **PPO Implementation**: Implement and train a PPO agent on a robotic task
2. **Simulation Training**: Train an RL agent in Isaac Sim and evaluate performance
3. **Domain Randomization**: Apply domain randomization to improve sim-to-real transfer
4. **Hardware Acceleration**: Optimize RL training using GPU acceleration

## Summary

Reinforcement learning with NVIDIA Isaac provides powerful tools for training robots to perform complex tasks through interaction with their environment. By leveraging GPU acceleration and simulation environments, we can efficiently train policies that can transfer to real robots. Understanding the algorithms, implementation techniques, and best practices is crucial for developing effective RL-based robotic systems.

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/source/index.html)
- [Reinforcement Learning in Robotics](https://arxiv.org/abs/2103.13073)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Sim-to-Real Transfer](https://arxiv.org/abs/2006.12884)

## Next Steps

[← Previous Chapter: AI-Powered Perception](./chapter-4-ai-powered-perception) | [Next Module: VLA & Humanoids →](../module-4-vla-humanoids/chapter-1-humanoid-robot-development)