import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p className="hero__description">
          A comprehensive AI-native textbook on Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/introduction">
            Start Learning - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

function CourseOverview() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--4">
            <h2>Module 1: ROS 2 (Weeks 3-5)</h2>
            <p>Robotic Nervous System – Middleware for Robot Control</p>
            <ul>
              <li>Introduction to ROS 2</li>
              <li>Nodes and Topics</li>
              <li>Services, Actions, and Parameters</li>
              <li>URDF Robot Modeling</li>
              <li>Launch Files and Package Management</li>
            </ul>
          </div>
          <div className="col col--4">
            <h2>Module 2: Digital Twin (Weeks 6-7)</h2>
            <p>Physics Simulation & Environment Building</p>
            <ul>
              <li>Introduction to Gazebo & Unity</li>
              <li>Simulating Physics, Gravity, and Collisions</li>
              <li>Sensor Simulation: LiDAR, Depth Cameras, IMUs</li>
              <li>High-fidelity Rendering & Human-Robot Interaction</li>
            </ul>
          </div>
          <div className="col col--4">
            <h2>Module 3: NVIDIA Isaac (Weeks 8-10)</h2>
            <p>AI-Robot Brain – Advanced Perception and Training</p>
            <ul>
              <li>NVIDIA Isaac Sim Overview</li>
              <li>Hardware-accelerated VSLAM (Isaac ROS)</li>
              <li>Navigation & Path Planning (Nav2)</li>
              <li>AI-powered Perception & Manipulation</li>
              <li>Reinforcement Learning and Sim-to-Real Techniques</li>
            </ul>
          </div>
        </div>
        <div className="row" style={{marginTop: '2rem'}}>
          <div className="col col--4">
            <h2>Module 4: VLA & Humanoids (Weeks 11-13)</h2>
            <p>Convergence of LLMs and Robotics</p>
            <ul>
              <li>Humanoid Robot Development (Kinematics, Dynamics, Bipedal Locomotion)</li>
              <li>Manipulation and Grasping with Humanoid Hands</li>
              <li>Natural Human-Robot Interaction Design</li>
              <li>Conversational Robotics (GPT Integration, Whisper Voice-to-Action, Multi-modal Interaction)</li>
              <li>Capstone Project: Autonomous Humanoid (Full Simulation with Voice Commands, Path Planning, Object Identification, Manipulation)</li>
            </ul>
          </div>
          <div className="col col--4">
            <h2>Learning Outcomes</h2>
            <ul>
              <li>Master ROS 2 for humanoid robot control</li>
              <li>Build digital twins with Gazebo and Unity</li>
              <li>Implement AI-powered perception and navigation</li>
              <li>Develop vision-language-action systems</li>
              <li>Create conversational robotics applications</li>
            </ul>
          </div>
          <div className="col col--4">
            <h2>Hardware Requirements</h2>
            <ul>
              <li>High-performance workstation: RTX 4070 Ti+ GPU, Intel i7/AMD Ryzen 9 CPU, 64GB RAM, Ubuntu 22.04</li>
              <li>Physical AI Edge Kit: Jetson Orin Nano/NX, RealSense camera, USB IMU, ReSpeaker mic array</li>
              <li>Robot Lab options: Proxy Approach, Miniature Humanoid, Premium Lab</li>
              <li>Cloud/Hybrid lab options: AWS/Azure NVIDIA Isaac Sim cloud</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

function CapstoneProject() {
  return (
    <section className={styles.capstone}>
      <div className="container padding-vert--lg">
        <div className="row">
          <div className="col col--12">
            <h2>Capstone Project: Autonomous Humanoid Robot</h2>
            <p>
              Integrate all concepts learned across the four modules to create an autonomous humanoid robot capable of
              voice commands, path planning, object identification, and manipulation in both simulation and real-world environments.
            </p>
            <div className="text--center padding-vert--md">
              <Link
                className="button button--primary button--lg"
                to="/docs/capstone">
                Explore Capstone Project
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A comprehensive AI-native textbook on Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems">
      <HomepageHeader />
      <main>
        <CourseOverview />
        <CapstoneProject />
      </main>
    </Layout>
  );
}
