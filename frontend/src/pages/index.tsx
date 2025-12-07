import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
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
            Start Learning →
          </Link>
          {/* <Link
            className="button button--primary button--lg"
            to="/docs/hardware-requirements">
            Hardware Requirements
          </Link> */}
        </div>
      </div>
    </header>
  );
}

function Card({ title, description, items, to, buttonLabel }: { title: string, description: string, items: string[], to: string, buttonLabel: string }) {
  return (
    <div className={`col col--4 ${styles.moduleCard}`}>
      <div className={styles.card}>
        <Link to={to}>
          <h3>{title}</h3>
        </Link>
        <p>{description}</p>
        <ul>
          {items.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ModuleCards() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>Course Modules</h2>
          </div>
        </div>

        <div className="row">
          <Card
            title="Module 1: ROS 2 (Weeks 3-5)"
            description="Robotic Nervous System – Middleware for Robot Control"
            items={[
              "Introduction to ROS 2",
              "Nodes and Topics",
              "Services, Actions, and Parameters",
              "URDF Robot Modeling",
              "Launch Files and Package Management"
            ]}
            to="/docs/module-1-ros2"
            buttonLabel="Explore Module"
          />

          <Card
            title="Module 2: Digital Twin (Weeks 6-7)"
            description="Physics Simulation & Environment Building"
            items={[
              "Introduction to Gazebo & Unity",
              "Simulating Physics, Gravity, and Collisions",
              "Sensor Simulation: LiDAR, Depth Cameras, IMUs",
              "High-fidelity Rendering & Human-Robot Interaction"
            ]}
            to="/docs/module-2-digital-twin"
            buttonLabel="Explore Module"
          />

          <Card
            title="Module 3: NVIDIA Isaac (Weeks 8-10)"
            description="AI-Robot Brain – Advanced Perception and Training"
            items={[
              "NVIDIA Isaac Sim Overview",
              "Hardware-accelerated VSLAM (Isaac ROS)",
              "Navigation & Path Planning (Nav2)",
              "AI-powered Perception & Manipulation",
              "Reinforcement Learning and Sim-to-Real Techniques"
            ]}
            to="/docs/module-3-nvidia-isaac"
            buttonLabel="Explore Module"
          />
        </div>

        <div className="row" style={{ marginTop: '2rem' }}>
          <Card
            title="Module 4: VLA & Humanoids (Weeks 11-13)"
            description="Convergence of LLMs and Robotics"
            items={[
              "Humanoid Robot Development (Kinematics, Dynamics, Bipedal Locomotion)",
              "Manipulation and Grasping with Humanoid Hands",
              "Natural Human-Robot Interaction Design",
              "Conversational Robotics (GPT Integration, Whisper Voice-to-Action, Multi-modal Interaction)",
              "Capstone Project: Autonomous Humanoid"
            ]}
            to="/docs/module-4-vla-humanoids"
            buttonLabel="Explore Module"
          />

          <div className="col col--4">
            <div className={`${styles.card} ${styles.learningOutcomes}`}>
              <h3>Learning Outcomes</h3>
              <ul>
                <li>Master ROS 2 for humanoid robot control</li>
                <li>Build digital twins with Gazebo and Unity</li>
                <li>Implement AI-powered perception and navigation</li>
                <li>Develop vision-language-action systems</li>
                <li>Create conversational robotics applications</li>
              </ul>
            </div>
          </div>

          <Card
            title="Hardware Requirements"
            description="Essential setup for learning and development"
            items={[
              "High-performance workstation: RTX 4070 Ti+ GPU, Intel i7/AMD Ryzen 9 CPU, 64GB RAM, Ubuntu 22.04",
              "Physical AI Edge Kit: Jetson Orin Nano/NX, RealSense camera, USB IMU, ReSpeaker mic array",
              "Robot Lab options: Proxy Approach, Miniature Humanoid, Premium Lab",
              "Cloud/Hybrid lab options: AWS/Azure NVIDIA Isaac Sim cloud"
            ]}
            to="/docs/hardware-requirements"
            buttonLabel="View Requirements"
          />
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
            <h2 style={{ textAlign: 'center' }}>Capstone Project: Autonomous Humanoid Robot</h2>
            <p style={{ textAlign: 'center', fontSize: '1.2rem', maxWidth: '800px', margin: '0 auto' }}>
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
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A comprehensive AI-native textbook on Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems">
      <HomepageHeader />
      <main>
        <ModuleCards />
        <CapstoneProject />
      </main>
    </Layout>
  );
}
