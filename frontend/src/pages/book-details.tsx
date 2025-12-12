import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './book-details.module.css';

function BookOverview() {
  return (
    <section className={styles.bookOverview}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <div className={styles.bookHeader}>
              <div className={styles.bookIcon}>🤖</div>
              <Heading as="h1" className={styles.bookTitle}>
                Physical AI & Humanoid Robotics
              </Heading>
            </div>
            <p className={styles.bookDescription}>
              A comprehensive AI-native textbook covering ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems. 
              Master the fundamentals of robotics, autonomous systems, and intelligent machine interaction.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModuleCard({ title, description, items }: { title: string, description: string, items: string[] }) {
  return (
    <div className={`col col--6 ${styles.moduleCard}`}>
      <div className={styles.card}>
        <h3>{title}</h3>
        <p>{description}</p>
        <ul className={styles.moduleList}>
          {items.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function BookModules() {
  return (
    <section className={styles.bookModules}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Book Modules</h2>
          </div>
        </div>

        <div className="row" style={{ marginTop: '2rem' }}>
          <ModuleCard
            title="Module 1: ROS 2 (Weeks 3-5)"
            description="Robotic Nervous System – Middleware for Robot Control"
            items={[
              "Introduction to ROS 2",
              "Nodes and Topics",
              "Services, Actions, and Parameters",
              "URDF Robot Modeling",
              "Launch Files and Package Management"
            ]}
          />

          <ModuleCard
            title="Module 2: Digital Twin (Weeks 6-7)"
            description="Physics Simulation & Environment Building"
            items={[
              "Introduction to Gazebo & Unity",
              "Simulating Physics, Gravity, and Collisions",
              "Sensor Simulation: LiDAR, Depth Cameras, IMUs",
              "High-fidelity Rendering & Human-Robot Interaction"
            ]}
          />

          <ModuleCard
            title="Module 3: NVIDIA Isaac (Weeks 8-10)"
            description="AI-Robot Brain – Advanced Perception and Training"
            items={[
              "NVIDIA Isaac Sim Overview",
              "Hardware-accelerated VSLAM (Isaac ROS)",
              "Navigation & Path Planning (Nav2)",
              "AI-powered Perception & Manipulation",
              "Reinforcement Learning and Sim-to-Real Techniques"
            ]}
          />

          <ModuleCard
            title="Module 4: VLA & Humanoids (Weeks 11-13)"
            description="Convergence of LLMs and Robotics"
            items={[
              "Humanoid Robot Development (Kinematics, Dynamics, Bipedal Locomotion)",
              "Manipulation and Grasping with Humanoid Hands",
              "Natural Human-Robot Interaction Design",
              "Conversational Robotics (GPT Integration, Whisper Voice-to-Action, Multi-modal Interaction)",
              "Capstone Project: Autonomous Humanoid"
            ]}
          />
        </div>
      </div>
    </section>
  );
}

export default function BookDetails(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics - ${siteConfig.tagline}`}
      description="Detailed overview of the Physical AI & Humanoid Robotics textbook">
      <main>
        <BookOverview />
        <BookModules />
        <div className="container padding-vert--lg">
          <div className="row">
            <div className="col col--12 text--center">
              <Link
                className="button button--primary button--lg"
                to="/docs/introduction">
                Start Learning
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}