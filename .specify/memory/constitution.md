<!--
Sync Impact Report:
Version change: 1.0.0 → 1.0.1
Added sections: Detailed textbook structure with weekly breakdown, specific hardware requirements, enhanced tech stack details
Modified principles: Content Accuracy & Educational Clarity (renamed from Educational Excellence & Technical Accuracy), added Code Quality & Versioning Standards, Docusaurus Structure & Accessibility (revised), updated Hardware-Aware Implementation
Modified templates: N/A (new project)
Templates requiring updates: ⚠ .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md (pending alignment)
Follow-up TODOs: Update templates to align with new constitution
-->
# Neuro Library: Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### Content Accuracy & Educational Clarity
All content must be educationally sound and technically accurate; Formulas, physics, algorithms, and code must be correct and verifiable; Learning objectives, prerequisites, and outcomes must be clearly defined for each module and chapter; Content must follow logical learning progression from fundamentals to advanced topics.

### Modular Learning Architecture
Content organized in detailed modules following weekly schedule; Each module includes multiple chapters with theory, examples, applications, exercises, and summaries; Hierarchical sidebar structure from fundamentals to advanced topics. The textbook structure is:

## Physical AI & Humanoid Robotics – Textbook Structure

### Introduction
**Topic:** Introduction to Physical AI & Humanoid Robotics
**Focus:** AI Systems in the Physical World, Embodied Intelligence
**Goal:** Bridge the gap between digital brain and physical body; apply AI knowledge to control humanoid robots in simulated and real-world environments.

---

### Module 1: ROS 2 (Weeks 3–5)
**Focus:** Robotic Nervous System – Middleware for Robot Control
**Chapters:**
- Chapter 1: Introduction to ROS 2
- Chapter 2: ROS 2 Nodes and Topics
- Chapter 3: Services, Actions, and Parameters
- Chapter 4: URDF Robot Modeling
- Chapter 5: Launch Files and Package Management

---

### Module 2: Digital Twin (Weeks 6–7)
**Focus:** Physics Simulation & Environment Building
**Chapters:**
- Chapter 1: Introduction to Gazebo & Unity
- Chapter 2: Simulating Physics, Gravity, and Collisions
- Chapter 3: Sensor Simulation: LiDAR, Depth Cameras, IMUs
- Chapter 4: High-fidelity Rendering & Human-Robot Interaction

---

### Module 3: NVIDIA Isaac (Weeks 8–10)
**Focus:** AI-Robot Brain – Advanced Perception and Training
**Chapters:**
- Chapter 1: NVIDIA Isaac Sim Overview
- Chapter 2: Hardware-accelerated VSLAM (Isaac ROS)
- Chapter 3: Navigation & Path Planning (Nav2)
- Chapter 4: AI-powered Perception & Manipulation
- Chapter 5: Reinforcement Learning and Sim-to-Real Techniques

---

### Module 4: Vision-Language-Action (VLA) & Humanoids (Weeks 11–13)
**Focus:** Convergence of LLMs and Robotics
**Chapters:**
- Chapter 1: Humanoid Robot Development (Kinematics, Dynamics, Bipedal Locomotion)
- Chapter 2: Manipulation and Grasping with Humanoid Hands
- Chapter 3: Natural Human-Robot Interaction Design
- Chapter 4: Conversational Robotics (GPT Integration, Whisper Voice-to-Action, Multi-modal Interaction)
- Chapter 5: Capstone Project: Autonomous Humanoid (Full Simulation with Voice Commands, Path Planning, Object Identification, Manipulation)

### Code Quality & Versioning Standards
Every concept must include practical examples, code samples, and simulation exercises; Content must be tested with ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA systems; All code must be complete, runnable, safe, and properly versioned; Code examples must be in ROS 2, Python, NumPy, PyTorch and follow best practices.

### Docusaurus Structure & Accessibility
Content must include diagrams, media with descriptive alt text, and visual aids; Images stored in `/static/img/[chapter-name]/` with SVG preferred; Docusaurus frontend optimized for accessibility, search, and performance (LCP < 2.5s, CLS < 0.1); Ensure metadata for every .md file (title, description, keywords, sidebar_position).

### AI-Agent Integration & Personalization
Chapters must support RAG chatbot integration, personalized learning paths, and multi-language support; Content must integrate with Claude Code Subagents and Agent Skills; Include placeholders for Urdu translation support.

### Hardware-Aware Implementation
Content must account for specific hardware requirements and lab setups:
- High-performance workstation: RTX 4070 Ti+ GPU, Intel i7/AMD Ryzen 9 CPU, 64GB RAM, Ubuntu 22.04
- Physical AI Edge Kit: Jetson Orin Nano/NX, RealSense camera, USB IMU, ReSpeaker mic array
- Robot Lab options: Proxy Approach, Miniature Humanoid, Premium Lab
- Cloud/Hybrid lab options: AWS/Azure NVIDIA Isaac Sim cloud, with latency handling considerations
Include guidance for sim-to-real transfer, latency considerations, and cloud vs local lab environments.

## Technology Stack & Deployment Standards
Use Docusaurus for frontend publishing with the command `npx create-docusaurus@latest frontend classic` - do not create folder structure manually; Claude Code + Spec-Kit Plus for book writing and task orchestration; GitHub Pages / Vercel deployment; Optional: FastAPI + Qdrant + Neon Postgres for RAG chatbot integration; Prefer widely-used libraries (ROS 2, Python, NumPy, PyTorch); Ensure metadata for every .md file (title, description, keywords, sidebar_position); Hardware setup notes required for each module.

## Quality Gates & Assessment Integration
Pre-merge gates: Docusaurus build validation, broken link check, code & formula accuracy, accessibility, SEO, performance; Include weekly projects and exercises aligned with each module; Capstone project integrating all technologies; Spec-Kit workflow: Constitution → Specification → Plan → Tasks → Implementation.

## Governance
Constitution governs all content creation for the textbook; All PRs/reviews must verify compliance with educational standards, technical accuracy, and accessibility; Changes must follow Spec-Kit Plus workflow with ADRs for significant architectural decisions; Compliance review required before merging.

**Version**: 1.0.1 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06