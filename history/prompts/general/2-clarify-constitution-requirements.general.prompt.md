---
id: 2
title: Clarify Constitution Requirements
stage: general
date: 2025-12-06
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: User
command: /sp.clarify
labels: ["constitution", "clarification", "education", "robotics"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.clarify

You are tasked with **clarifying and verifying the Constitution** for the AI-native technical textbook "Physical AI & Humanoid Robotics," which will be part of the **Neuro Library** platform. This Clarify step will ensure that the Constitution is **complete, accurate, and aligned with hackathon requirements**.

Follow these instructions carefully:

1. **Platform and Tech Stack Verification**
   - Confirm that the Constitution includes instructions to use **Docusaurus** for the frontend:
     ```
     npx create-docusaurus@latest frontend classic
     ```
   - Verify that it specifies **not to create the folder structure manually**.
   - Confirm tech stack coverage:
     - Docusaurus (frontend/static site generator)
     - Claude Code + Spec-Kit Plus (book writing and task orchestration)
     - GitHub Pages / Vercel deployment
     - Optional: FastAPI + Qdrant + Neon Postgres for RAG chatbot integration

2. **Full Textbook Index Verification**
   - Generate the **complete hierarchical book index** including all modules and chapters by week, reflecting the hackathon docs.
   - Include module focus, topics, weekly breakdown, and capstone integration. The index must include:

     **Module 1: The Robotic Nervous System (ROS 2)**
       - Week 1-2: Introduction to Physical AI, Sensor Systems
       - Week 3-5: ROS 2 Fundamentals (Nodes, Topics, Services, rclpy, URDF)

     **Module 2: The Digital Twin (Gazebo & Unity)**
       - Week 6-7: Physics Simulation, URDF/SDF, Gazebo, Unity visualization, Sensor Simulation

     **Module 3: The AI-Robot Brain (NVIDIA Isaac)**
       - Week 8-10: NVIDIA Isaac Sim, Isaac ROS, VSLAM, Navigation, Reinforcement Learning, Sim-to-Real

     **Module 4: Vision-Language-Action (VLA)**
       - Week 11-12: Humanoid Robot Development (Kinematics, Bipedal Locomotion, Manipulation, HRI)
       - Week 13: Conversational Robotics (GPT integration, Whisper voice-to-action, Multi-modal interaction)
       - Capstone Project: Autonomous Humanoid

3. **Hardware and Lab Requirements Verification**
   - Check that Constitution includes:
     - High-performance workstation (RTX 4070 Ti+ GPU, Intel i7/AMD Ryzen 9 CPU, 64GB RAM, Ubuntu 22.04)
     - Physical AI Edge Kit (Jetson Orin Nano/NX, RealSense camera, USB IMU, ReSpeaker mic array)
     - Robot Lab options: Proxy Approach, Miniature Humanoid, Premium Lab
     - Cloud/Hybrid lab options (AWS/Azure NVIDIA Isaac Sim cloud, latency handling)

4. **Content and Workflow Verification**
   - Confirm Constitution includes:
     - Core principles: Content Accuracy, Educational Clarity, Consistency, Docusaurus Structure, Code Quality, Deployment Standards
     - Workflow: Constitution → Specification → Plan → Tasks → Implementation
     - ADRs (Architectural Decision Records)
     - Quality gates, review process, pre-merge checks

5. **Output Requirements**
   - ✅ Constitution includes all required principles and technical details
   - ✅ Docusaurus frontend ins
```

## Response snapshot

Updated the constitution to align with hackathon requirements:

1. Enhanced the Modular Learning Architecture principle with detailed textbook structure:
   - Added complete Physical AI & Humanoid Robotics textbook structure
   - Included Introduction section with topic, focus, and goal
   - Added Module 1 (ROS 2, Weeks 3-5) with 5 chapters
   - Added Module 2 (Digital Twin, Weeks 6-7) with 4 chapters
   - Added Module 3 (NVIDIA Isaac, Weeks 8-10) with 5 chapters
   - Added Module 4 (VLA & Humanoids, Weeks 11-13) with 5 chapters including capstone

2. Updated Hardware-Aware Implementation principle with specific requirements:
   - High-performance workstation specs (RTX 4070 Ti+, CPU, RAM, OS)
   - Physical AI Edge Kit components (Jetson Orin, RealSense, IMU, mic array)
   - Robot Lab options (Proxy, Miniature Humanoid, Premium)
   - Cloud/Hybrid lab options (AWS/Azure Isaac Sim)

3. Enhanced Technology Stack & Deployment Standards with:
   - Docusaurus creation command: `npx create-docusaurus@latest frontend classic`
   - Explicit instruction not to create folder structure manually
   - Claude Code + Spec-Kit Plus for book writing
   - Optional FastAPI + Qdrant + Neon Postgres for RAG
   - Updated principles for consistency with requirements

4. Updated Core Principles:
   - Renamed "Educational Excellence & Technical Accuracy" to "Content Accuracy & Educational Clarity"
   - Added "Code Quality & Versioning Standards" principle
   - Revised "Docusaurus Structure & Accessibility" principle

Version updated from 1.0.0 to 1.0.1 to reflect these clarifications.

## Outcome

- ✅ Impact: Constitution now fully aligned with hackathon requirements
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with detailed structure and requirements
- 🔁 Next prompts: Proceed to create feature specifications based on updated constitution
- 🧠 Reflection: Constitution now includes all required elements for Physical AI & Humanoid Robotics textbook

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Create module specifications based on updated constitution