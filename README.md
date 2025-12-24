# Neuro Library

<div align="center">

[![Neuro Library Logo](./frontend/static/img/logo.png)](https://neurolibrary.vercel.app)

</div>

<div align="center">

[![License](https://img.shields.io/github/license/ausafulislam/neuro-library)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/) [![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)](https://www.typescriptlang.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green?logo=fastapi)](https://fastapi.tiangolo.com/) [![Docusaurus](https://img.shields.io/badge/Docusaurus-3.x-informational?logo=docusaurus)](https://docusaurus.io/) [![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)](https://qdrant.tech/) [![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/ausafulislam/neuro-library)

</div>


**AI-Native Learning Platform for Physical AI & Humanoid Robotics** 🤖

>_A comprehensive educational resource with RAG-powered AI assistance_


---

Neuro Library is an **AI-native learning platform** for **Physical AI & Humanoid Robotics** education. It provides comprehensive textbook content organized in structured modules covering **ROS 2**, **Digital Twin technologies**, **NVIDIA Isaac**, and **Vision-Language-Action systems** with humanoid robotics. 📘

## Overview

The Neuro Library platform is designed as an educational resource for students, engineers, and AI enthusiasts interested in learning about physical AI and humanoid robotics. The platform combines a Docusaurus-based frontend for content delivery with a FastAPI backend that implements RAG (Retrieval-Augmented Generation) capabilities for AI-powered learning assistance.

### Key Features 🚀

- 📘 **Comprehensive Educational Content**: Structured modules covering ROS 2, Digital Twin, NVIDIA Isaac, and VLA & Humanoids
- 🤖 **AI-Powered Learning Assistance**: RAG-based chatbot for answering questions about textbook content
- 📱 **Responsive Web Interface**: Built with Docusaurus for optimal learning experience across devices
- 📊 **Modular Learning Path**: Clear progression from fundamentals to advanced topics
- 💻 **Code Examples**: Practical examples and exercises integrated throughout the content
- ⚡ **Performance Optimized**: Fast loading times and responsive user experience

## Architecture 🏗️

The project is organized into two main components:

### 🐍 Backend (Python/FastAPI)

Located in the `backend/` directory, the backend provides:

- 🤖 **RAG Server**: Implements Retrieval-Augmented Generation for AI-powered question answering
- 📥 **Content Ingestion**: Automatically fetches and processes website content into vector database
- 🔍 **Vector Database Integration**: Uses Qdrant for efficient similarity search
- 🧠 **AI Agent Integration**: Implements AI agents for enhanced learning assistance
- 🌐 **API Endpoints**: Provides RESTful APIs for frontend integration

**Key technologies:**
- FastAPI for web framework
- Sentence Transformers for text embeddings
- Qdrant for vector database
- BeautifulSoup for web scraping
- OpenAI agents for AI assistance

### 🌐 Frontend (Docusaurus)

Located in the `frontend/` directory, the frontend provides:

- 📚 **Educational Content Platform**: Docusaurus-based static site for textbook content
- 📱 **Responsive Design**: Mobile and desktop optimized reading experience
- 🔍 **Search Functionality**: Local search across all textbook content
- 🧭 **Navigation Structure**: Organized by modules and chapters as per curriculum
- 🎨 **Branded UI**: Custom "Neuro Library" branding instead of default Docusaurus

**Key technologies:**
- Docusaurus v3.x
- React
- TypeScript
- Custom CSS for styling

## Project Structure

```
Neuro Library/
├── backend/                    # FastAPI backend with RAG capabilities
│   ├── main.py                 # Main RAG server implementation
│   ├── agent.py                # AI agent for enhanced interactions
│   ├── requirements.txt        # Python dependencies
│   └── pyproject.toml          # Project configuration
├── frontend/                   # Docusaurus frontend
│   ├── docs/                   # Textbook content (modules, chapters)
│   ├── src/                    # Custom components and pages
│   ├── static/                 # Static assets (images, etc.)
│   ├── docusaurus.config.ts    # Docusaurus configuration
│   └── package.json            # Frontend dependencies
├── specs/                      # Project specifications and requirements
│   └── 1-neuro-library-platform/ # Feature specification files
├── history/                    # Development history and prompts
│   └── prompts/                # Prompt history records
├── .specify/                   # SpecKit Plus configuration
└── README.md                   # This file
```

## Curriculum Structure 📚

The textbook content is organized into **4 main modules**:

### 🤖 Module 1: ROS 2 (Weeks 3-5)
- Chapter 1: Introduction to ROS 2
- Chapter 2: Nodes and Topics
- Chapter 3: Services, Actions, and Parameters
- Chapter 4: URDF Robot Modeling
- Chapter 5: Launch Files and Package Management

### 🌐 Module 2: Digital Twin (Weeks 6-7)
- Chapter 1: Introduction to Gazebo and Unity
- Chapter 2: Physics Simulation
- Chapter 3: Sensor Simulation
- Chapter 4: High-Fidelity Rendering

### ⚡ Module 3: NVIDIA Isaac (Weeks 8-10)
- Chapter 1: Isaac Sim Overview
- Chapter 2: Hardware-Accelerated VSLAM
- Chapter 3: Navigation and Path Planning
- Chapter 4: AI-Powered Perception
- Chapter 5: Reinforcement Learning

### 🤖 Module 4: VLA & Humanoids (Weeks 11-13)
- Chapter 1: Humanoid Robot Development
- Chapter 2: Manipulation and Grasping
- Chapter 3: Human-Robot Interaction
- Chapter 4: Conversational Robotics
- Chapter 5: Capstone - Autonomous Humanoid

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend/
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```env
SITEMAP_URL=your_sitemap_url
COLLECTION_NAME=your_collection_name
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPEN_ROUTER_API_KEY=your_openrouter_api_key
OPEN_ROUTER_BASE_URL=https://openrouter.ai/api/v1
OPEN_ROUTER_MODEL=model_name
```

5. Run the backend server:
```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend/
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Start the development server:
```bash
npm run start
# or
yarn start
```

## Usage

1. The backend server will automatically ingest content from the specified sitemap during startup
2. The frontend provides a web interface to browse the textbook content
3. The RAG API endpoint (`/ask`) allows querying the ingested content
4. The AI agent endpoint (`/chat`) provides conversational access to the textbook content

## Configuration

### Environment Variables

The application requires several environment variables to be set:

**Backend:**
- `SITEMAP_URL`: URL to the sitemap containing textbook content
- `COLLECTION_NAME`: Qdrant collection name for storing embeddings
- `QDRANT_URL`: Qdrant vector database URL
- `QDRANT_API_KEY`: Qdrant API key for authentication
- `OPEN_ROUTER_API_KEY`: API key for OpenRouter (if using AI agent)
- `OPEN_ROUTER_BASE_URL`: Base URL for OpenRouter API
- `OPEN_ROUTER_MODEL`: Model name for OpenRouter

## Development

### Adding Content

To add new textbook content:

1. Create new markdown files in the `frontend/docs/` directory following the module/chapter structure
2. Update the sidebar configuration in `frontend/sidebars.ts`
3. Add images to the `frontend/static/img/` directory with descriptive filenames
4. Ensure each content file includes proper metadata (title, description, keywords, sidebar_position)

### API Endpoints

**Backend:**
- `GET /` - Health check endpoint
- `POST /ask` - RAG question answering (returns context and sources)
- `POST /chat` - AI agent conversation endpoint

## Deployment 🚀

### Frontend Deployment

The frontend is built for static hosting and can be deployed to platforms like:

- 🟦 [Vercel](https://vercel.com/) (recommended, as configured in docusaurus.config.ts)
- 🟨 [Netlify](https://www.netlify.com/)
- 🟪 [GitHub Pages](https://pages.github.com/)
- 🤗 [Hugging Face Spaces](https://huggingface.co/spaces) (for static sites)
- Any static hosting service

Build command:
```bash
npm run build
```

### Backend Deployment

The backend can be deployed to platforms that support Python applications:

- 🟦 [Vercel](https://vercel.com/)
- 🟪 [Railway](https://railway.app/)
- 🅱️ [Heroku](https://www.heroku.com/)
- ☁️ [AWS](https://aws.amazon.com/), [GCP](https://cloud.google.com/), or [Azure](https://azure.microsoft.com/)
- 🤗 [Hugging Face Inference API](https://huggingface.co/inference-api) (for API deployment)

## Performance Targets ⚡

- ⏱️ **Page load time**: < 3 seconds
- 🖼️ **Largest Contentful Paint**: < 2.5s
- 📐 **Cumulative Layout Shift**: < 0.1
- 📱 **Responsive design**: Optimized for all device sizes

## Contributing 🤝

We welcome contributions from the community! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌟 **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. ✏️ **Make** your changes
4. 💾 **Commit** your changes (`git commit -m 'Add amazing feature'`)
5. 📤 **Push** to the branch (`git push origin feature/amazing-feature`)
6. 🔄 **Open** a Pull Request

## License 📄

This project is licensed under the terms specified in the project documentation.

## Support 🛟

For support, please open an issue in the GitHub repository or contact the development team.

## About ℹ️

<div align="center">

**Neuro Library** was developed as part of a **Spec-Driven Hackathon** by **Ausaf ul Islam**.

The platform follows the principles of **Spec-Driven Development (SDD)** and implements an **AI-native approach** to technical education. 🎯

</div>

<div align="center">

🤖 **Physical AI & Humanoid Robotics Education** | 🧠 **AI-Powered Learning** | 📘 **Comprehensive Textbook Content**

</div>