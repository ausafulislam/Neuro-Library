# Quickstart Guide: Neuro Library Platform

**Feature**: 1-neuro-library-platform
**Date**: 2025-12-06
**Status**: Completed

## Overview

This quickstart guide provides the essential steps to set up, develop, and deploy the Neuro Library platform for the Physical AI & Humanoid Robotics textbook.

## Prerequisites

- Node.js LTS (v18 or higher)
- npm or yarn package manager
- Git
- Basic knowledge of Docusaurus and React

## Setup Instructions

### 1. Clone and Initialize the Repository

```bash
# Clone your repository
git clone <repository-url>
cd <repository-name>

# Navigate to the frontend directory
# (if this is a new setup, follow the steps below to create it)
```

### 2. Create the Neuro Library Frontend

If you're setting up a new Neuro Library platform:

```bash
# Create the frontend directory
mkdir frontend
cd frontend

# Initialize Docusaurus project
npx create-docusaurus@latest . classic

# Install additional dependencies as needed
npm install @docusaurus/module-type-aliases @docusaurus/types
```

### 3. Configure Docusaurus for Neuro Library

Replace the default Docusaurus configuration with Neuro Library branding:

1. Update `docusaurus.config.js`:

```javascript
// docusaurus.config.js
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Neuro Library – Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive AI-native textbook on Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-organization.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<org-name>/<repo-name>'
  baseUrl: '/neuro-library/',

  // GitHub pages deployment config.
  organizationName: 'your-organization', // Usually your GitHub org/user name.
  projectName: 'neuro-library', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-organization/neuro-library/edit/main/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Neuro Library',
        logo: {
          alt: 'Neuro Library Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            href: 'https://github.com/your-organization/neuro-library',
            className: 'header-github-link',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook',
            items: [
              {
                label: 'Introduction',
                to: '/docs/introduction',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/neuro-library',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-organization/neuro-library',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Neuro Library. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
```

### 4. Set up Content Structure

Create the directory structure according to the constitution:

```bash
# In the frontend directory
mkdir -p docs/{introduction,module-1-ros2,module-2-digital-twin,module-3-nvidia-isaac,module-4-vla-humanoids}
mkdir -p static/img/{introduction,module-1-ros2,module-2-digital-twin,module-3-nvidia-isaac,module-4-vla-humanoids}
mkdir -p examples/{module-1-ros2,module-2-digital-twin,module-3-nvidia-isaac,module-4-vla-humanoids}
mkdir -p src/pages
```

### 5. Create the Landing Page

Create `src/pages/index.js` for the landing page:

```jsx
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
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

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A comprehensive AI-native textbook on Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems">
      <HomepageHeader />
      <main>
        {/* Add additional content here */}
      </main>
    </Layout>
  );
}
```

### 6. Configure Sidebar Navigation

Update `sidebars.js` to reflect the module structure:

```javascript
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'autogenerated',
      dirName: '.',
    },
  ],
};

module.exports = sidebars;
```

For a more structured approach, you can manually define the sidebar:

```javascript
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['introduction/index'],
      link: {
        type: 'doc',
        id: 'introduction/index',
      },
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 (Weeks 3-5)',
      items: [
        'module-1-ros2/chapter-1-introduction-to-ros2',
        'module-1-ros2/chapter-2-nodes-and-topics',
        'module-1-ros2/chapter-3-services-actions-parameters',
        'module-1-ros2/chapter-4-urdf-robot-modeling',
        'module-1-ros2/chapter-5-launch-files-package-management'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Weeks 6-7)',
      items: [
        'module-2-digital-twin/chapter-1-introduction-gazebo-unity',
        'module-2-digital-twin/chapter-2-physics-simulation',
        'module-2-digital-twin/chapter-3-sensor-simulation',
        'module-2-digital-twin/chapter-4-high-fidelity-rendering'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac (Weeks 8-10)',
      items: [
        'module-3-nvidia-isaac/chapter-1-isaac-sim-overview',
        'module-3-nvidia-isaac/chapter-2-hardware-accelerated-vslam',
        'module-3-nvidia-isaac/chapter-3-navigation-path-planning',
        'module-3-nvidia-isaac/chapter-4-ai-powered-perception',
        'module-3-nvidia-isaac/chapter-5-reinforcement-learning'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: VLA & Humanoids (Weeks 11-13)',
      items: [
        'module-4-vla-humanoids/chapter-1-humanoid-robot-development',
        'module-4-vla-humanoids/chapter-2-manipulation-grasping',
        'module-4-vla-humanoids/chapter-3-human-robot-interaction',
        'module-4-vla-humanoids/chapter-4-conversational-robotics',
        'module-4-vla-humanoids/chapter-5-capstone-autonomous-humanoid'
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;
```

### 7. Add Content with Proper Frontmatter

Create a sample chapter with required frontmatter:

```markdown
---
title: Introduction to ROS 2
description: Learn the fundamentals of Robot Operating System 2 (ROS 2) for humanoid robotics applications
keywords: [ros2, robotics, middleware, nodes, topics]
sidebar_position: 1
---

# Introduction to ROS 2

## Learning Objectives

- Understand the basic concepts of ROS 2
- Learn about nodes, topics, services, and actions
- Set up your first ROS 2 workspace

## Prerequisites

- Basic knowledge of Linux command line
- Familiarity with Python programming

## Content

[Your chapter content here]

## Exercises

1. [Exercise 1 description]
2. [Exercise 2 description]

## References

- [External reference 1]
- [External reference 2]
```

### 8. Development Workflow

Start the development server:

```bash
cd frontend
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### 9. Build and Deployment

To build the static files for production:

```bash
cd frontend
npm run build
```

The build artifacts will be stored in the `build` directory.

For GitHub Pages deployment:

```bash
npm run deploy
```

### 10. Content Creation Guidelines

When creating content for the Neuro Library platform:

1. **Follow Constitution Principles**:
   - Ensure content accuracy and educational clarity
   - Include learning objectives and prerequisites
   - Provide practical examples and exercises
   - Use proper alt text for images

2. **Code Examples**:
   - Include complete, runnable code
   - Add safety warnings where applicable
   - Provide expected output
   - Use Python, ROS 2, or PyTorch as appropriate

3. **Accessibility**:
   - Use proper heading hierarchy (H1, H2, H3, etc.)
   - Include descriptive alt text for all images
   - Maintain good color contrast
   - Ensure keyboard navigation works

4. **Performance**:
   - Optimize images (use WebP when possible)
   - Keep page load times under 3 seconds
   - Use lazy loading for images below the fold

## Troubleshooting

### Common Issues

1. **Page not loading**: Check that all required frontmatter fields are present
2. **Sidebar not showing**: Verify that `sidebar_position` is set in the document frontmatter
3. **Images not displaying**: Ensure images are in the `static/img` directory and referenced with the correct path

### Performance Tips

- Use SVG format for diagrams when possible
- Compress images without sacrificing quality
- Use code splitting for large examples
- Implement lazy loading for images