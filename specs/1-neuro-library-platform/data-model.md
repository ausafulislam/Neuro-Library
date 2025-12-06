# Data Model: Neuro Library Platform for Physical AI & Humanoid Robotics Textbook

**Feature**: 1-neuro-library-platform
**Date**: 2025-12-06
**Status**: Completed

## Overview

This document defines the data models for the Neuro Library platform, focusing on the content structure and metadata required to support the Physical AI & Humanoid Robotics textbook.

## Content Entity Models

### Textbook Module
**Description**: Represents a major section of the textbook as defined in the constitution

**Fields**:
- `id` (string): Unique identifier (e.g., "module-1-ros2")
- `title` (string): Display title (e.g., "ROS 2")
- `description` (string): Brief description of the module
- `weeks` (string): Week range (e.g., "Weeks 3-5")
- `focus` (string): Module focus area from constitution
- `chapters` (array): List of chapter IDs belonging to this module
- `position` (integer): Order in the curriculum sequence
- `prerequisites` (array): List of prerequisite module IDs
- `learningOutcomes` (array): List of learning outcomes for the module

**Validation**:
- `id` must be unique across all modules
- `position` must be a positive integer
- `weeks` must follow format "Weeks X-Y" where X <= Y

### Textbook Chapter
**Description**: Represents an individual learning unit within a module

**Fields**:
- `id` (string): Unique identifier (e.g., "module-1-ros2-chapter-1")
- `title` (string): Chapter title
- `module_id` (string): Reference to parent module
- `position` (integer): Order within the module
- `learningObjectives` (array): List of learning objectives
- `prerequisites` (array): List of prerequisite chapter IDs
- `content` (string): Main content in Markdown format
- `examples` (array): List of code example file paths
- `exercises` (array): List of exercise descriptions
- `diagrams` (array): List of diagram file paths
- `references` (array): List of external references/citations
- `duration` (string): Estimated time to complete (e.g., "2-3 hours")

**Validation**:
- `id` must be unique across all chapters
- `module_id` must reference an existing module
- `position` must be a positive integer
- `learningObjectives` must not be empty

### Content Page
**Description**: Represents a Docusaurus content page with metadata

**Fields**:
- `id` (string): Unique identifier
- `title` (string): Page title for display
- `description` (string): Meta description for SEO
- `keywords` (array): SEO keywords
- `sidebar_position` (integer): Position in sidebar navigation
- `slug` (string): URL-friendly path
- `authors` (array): List of content authors
- `tags` (array): Content tags for categorization
- `draft` (boolean): Whether page is a draft
- `frontMatter` (object): Additional Docusaurus front matter

**Validation**:
- `title` must not be empty
- `sidebar_position` must be a positive integer
- `slug` must follow URL-friendly format

### Code Example
**Description**: Represents a code example associated with a chapter

**Fields**:
- `id` (string): Unique identifier
- `chapter_id` (string): Reference to parent chapter
- `title` (string): Descriptive title
- `description` (string): Explanation of what the code does
- `language` (string): Programming language (e.g., "python", "bash")
- `code` (string): The actual code content
- `expectedOutput` (string): Expected output when running the code
- `safetyWarnings` (array): List of safety warnings if applicable
- `dependencies` (array): List of required dependencies/libraries
- `version` (string): Version of the library/tool used

**Validation**:
- `chapter_id` must reference an existing chapter
- `language` must be a supported language
- `code` must not be empty

### Diagram/Image Asset
**Description**: Represents a visual asset used in the textbook

**Fields**:
- `id` (string): Unique identifier
- `filename` (string): Descriptive filename
- `path` (string): Relative path from static directory
- `altText` (string): Alt text for accessibility
- `title` (string): Title for the image
- `description` (string): Detailed description
- `format` (string): File format (e.g., "svg", "png", "jpg")
- `chapter_id` (string): Reference to associated chapter
- `usageContext` (string): Where the image is used (e.g., "theory", "example", "exercise")

**Validation**:
- `filename` must be descriptive and URL-friendly
- `altText` must not be empty
- `format` must be a supported image format

## System Configuration Models

### Site Configuration
**Description**: Configuration for the Neuro Library Docusaurus site

**Fields**:
- `title` (string): Site title ("Neuro Library – Physical AI & Humanoid Robotics")
- `tagline` (string): Short description
- `url` (string): Base URL
- `baseUrl` (string): Base path
- `favicon` (string): Path to favicon
- `organizationName` (string): GitHub organization name
- `projectName` (string): GitHub project name
- `themeConfig` (object): Theme-specific configuration
- `plugins` (array): List of Docusaurus plugins
- `presets` (array): List of Docusaurus presets

### Navigation Model
**Description**: Structure for sidebar navigation

**Fields**:
- `module` (string): Module identifier
- `label` (string): Display label for the module
- `items` (array): List of chapter navigation items
- `collapsible` (boolean): Whether the module section can be collapsed
- `className` (string): CSS class for styling

## Relationships

- `Textbook Module` 1 -- * `Textbook Chapter` (one module has many chapters)
- `Textbook Chapter` 1 -- * `Code Example` (one chapter has many examples)
- `Textbook Chapter` 1 -- * `Diagram/Image Asset` (one chapter has many diagrams)
- `Content Page` 1 -- 1 `Textbook Chapter` (one-to-one mapping for chapter pages)

## State Transitions

### Chapter State Transitions
- `draft` → `review` → `published` → `archived`
- `draft`: Initial state, content is being written
- `review`: Content is ready for review
- `published`: Content is live and accessible
- `archived`: Content is no longer maintained

## Validation Rules

1. All content must have proper learning objectives defined
2. Code examples must be runnable and properly tested
3. All images must have descriptive alt text
4. Content must follow the hierarchical structure defined in the constitution
5. All references must be properly cited
6. Content must be educationally sound and technically accurate