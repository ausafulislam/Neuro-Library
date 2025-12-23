import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";

import styles from "./index.module.css";
import NeuroBot from "../components/NeuroBot/NeuroBot";

function HomepageHeader() {
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          Neuro Library
        </Heading>

        <p className="hero__subtitle">
          AI Native Technical Books for the Future
        </p>

        <p className={styles.heroDescription}>
          A modern learning platform where students explore deep technical books
          on AI, robotics, future technologies, and intelligent systems.
        </p>

        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="#books">
            Explore Books
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className={styles.card}>
      <div className={styles.icon}>{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function AboutSection() {
  return (
    <section className={styles.aboutSection}>
      <div className="container">
        <h2 className={styles.sectionTitle}>About Neuro Library</h2>

        <p className={styles.sectionDescription}>
          Neuro Library is a modern AI native learning platform where students
          explore deep technical books on AI, robotics, future technologies, and
          intelligent systems. Every book is created for hands on learning, real
          world applications, and next generation engineering skills.
        </p>

        <div className={styles.grid}>
          <FeatureCard
            icon="📚"
            title="AI Native Textbooks"
            description="Deep technical content designed for the age of artificial intelligence and intelligent systems."
          />

          <FeatureCard
            icon="🤖"
            title="Hands On Learning"
            description="Practical exercises, simulations, and real world applications that bring concepts to life."
          />

          <FeatureCard
            icon="🚀"
            title="Future Technologies"
            description="Cutting edge content covering robotics, VLA systems, and next generation AI fields."
          />
        </div>
      </div>
    </section>
  );
}

function BookCard({ title, description, icon, link }) {
  return (
    <div className={styles.card}>
      <div className={styles.icon}>{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>

      <Link className="button button--secondary button--block" to={link}>
        View Details
      </Link>
    </div>
  );
}

function BooksSection() {
  return (
    <section className={styles.booksSection} id="books">
      <div className="container">
        <h2 className={styles.sectionTitle}>Our Books</h2>

        <p className={styles.sectionDescription}>
          Explore our AI native technical books designed for the future of
          technology.
        </p>

        <div className={styles.grid}>
          <BookCard
            title="Physical AI and Humanoid Robotics"
            description="A complete textbook that covers ROS 2, Gazebo, NVIDIA Isaac, and Vision Language Action systems."
            icon="🤖"
            link="/book-details"
          />
        </div>
      </div>
    </section>
  );
}

function WhyLearnHere() {
  return (
    <section className={styles.whySection}>
      <div className="container">
        <h2 className={styles.sectionTitle}>Why Learn Here?</h2>

        <div className={styles.grid}>
          <FeatureCard
            icon="🧠"
            title="AI Native Approach"
            description="Learn concepts built for modern AI technologies and workflows."
          />

          <FeatureCard
            icon="⚙️"
            title="Hands On Simulations"
            description="Integrate theory with practice using real robotics simulations."
          />

          <FeatureCard
            icon="🏭"
            title="Industry Focused"
            description="Experience the workflows of real intelligent systems engineering."
          />
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Neuro Library, ${siteConfig.tagline}`}
      description="AI native technical textbooks for robotics, AI, and future technologies"
    >
      <HomepageHeader />
      <main>
        <NeuroBot />
        <AboutSection />
        <BooksSection />
        <WhyLearnHere />
      </main>
    </Layout>
  );
}
