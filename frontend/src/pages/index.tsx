import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          Neuro Library
        </Heading>
        <p className="hero__subtitle">AI-Native Technical Books for the Future</p>
        <p className="hero__description">
          A modern learning platform where students explore deep technical books on AI, robotics, future technologies, and intelligent systems.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="#books">
            Explore Books
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({ icon, title, description }: { icon: string, title: string, description: string }) {
  return (
    <div className={`col col--4 ${styles.featureCard}`}>
      <div className={styles.card}>
        <div className={styles.icon}>{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function AboutSection() {
  return (
    <section className={styles.aboutSection}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>About Neuro Library</h2>
            <p className={styles.sectionDescription}>
              Neuro Library is a modern AI-native learning platform where students explore deep technical books on AI, robotics, future technologies, and intelligent systems.
              Every book is crafted for hands-on learning, real-world applications, and next-generation engineering skills.
            </p>

            <div className="row" style={{ marginTop: '2rem' }}>
              <FeatureCard
                icon="📚"
                title="AI-Native Textbooks"
                description="Deep technical content designed for the age of artificial intelligence and intelligent systems."
              />
              <FeatureCard
                icon="🤖"
                title="Hands-On Learning"
                description="Practical exercises, simulations, and real-world applications that bring concepts to life."
              />
              <FeatureCard
                icon="🚀"
                title="Future Technologies"
                description="Cutting-edge content covering robotics, vision-language-action systems, and emerging AI fields."
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}


function BookCard({ title, description, icon, link }: { title: string, description: string, icon: string, link: string }) {
  return (
    <div className={`col col--4 ${styles.bookCard}`}>
      <div className={styles.card}>
        <div className={styles.icon}>{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
        <Link
          className="button button--secondary button--block"
          to={link}>
          View Details
        </Link>
      </div>
    </div>
  );
}

function BooksSection() {
  return (
    <section className={styles.booksSection} id="books">
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Our Books</h2>
            <p className={styles.sectionDescription}>
              Explore our collection of AI-native technical books designed for the future of technology.
            </p>
          </div>
        </div>

        <div className="row" style={{ marginTop: '2rem' }}>
          <BookCard
            title="Physical AI & Humanoid Robotics"
            description="A comprehensive AI-native textbook covering ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems."
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
    <section>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Why Learn Here?</h2>
          </div>
        </div>

        <div className="row" style={{ marginTop: '2rem' }}>
          <div className={`col col--4 ${styles.cardbody}`}>
            <div className={styles.card}>
              <h3>AI-Native Approach</h3>
              <p>Content designed from the ground up for modern AI technologies and methodologies.</p>
            </div>
          </div>

          <div className={`col col--4 ${styles.benefitCard}`}>
            <div className={styles.card}>
              <h3>Hands-On Simulations</h3>
              <p>Integrate theory with practice through interactive simulations and real robotics pipelines.</p>
            </div>
          </div>

          <div className={`col col--4 ${styles.benefitCard}`}>
            <div className={styles.card}>
              <h3>Real Robotics Pipelines</h3>
              <p>Experience actual robotic systems with cutting-edge hardware and software stacks.</p>
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
      title={`Neuro Library - ${siteConfig.tagline}`}
      description="Modern AI-native technical textbooks for robotics, AI, and future technologies">
      <HomepageHeader />
      <main>
        <AboutSection />
        <hr />
        <BooksSection />
        <hr />
        <WhyLearnHere />
      </main>
    </Layout>
  );
}
