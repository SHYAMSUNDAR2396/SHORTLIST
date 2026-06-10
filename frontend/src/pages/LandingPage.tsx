import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
  const navigate = useNavigate();
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    // Scroll-triggered reveal via IntersectionObserver
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.setAttribute('data-visible', 'true');
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -60px 0px' }
    );

    document.querySelectorAll('[data-reveal]').forEach((el) => {
      observerRef.current?.observe(el);
    });

    return () => observerRef.current?.disconnect();
  }, []);

  return (
    <div className="landing">
      {/* ─── Hero ─── */}
      <section className="landing-hero">
        <div className="hero-bg" aria-hidden="true">
          <div className="hero-orb hero-orb-1" />
          <div className="hero-orb hero-orb-2" />
        </div>
        <div className="hero-content">
          <h1 className="hero-title">Shortlist</h1>
          <p className="hero-tagline">
            Rank 100,000 candidates in 11 seconds.<br />
            No LLMs. No GPU. Just signal.
          </p>
          <button
            onClick={() => navigate('/')}
            className="hero-cta"
          >
            Open Dashboard
          </button>
        </div>
        <div className="scroll-indicator" aria-hidden="true">
          <span />
        </div>
      </section>

      {/* ─── Principles ─── */}
      <section className="landing-section">
        <div className="section-inner">
          <h2 className="section-heading" data-reveal>How it thinks</h2>
          <div className="principles-grid">
            {[
              {
                num: '01',
                title: 'Career over keywords',
                desc: 'Reads role descriptions to infer production experience — not just skill tags.',
              },
              {
                num: '02',
                title: 'Honeypot detection',
                desc: 'Four temporal-consistency rules catch impossible profiles before they rank.',
              },
              {
                num: '03',
                title: 'Deterministic output',
                desc: 'Same input, same output. Every time. No randomness, no clock dependency.',
              },
              {
                num: '04',
                title: 'Fairness by design',
                desc: 'Never reads names or gender. Counterfactual audit measures institution bias.',
              },
              {
                num: '05',
                title: 'Six scoring dimensions',
                desc: 'Skill, career, experience, behavior, education, location — weighted and transparent.',
              },
            ].map((item, i) => (
              <div
                key={item.num}
                className="principle-card"
                data-reveal
                style={{ transitionDelay: `${i * 80}ms` }}
              >
                <span className="principle-num">{item.num}</span>
                <h3 className="principle-title">{item.title}</h3>
                <p className="principle-desc">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Who it's for ─── */}
      <section className="landing-section section-alt">
        <div className="section-inner">
          <h2 className="section-heading" data-reveal>Who it's for</h2>
          <div className="audience-grid">
            {[
              {
                title: 'Recruiters',
                desc: 'Upload candidates. Get a ranked shortlist with reasons. No ML degree required.',
                icon: '👤',
              },
              {
                title: 'Engineers',
                desc: 'Pure Python. Modular scorers. 129 property-based tests. Fork and extend.',
                icon: '⚙️',
              },
              {
                title: 'Teams',
                desc: 'Dashboard with fairness audit. Export CSV. Self-hosted, privacy-first.',
                icon: '🏢',
              },
            ].map((item, i) => (
              <div
                key={item.title}
                className="audience-card"
                data-reveal
                style={{ transitionDelay: `${i * 100}ms` }}
              >
                <span className="audience-icon">{item.icon}</span>
                <h3 className="audience-title">{item.title}</h3>
                <p className="audience-desc">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Usage ─── */}
      <section className="landing-section">
        <div className="section-inner">
          <h2 className="section-heading" data-reveal>Three commands. That's it.</h2>
          <div className="code-block" data-reveal>
            <div className="code-line"><span className="code-comment"># Install</span></div>
            <div className="code-line">pip install -r requirements.txt</div>
            <div className="code-line">&nbsp;</div>
            <div className="code-line"><span className="code-comment"># Rank</span></div>
            <div className="code-line">python rank.py --candidates data.jsonl --out shortlist.csv</div>
            <div className="code-line">&nbsp;</div>
            <div className="code-line"><span className="code-comment"># Or launch the full dashboard</span></div>
            <div className="code-line">./run_dev.sh</div>
          </div>
          <p className="code-footnote" data-reveal>
            No ceremony. No bloat. 100K candidates → top 100 in 11 seconds on a laptop.
          </p>
        </div>
      </section>

      {/* ─── Stats ─── */}
      <section className="landing-section section-alt">
        <div className="section-inner">
          <div className="stats-grid">
            {[
              { value: '11s', label: 'Runtime (100K)' },
              { value: '1.1GB', label: 'Peak memory' },
              { value: '129', label: 'Tests passing' },
              { value: '0%', label: 'Honeypot leakage' },
            ].map((stat, i) => (
              <div
                key={stat.label}
                className="stat-card"
                data-reveal
                style={{ transitionDelay: `${i * 80}ms` }}
              >
                <span className="stat-value">{stat.value}</span>
                <span className="stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Footer ─── */}
      <footer className="landing-footer">
        <div className="footer-inner">
          <span className="footer-brand">Shortlist</span>
          <span className="footer-tagline">Signal over noise.</span>
          <div className="footer-links">
            <a href="https://github.com/SHYAMSUNDAR2396/SHORTLIST" target="_blank" rel="noopener noreferrer">GitHub</a>
            <button onClick={() => navigate('/')}>Dashboard</button>
            <button onClick={() => navigate('/pipeline')}>Run Pipeline</button>
          </div>
          <span className="footer-copy">© 2025 Shortlist. Built with intention.</span>
        </div>
      </footer>
    </div>
  );
}
