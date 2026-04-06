import { useEffect, useRef, useState, useCallback, ReactNode, createContext, useContext } from 'react'
import { BrowserRouter, Routes, Route, Link, useParams, useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/* ═══════════════════════════════════════════════════════════════════════════
   TURBORAG — Ultra-refined Technical Excellence
   
   Aesthetic: Restrained precision, institutional gravitas
   Typography: Inter + Source Serif 4 + JetBrains Mono
   Motion: Nearly invisible, opacity only
   ═══════════════════════════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────────────────────────────────────
// Theme Context
// ─────────────────────────────────────────────────────────────────────────────

type Theme = 'light' | 'dark'

const ThemeContext = createContext<{
  theme: Theme
  toggleTheme: () => void
}>({
  theme: 'light',
  toggleTheme: () => {},
})

function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('turborag-theme') as Theme
      if (stored) return stored
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return 'light'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('turborag-theme', theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme(t => t === 'light' ? 'dark' : 'light')
  }, [])

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

function useTheme() {
  return useContext(ThemeContext)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hooks
// ─────────────────────────────────────────────────────────────────────────────

function useInView(threshold = 0.1) {
  const ref = useRef<HTMLDivElement>(null)
  const [isInView, setIsInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      setIsInView(true)
      return
    }
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true)
          observer.unobserve(el)
        }
      },
      { threshold, rootMargin: '0px 0px -50px 0px' }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold])

  return { ref, isInView }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout Components
// ─────────────────────────────────────────────────────────────────────────────

function Container({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`container ${className}`}>{children}</div>
}

function Section({ 
  children, 
  id, 
  className = '',
  dark = false 
}: { 
  children: ReactNode
  id?: string
  className?: string
  dark?: boolean 
}) {
  return (
    <section id={id} className={`section ${dark ? 'section--dark' : ''} ${className}`}>
      {children}
    </section>
  )
}

function Reveal({ 
  children, 
  className = '',
  delay = 0 
}: { 
  children: ReactNode
  className?: string
  delay?: number 
}) {
  const { ref, isInView } = useInView()
  
  return (
    <div 
      ref={ref}
      className={`reveal ${isInView ? 'reveal--visible' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Copy Button Component
// ─────────────────────────────────────────────────────────────────────────────

function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [text])

  return (
    <button 
      className={`${className} ${copied ? `${className}--copied` : ''}`}
      onClick={handleCopy}
      aria-label={copied ? 'Copied!' : 'Copy to clipboard'}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <svg viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      ) : (
        <svg viewBox="0 0 20 20" fill="currentColor">
          <path d="M8 2a1 1 0 000 2h2a1 1 0 100-2H8z" />
          <path d="M3 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v6h-4.586l1.293-1.293a1 1 0 00-1.414-1.414l-3 3a1 1 0 000 1.414l3 3a1 1 0 001.414-1.414L10.414 13H15v3a2 2 0 01-2 2H5a2 2 0 01-2-2V5zM15 11h2a1 1 0 110 2h-2v-2z" />
        </svg>
      )}
    </button>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────

function Header() {
  const [scrolled, setScrolled] = useState(false)
  const { theme, toggleTheme } = useTheme()
  const location = useLocation()
  const isDocsPage = location.pathname.startsWith('/docs')

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <header className={`header ${scrolled ? 'header--scrolled' : ''}`}>
      <Container className="header__inner">
        <Link to="/" className="header__logo">
          <svg className="header__mark" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M8 12h8M12 8v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <span className="header__wordmark">TurboRAG</span>
        </Link>
        
        <nav className="header__nav">
          {!isDocsPage && (
            <>
              <a href="#features" className="header__link">Features</a>
              <a href="#performance" className="header__link">Performance</a>
            </>
          )}
          <Link to="/docs" className="header__link">Docs</Link>
          <a 
            href="https://github.com/ratnam1510/turborag" 
            className="header__link"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <button 
            className="header__theme-toggle" 
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? (
              <svg viewBox="0 0 20 20" fill="currentColor">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
              </svg>
            ) : (
              <svg viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
              </svg>
            )}
          </button>
        </nav>
      </Container>
    </header>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Hero
// ─────────────────────────────────────────────────────────────────────────────

function Hero() {
  const [installTab, setInstallTab] = useState<'pip' | 'npm'>('pip')
  const installCommand = installTab === 'pip' ? 'pip install turborag' : 'npm install turborag'
  
  return (
    <section className="hero">
      <Container>
        <div className="hero__content">
          <Reveal delay={50}>
            <h1 className="hero__title">
              Compressed Vector<br />
              <span className="hero__title--emphasis">Retrieval Engine</span>
            </h1>
          </Reveal>
          
          <Reveal delay={100}>
            <p className="hero__description">
              Production-grade RAG infrastructure with 8x memory compression 
              and perfect recall. Built on TurboQuant, QJL, and PolarQuant.
            </p>
          </Reveal>
          
          <Reveal delay={150}>
            <div className="hero__actions">
              <div className="hero__install">
                <div className="hero__install-header">
                  <div className="hero__install-traffic-lights">
                    <span className="hero__install-traffic-light hero__install-traffic-light--red" />
                    <span className="hero__install-traffic-light hero__install-traffic-light--yellow" />
                    <span className="hero__install-traffic-light hero__install-traffic-light--green" />
                  </div>
                  <div className="hero__install-tabs">
                    <button 
                      className={`hero__install-tab ${installTab === 'pip' ? 'hero__install-tab--active' : ''}`}
                      onClick={() => setInstallTab('pip')}
                    >
                      pip
                    </button>
                    <button 
                      className={`hero__install-tab ${installTab === 'npm' ? 'hero__install-tab--active' : ''}`}
                      onClick={() => setInstallTab('npm')}
                    >
                      npm
                    </button>
                  </div>
                </div>
                <div className="hero__install-content">
                  <span className="hero__install-prompt">$</span>
                  <code>{installCommand}</code>
                  <CopyButton text={installCommand} className="hero__install-copy" />
                </div>
              </div>
            </div>
            
            <Link to="/docs" className="hero__docs-link">
              Read the Docs <span className="hero__docs-arrow">&rarr;</span>
            </Link>
          </Reveal>
        </div>
        
        <Reveal delay={200} className="hero__metrics">
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric__value">8x</span>
              <span className="metric__label">Memory Compression</span>
            </div>
            <div className="metric">
              <span className="metric__value">1.000</span>
              <span className="metric__label">Recall@10 (exact)</span>
            </div>
            <div className="metric">
              <span className="metric__value">6,209</span>
              <span className="metric__label">Queries/sec</span>
            </div>
            <div className="metric">
              <span className="metric__value">18.3</span>
              <span className="metric__unit">MB</span>
              <span className="metric__label">100K vectors</span>
            </div>
          </div>
        </Reveal>
      </Container>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Features
// ─────────────────────────────────────────────────────────────────────────────

function Features() {
  const features = [
    {
      title: 'Adaptive Search',
      description: 'Two-stage retrieval with binary sketch pre-filter and LUT refinement. Automatically selects exact or fast mode based on index size.',
      detail: 'SimHash + POPCNT'
    },
    {
      title: 'Graph-Augmented',
      description: 'Entity extraction, community detection, and hybrid dense + graph search with full explainability for retrieved passages.',
      detail: 'HybridRetriever'
    },
    {
      title: 'Sidecar Architecture',
      description: 'Drop-in alongside your existing database. No migration required. Gradual traffic shifting with zero downtime deployment.',
      detail: 'No lock-in'
    },
    {
      title: 'Production Ready',
      description: 'Atomic persistence, thread-safe HTTP service, request tracking, latency metrics, CORS, and comprehensive test coverage.',
      detail: '104+ tests'
    }
  ]

  return (
    <Section id="features">
      <Container>
        <Reveal>
          <h2 className="section__title">
            Engineered for production
          </h2>
          <p className="section__subtitle">
            A complete retrieval engine with the infrastructure teams need to ship confidently.
          </p>
        </Reveal>
        
        <div className="features-grid">
          {features.map((feature, i) => (
            <Reveal key={feature.title} delay={i * 50}>
              <article className="feature-card">
                <h3 className="feature-card__title">{feature.title}</h3>
                <p className="feature-card__description">{feature.description}</p>
                <span className="feature-card__detail">{feature.detail}</span>
              </article>
            </Reveal>
          ))}
        </div>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance (Updated with both scales and modes)
// ─────────────────────────────────────────────────────────────────────────────

type BenchmarkRow = {
  name: string
  recall: string
  qps: string
  memory: string
  highlight: boolean
  note?: string
}

function Performance() {
  const [activeScale, setActiveScale] = useState<'small' | 'large'>('large')

  const smallBenchmarks: BenchmarkRow[] = [
    { name: 'TurboRAG 4-bit', recall: '1.000', qps: '6,209', memory: '0.08 MB', highlight: true },
    { name: 'Exact float32', recall: '1.000', qps: '26,774', memory: '0.49 MB', highlight: false },
    { name: 'FAISS Flat', recall: '1.000', qps: '32,384', memory: '0.49 MB', highlight: false },
    { name: 'FAISS HNSW', recall: '1.000', qps: '23,640', memory: '0.55 MB', highlight: false },
    { name: 'FAISS IVF-PQ', recall: '0.990', qps: '27,438', memory: '0.49 MB', highlight: false },
  ]

  const largeBenchmarks: BenchmarkRow[] = [
    { name: 'TurboRAG exact', recall: '1.000', qps: '67', memory: '18.3 MB', highlight: true, note: 'Perfect recall guaranteed' },
    { name: 'TurboRAG fast', recall: '0.975', qps: '274', memory: '18.3 MB', highlight: true, note: '4x faster, 97.5% recall' },
    { name: 'Exact float32', recall: '1.000', qps: '240', memory: '146.5 MB', highlight: false },
    { name: 'FAISS Flat', recall: '1.000', qps: '232', memory: '146.5 MB', highlight: false },
    { name: 'FAISS HNSW', recall: '0.645', qps: '1,928', memory: '152.6 MB', highlight: false, note: 'Recall drops significantly' },
  ]

  const benchmarks = activeScale === 'small' ? smallBenchmarks : largeBenchmarks
  const scaleInfo = activeScale === 'small' 
    ? '1K vectors · 128-dim · 100 queries · k=10 · 4-bit'
    : '100K vectors · 384-dim · 200 queries · k=10 · 3-bit'

  return (
    <Section id="performance" dark>
      <Container>
        <Reveal>
          <span className="section__label">Performance</span>
          <h2 className="section__title section__title--light">
            Perfect recall at a fraction of the memory
          </h2>
        </Reveal>
        
        <Reveal delay={50}>
          <div className="benchmark-tabs">
            <button 
              className={`benchmark-tab ${activeScale === 'large' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveScale('large')}
            >
              100K Scale
            </button>
            <button 
              className={`benchmark-tab ${activeScale === 'small' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveScale('small')}
            >
              1K Scale
            </button>
          </div>
        </Reveal>
        
        <Reveal delay={100}>
          <div className="benchmark-table">
            <div className="benchmark-table__header">
              <span>Backend</span>
              <span>Recall@10</span>
              <span>QPS</span>
              <span>Memory</span>
            </div>
            
            {benchmarks.map((row) => (
              <div 
                key={row.name}
                className={`benchmark-table__row ${row.highlight ? 'benchmark-table__row--highlight' : ''}`}
              >
                <span className="benchmark-table__name">
                  {row.name}
                  {row.note && <span className="benchmark-table__note">{row.note}</span>}
                </span>
                <span className="benchmark-table__value">{row.recall}</span>
                <span className="benchmark-table__value">{row.qps}</span>
                <span className="benchmark-table__value">{row.memory}</span>
              </div>
            ))}
          </div>
          
          <p className="benchmark-context">{scaleInfo}</p>
          
          {activeScale === 'large' && (
            <div className="benchmark-highlight">
              <strong>Key insight:</strong> At 100K scale, TurboRAG uses 8x less memory (18.3 MB vs 146.5 MB) 
              while maintaining perfect recall in exact mode. Fast mode achieves 97.5% recall with 4x throughput.
              FAISS HNSW drops to 64.5% recall.
            </div>
          )}
        </Reveal>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

function Pipeline() {
  const steps = [
    { phase: '01', title: 'Ingest', description: 'Token-aware document chunking', code: 'turborag.chunker' },
    { phase: '02', title: 'Compress', description: 'Quantize to 2-4 bit precision', code: 'turborag.compress' },
    { phase: '03', title: 'Index', description: 'Build sharded index with sketches', code: 'TurboIndex.add()' },
    { phase: '04', title: 'Search', description: 'Two-stage retrieval pipeline', code: 'index.search()' },
  ]

  return (
    <Section>
      <Container>
        <Reveal>
          <h2 className="section__title">How it works</h2>
        </Reveal>
        
        <div className="pipeline-grid">
          {steps.map((step, i) => (
            <Reveal key={step.phase} delay={i * 50}>
              <div className="pipeline-step">
                <span className="pipeline-step__phase">{step.phase}</span>
                <h3 className="pipeline-step__title">{step.title}</h3>
                <p className="pipeline-step__description">{step.description}</p>
                <code className="pipeline-step__code">{step.code}</code>
              </div>
            </Reveal>
          ))}
        </div>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Quickstart
// ─────────────────────────────────────────────────────────────────────────────

function Quickstart() {
  const [activeTab, setActiveTab] = useState<'python' | 'nodejs' | 'cli' | 'docker'>('python')

  const code = {
    python: `import numpy as np
from turborag import TurboIndex

# Create index
index = TurboIndex(dim=128, bits=4, seed=42)

# Add vectors
vectors = np.random.randn(1000, 128).astype(np.float32)
ids = [f"doc-{i}" for i in range(1000)]
index.add(vectors, ids)

# Search (auto mode selects exact or fast)
results = index.search(vectors[0], k=10)
for chunk_id, score in results:
    print(f"{chunk_id}: {score:.4f}")`,

    nodejs: `import { TurboRAG } from 'turborag';

// Connect to TurboRAG server
const client = new TurboRAG('http://localhost:8080');

// Query with vector
const results = await client.query({
  queryVector: new Float32Array(128).fill(0.1),
  topK: 10
});

// Or query with text (auto-embeds)
const textResults = await client.queryText({
  text: 'What is machine learning?',
  topK: 5
});

// Batch queries for high throughput
const batchResults = await client.queryBatch({
  queries: [vector1, vector2, vector3],
  topK: 10
});`,

    cli: `# Install
pip install turborag[all]

# Import existing embeddings
turborag import-existing-index \\
  --input ./embeddings.jsonl \\
  --index ./my_index --bits 3

# Start server
turborag serve --index ./my_index --port 8080

# Query
curl -X POST http://localhost:8080/query \\
  -H "Content-Type: application/json" \\
  -d '{"query_vector": [...], "top_k": 10}'`,

    docker: `# Build
docker build -t turborag .

# Run with volume mount
docker run -p 8080:8080 \\
  -v ./my_index:/data/index \\
  turborag serve --index /data/index

# Health check
curl http://localhost:8080/health`
  }

  return (
    <Section id="quickstart" className="quickstart-section">
      <Container>
        <div className="quickstart-layout">
          <Reveal className="quickstart-info">
            <span className="section__label">Get Started</span>
            <h2 className="section__title">
              From zero to search<br />in minutes
            </h2>
            <p className="section__subtitle">
              TurboRAG ships as a Python package and Node.js client 
              for graph retrieval, HTTP serving, and MCP integration.
            </p>
            
            <div className="extras-list">
              <code>pip install turborag[all]</code>
              <code>npm install turborag</code>
            </div>
          </Reveal>
          
          <Reveal delay={100} className="quickstart-code">
            <div className="code-block">
              <div className="code-block__header">
                <div className="code-block__traffic-lights">
                  <span className="code-block__traffic-light code-block__traffic-light--red" />
                  <span className="code-block__traffic-light code-block__traffic-light--yellow" />
                  <span className="code-block__traffic-light code-block__traffic-light--green" />
                </div>
                <span className="code-block__title">
                  {activeTab === 'python' ? 'example.py' : activeTab === 'nodejs' ? 'example.ts' : activeTab === 'cli' ? 'Terminal' : 'Dockerfile'}
                </span>
                <CopyButton text={code[activeTab]} className="code-block__copy" />
              </div>
              <div className="code-block__tabs">
                {(['python', 'nodejs', 'cli', 'docker'] as const).map(tab => (
                  <button
                    key={tab}
                    className={`code-block__tab ${activeTab === tab ? 'code-block__tab--active' : ''}`}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab === 'cli' ? 'CLI' : tab === 'nodejs' ? 'Node.js' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
              <pre className="code-block__pre">
                <code>{code[activeTab]}</code>
              </pre>
            </div>
          </Reveal>
        </div>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture
// ─────────────────────────────────────────────────────────────────────────────

function Architecture() {
  const modules = [
    { name: 'index.py', desc: 'TurboIndex with search, batch, delete, update' },
    { name: 'compress.py', desc: 'Rotation, quantization, LUT scoring' },
    { name: '_cscore.c', desc: 'C kernel with POPCNT + fused byte LUT' },
    { name: 'graph.py', desc: 'Entity graph with persistence + communities' },
    { name: 'hybrid.py', desc: 'Dense + graph hybrid retrieval' },
    { name: 'service.py', desc: 'Starlette HTTP with CORS + metrics' },
    { name: 'chunker.py', desc: 'Token-aware PDF/MD/text chunking' },
    { name: 'mcp_server.py', desc: 'MCP stdio server for agents' },
  ]

  return (
    <Section className="architecture-section">
      <Container>
        <Reveal>
          <h2 className="section__title">Inside the engine</h2>
          <p className="section__subtitle">Modular design. Use what you need.</p>
        </Reveal>
        
        <div className="modules-grid">
          {modules.map((mod, i) => (
            <Reveal key={mod.name} delay={i * 30}>
              <div className="module-card">
                <code className="module-card__name">{mod.name}</code>
                <span className="module-card__desc">{mod.desc}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AI Agent Prompt
// ─────────────────────────────────────────────────────────────────────────────

function AIAgentPrompt() {
  const prompt = `Implement TurboRAG in my project. Read https://turborag.jpdz.app/llms.txt first, then analyze my codebase and integrate it.`

  return (
    <section className="ai-agent-section">
      <Container>
        <div className="ai-agent-banner">
          <div className="ai-agent-banner__content">
            <span className="ai-agent-banner__badge">AI Agents</span>
            <p className="ai-agent-banner__text">
              Give this prompt to your AI coding agent to integrate TurboRAG automatically
            </p>
          </div>
          <div className="ai-agent-banner__prompt">
            <code className="ai-agent-banner__code">{prompt}</code>
            <CopyButton text={prompt} className="ai-agent-banner__copy" />
          </div>
        </div>
      </Container>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Documentation Links (Home page)
// ─────────────────────────────────────────────────────────────────────────────

function DocumentationLinks() {
  const docs = [
    { title: 'Installation', description: 'Setup and installation options', slug: 'installation' },
    { title: 'Client SDKs', description: 'Node.js, Go, Ruby clients', slug: 'client-sdks' },
    { title: 'Architecture', description: 'System design and components', slug: 'architecture' },
    { title: 'Benchmarking', description: 'Performance testing guide', slug: 'benchmarking' },
    { title: 'Adoption Guide', description: 'Integration strategies', slug: 'adoption' },
    { title: 'Service API', description: 'HTTP endpoints reference', slug: 'service' },
  ]

  return (
    <Section dark>
      <Container>
        <Reveal>
          <span className="section__label">Documentation</span>
          <h2 className="section__title section__title--light">
            Everything you need to get started
          </h2>
        </Reveal>
        
        <div className="docs-grid">
          {docs.map((doc, i) => (
            <Reveal key={doc.slug} delay={i * 40}>
              <Link to={`/docs/${doc.slug}`} className="doc-card">
                <h3 className="doc-card__title">{doc.title}</h3>
                <p className="doc-card__description">{doc.description}</p>
                <span className="doc-card__arrow">Read</span>
              </Link>
            </Reveal>
          ))}
        </div>
      </Container>
    </Section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Footer
// ─────────────────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer className="footer">
      <Container>
        <div className="footer__content">
          <div className="footer__brand">
            <span className="footer__logo">TurboRAG</span>
            <p className="footer__tagline">
              Compressed vector retrieval for production AI systems.
            </p>
          </div>
          
          <div className="footer__links">
            <div className="footer__column">
              <span className="footer__heading">Documentation</span>
              <Link to="/docs/installation">Installation</Link>
              <Link to="/docs/architecture">Architecture</Link>
              <Link to="/docs/benchmarking">Benchmarking</Link>
              <Link to="/docs/adoption">Adoption Guide</Link>
            </div>
            
            <div className="footer__column">
              <span className="footer__heading">Project</span>
              <a href="https://github.com/ratnam1510/turborag" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="https://pypi.org/project/turborag/" target="_blank" rel="noopener noreferrer">PyPI</a>
              <Link to="/docs/spec-status">Changelog</Link>
            </div>
          </div>
        </div>
        
        <div className="footer__bottom">
          <span>Open source under MIT License</span>
        </div>
      </Container>
    </footer>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Documentation Content
// ─────────────────────────────────────────────────────────────────────────────

const docsContent: Record<string, { title: string; content: string }> = {
  'installation': {
    title: 'Installation',
    content: `# Installation

TurboRAG can be installed from PyPI or from source. Both methods support the same optional extras.

## From PyPI

\`\`\`bash
# Core (compression, indexing, CLI)
pip install turborag

# With HTTP service
pip install turborag[serve]

# With MCP agent server
pip install turborag[mcp]

# With known DB adapter integrations (Neon/Supabase/Pinecone/Qdrant/Chroma)
pip install turborag[adapters]

# With local embedding support (sentence-transformers)
pip install turborag[embed]

# With graph retrieval (networkx, Leiden clustering)
pip install turborag[graph]

# With PDF/document ingestion (pdfminer, tiktoken)
pip install turborag[ingest]

# Everything
pip install turborag[all]

# Everything + adapter integrations
pip install turborag[all,adapters]

# Everything + dev/test dependencies
pip install turborag[all,dev]
\`\`\`

## From Source (Git Clone)

\`\`\`bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag

# Core only
pip install -e .

# With specific extras
pip install -e '.[serve]'
pip install -e '.[mcp]'
pip install -e '.[adapters]'

# Everything
pip install -e '.[all]'

# Development (includes pytest)
pip install -e '.[all,dev]'
\`\`\`

## Requirements

- Python >= 3.11
- A C compiler (gcc or clang) is recommended for the C scoring kernel — TurboRAG auto-compiles it on first use. If no compiler is available, it falls back to a pure Python implementation.

## Extras Reference

| Extra | What it adds | When you need it |
|---|---|---|
| *(core)* | numpy, scipy, click | Always installed |
| \`serve\` | starlette, uvicorn | Running \`turborag serve\` (HTTP API) |
| \`mcp\` | mcp | Running \`turborag mcp\` (MCP stdio server) |
| \`adapters\` | psycopg, supabase, pinecone, qdrant-client, chromadb | Plug-and-play known DB integrations |
| \`embed\` | sentence-transformers | Local text embedding with \`--model\` |
| \`graph\` | networkx, leidenalg, python-igraph | Graph-augmented retrieval |
| \`ingest\` | pdfminer.six, tiktoken | PDF extraction, token-aware chunking |
| \`all\` | All of the above | Full feature set |
| \`dev\` | pytest | Running the test suite |

## Verify Installation

\`\`\`bash
# Check the CLI works
turborag --help

# Check C kernel availability
python -c "from turborag._cscore_wrapper import is_available; print('C kernel:', 'available' if is_available() else 'not available (using Python fallback)')"

# Run tests (requires dev extra)
pytest
\`\`\`

## Docker

No local Python installation needed — run TurboRAG directly from Docker:

\`\`\`bash
# Build
docker build -t turborag .

# Serve an index
docker run -p 8080:8080 -v ./my_index:/data/index turborag \\
  turborag serve --index /data/index --host 0.0.0.0

# Run MCP server over stdio
docker run -i turborag turborag mcp --index /data/index
\`\`\`

## What Each Component Needs

### CLI (import, query, describe, benchmark)

\`\`\`bash
pip install turborag
\`\`\`

For \`turborag adapt\` with known backends:

\`\`\`bash
pip install turborag[adapters]
\`\`\`

### HTTP Service

\`\`\`bash
pip install turborag[serve]
turborag serve --index ./my_index --host 0.0.0.0 --port 8080
\`\`\`

### MCP Server (Claude Desktop, agent integration)

\`\`\`bash
pip install turborag[mcp]
turborag mcp --index ./my_index
\`\`\`

## Node.js / TypeScript Client

For Node.js and TypeScript applications, install the official client from npm:

\`\`\`bash
npm install turborag
\`\`\`

The client connects to a running TurboRAG HTTP server:

\`\`\`typescript
import { TurboRAG } from 'turborag';

const client = new TurboRAG('http://localhost:8080');

// Query with vector
const results = await client.query({
  queryVector: new Float32Array(128),
  topK: 10
});

// Query with text (auto-embeds server-side)
const textResults = await client.queryText({
  text: 'What is machine learning?',
  topK: 5
});
\`\`\`

Requirements:
- Node.js >= 18.0.0
- A running TurboRAG HTTP server (\`pip install turborag[serve]\`)

See the [Service documentation](/docs/service) for server setup.`
  },
  'architecture': {
    title: 'Architecture',
    content: `# TurboRAG Architecture

## Goal

Build a retrieval library that combines:

- a compressed dense retrieval layer,
- an entity graph for multi-hop expansion, and
- a clean path toward adapters, ingestion, benchmarking, and services.

## Current Module Map

### \`turborag.compress\`

Implements the compression primitives used by the rest of the package.

- \`generate_rotation(dim, seed)\` builds the deterministic orthogonal transform.
- \`quantize_qjl(...)\` and \`dequantize_qjl(...)\` encode and decode packed vectors.
- \`compressed_dot(...)\` provides prototype approximate scoring.

### \`turborag.index\`

Implements \`TurboIndex\`, the current storage and retrieval engine.

- Validates vector shapes and IDs.
- Normalises vectors before compression by default.
- Persists shard files plus a \`config.json\` and \`rotation.npy\`.
- Loads saved indexes with \`numpy.memmap\` so large indexes do not need full RAM materialisation.

### \`turborag.graph\`

Implements \`GraphBuilder\`.

- Uses a structured extraction prompt.
- Caches LLM output in SQLite.
- Builds a \`networkx.Graph\` when graph dependencies are installed.
- Assigns communities with Leiden when available, otherwise uses connected components.

### \`turborag.hybrid\`

Implements \`HybridRetriever\`.

- Runs dense retrieval through \`TurboIndex\`.
- Detects query entities by graph node name match.
- Expands graph paths with breadth-first search.
- Merges dense and graph candidates and emits explanations.

### \`turborag.adapters\`

Implements the adoption layer for existing RAG systems.

- \`ExistingRAGAdapter\` lets TurboRAG reuse an application's current chunk IDs and metadata store.
- \`resolve_records_backend(...)\` adapts common existing DB client shapes into TurboRAG's hydration callback.
- \`ExistingRAGAdapter.from_existing_backend(...)\` binds TurboRAG directly on top of existing backends.
- \`adapters.backends\` includes known backend builders for Postgres/Neon/Supabase, Pinecone, Qdrant, and Chroma.
- \`adapters.config\` provides persisted adapter config loading for plug-and-play mode.
- \`TurboVectorStore\` exposes familiar vector-store-like methods.

### \`turborag.service\`

Implements the HTTP deployment surface.

- Exposes health and index metadata endpoints.
- Accepts vector queries and optional text queries.
- Supports ID-only retrieval responses (\`hydrate=false\`) for low-memory sidecar operation.
- Supports startup without snapshot load (\`load_snapshot=False\`).

## Storage Layout

Saved indexes follow this layout:

\`\`\`text
index/
  config.json
  records.jsonl
  rotation.npy
  sketch.bin
  adapter.json (optional)
  shards/
    shard_000.bin
    shard_000.ids.json
    shard_000.sketch.bin
\`\`\`

## Query Flow

TurboRAG supports three search modes: \`auto\` (default), \`exact\`, and \`fast\`.

1. Accept a float32 query vector.
2. Optionally L2-normalise it.
3. Rotate it with the stored orthogonal matrix.
4. Quantize and bit-pack it using the configured bit width.
5. **Mode selection**:
   - \`exact\`: Score every vector with the LUT-based C scoring kernel.
   - \`fast\`: Use POPCNT Hamming distance to pre-filter, then refine with LUT.
   - \`auto\`: Select based on index size.
6. Merge top candidates across shards.
7. If running through an adapter, hydrate from existing application record store.
8. Return structured results.`
  },
  'benchmarking': {
    title: 'Benchmarking',
    content: `# Benchmarking

## Latest Results

### Small Scale (1K vectors, 128-dim, 100 queries, k=10, 4-bit)

| Backend | Recall@10 | MRR | QPS | Memory |
|---|---|---|---|---|
| **TurboRAG 4-bit** | **1.000** | **1.000** | **6,209** | **0.08 MB** |
| Exact float32 | 1.000 | 1.000 | 26,774 | 0.49 MB |
| FAISS Flat | 1.000 | 1.000 | 32,384 | 0.49 MB |
| FAISS HNSW | 1.000 | 1.000 | 23,640 | 0.55 MB |
| FAISS IVF-PQ | 0.990 | 0.990 | 27,438 | < 0.49 MB |

### Large Scale (100K vectors, 384-dim, 200 queries, k=10, 3-bit)

| Backend | Recall@10 | MRR | QPS | Memory |
|---|---|---|---|---|
| **TurboRAG exact** | **1.000** | **1.000** | **67** | **18.3 MB** |
| **TurboRAG fast** | **0.975** | **0.975** | **274** | **18.3 MB** |
| Exact float32 | 1.000 | 1.000 | 240 | 146.5 MB |
| FAISS Flat | 1.000 | 1.000 | 232 | 146.5 MB |
| FAISS HNSW | 0.645 | 0.645 | 1,928 | 152.6 MB |

TurboRAG maintains perfect recall in exact mode at both scales. The \`fast\` mode uses a binary sketch head with POPCNT pre-filtering followed by LUT refine, achieving 4x throughput over exact with 97.5% recall.

## Running Benchmarks

### TurboRAG-Only Benchmark

\`\`\`bash
turborag benchmark \\
  --index ./turborag_sidecar \\
  --queries ./queries.jsonl \\
  --k 10
\`\`\`

### Side-By-Side Benchmark

\`\`\`bash
turborag benchmark \\
  --index ./turborag_sidecar \\
  --queries ./queries.jsonl \\
  --dataset ./corpus.jsonl \\
  --baseline exact \\
  --baseline faiss-flat \\
  --baseline faiss-hnsw \\
  --k 10
\`\`\`

## One-Command Local Comparison

\`\`\`bash
./scripts/benchmark_compare.sh
\`\`\``
  },
  'adoption': {
    title: 'Adoption Guide',
    content: `# Adoption Guide

## Goal

Let an existing RAG user adopt TurboRAG without rewriting application code or migrating their metadata database.

## The Easiest Migration Pattern

Keep your current system exactly as-is for:

- chunk storage,
- metadata,
- source documents,
- primary database tables,
- document hydration at query time.

Add TurboRAG only as a compressed sidecar index over the same chunk IDs you already use.

## Sidecar Architecture

1. Export or intercept the embeddings you already compute today.
2. Build a \`TurboIndex\` keyed by the same \`chunk_id\` values already stored in your application.
3. Keep your current database untouched.
4. At query time, ask TurboRAG only for ranked IDs.
5. Hydrate those IDs from your existing database or document store.

## Generic Integration Example

\`\`\`python
from turborag.adapters.compat import ExistingRAGAdapter

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings_matrix,
    ids=ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
    shard_size=100_000,
)

results = adapter.query("What did the CFO say about capex guidance?", k=5)
\`\`\`

## Plug-And-Play Adapter CLI

Use \`turborag adapt\` to configure known backends once, then just run \`turborag serve\`.

\`\`\`bash
# Auto-detect from environment
turborag adapt --index ./turborag_sidecar

# Force a specific backend
turborag adapt --index ./turborag_sidecar --backend supabase

# Dedicated backend commands
turborag adapt supabase --index ./turborag_sidecar
turborag adapt neon --index ./turborag_sidecar
turborag adapt pinecone --index ./turborag_sidecar
turborag adapt qdrant --index ./turborag_sidecar
turborag adapt chroma --index ./turborag_sidecar
\`\`\`

## Known Database Examples

### Neon / Postgres

\`\`\`python
from turborag.adapters.backends import build_neon_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

fetch_records = build_neon_fetch_records(
    dsn="postgresql://user:pass@host/db",
    table="public.chunks",
    id_column="chunk_id",
    text_column="text",
)

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings_matrix,
    ids=ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
)
\`\`\`

### Supabase

\`\`\`python
from supabase import create_client
from turborag.adapters.backends import build_supabase_fetch_records

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
fetch_records = build_supabase_fetch_records(supabase, table="chunks")
\`\`\`

### Pinecone

\`\`\`python
from pinecone import Pinecone
from turborag.adapters.backends import build_pinecone_fetch_records

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("your-index")
fetch_records = build_pinecone_fetch_records(index, namespace="prod")
\`\`\`

### Qdrant

\`\`\`python
from qdrant_client import QdrantClient
from turborag.adapters.backends import build_qdrant_fetch_records

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
fetch_records = build_qdrant_fetch_records(client, collection_name="chunks")
\`\`\`

### Chroma

\`\`\`python
import chromadb
from turborag.adapters.backends import build_chroma_fetch_records

chroma = chromadb.PersistentClient(path="./chroma")
collection = chroma.get_collection("chunks")
fetch_records = build_chroma_fetch_records(collection)
\`\`\`

## Fast Trial Path With The CLI

\`\`\`bash
turborag import-existing-index --input ./dataset.jsonl --index ./turborag_sidecar --bits 3
turborag query --index ./turborag_sidecar --query-vector '[0.1, 0.2, 0.3]' --top-k 5

# ID-only results (lowest memory)
turborag query --index ./turborag_sidecar --query-vector '[0.1, 0.2, 0.3]' --top-k 5 --ids-only
\`\`\`

## Migration Strategy

### Stage 1
Build TurboRAG offline from the embeddings and chunk IDs you already have.

### Stage 2
Run TurboRAG in shadow mode beside the current retriever.

### Stage 3
Compare top-k overlap, latency, and answer quality.

### Stage 4
Flip retrieval traffic to TurboRAG while keeping the old database and hydration path unchanged.`
  },
  'service': {
    title: 'Service API',
    content: `# Service API

## Usage

TurboRAG can run as a local or remote HTTP sidecar:

\`\`\`bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
\`\`\`

With text-query support and multiple workers:

\`\`\`bash
turborag serve \\
  --index ./turborag_sidecar \\
  --model sentence-transformers/all-MiniLM-L6-v2 \\
  --workers 4
\`\`\`

Low-memory sidecar mode (no local snapshot hydration):

\`\`\`bash
turborag serve \\
  --index ./turborag_sidecar \\
  --no-load-snapshot
\`\`\`

Plug-and-play backend mode using adapter config:

\`\`\`bash
turborag adapt --index ./turborag_sidecar
turborag serve --index ./turborag_sidecar
\`\`\`

## Endpoints

### \`GET /health\`

Returns a minimal readiness payload.

### \`GET /index\`

Returns the sidecar configuration and hydration mode:

- \`hydration_source\`: \`local_snapshot\`, \`external_backend\`, \`hybrid\`, or \`id_only\`
- \`allow_unhydrated\`: whether unresolved hits are returned as ID-only results
- \`text_query_enabled\`: whether \`query_text\` is enabled

### \`GET /metrics\`

Returns latency histograms and error counts.

### \`POST /query\`

Accepts \`query_vector\` or \`query_text\`.

Optional: \`hydrate\` (boolean, default \`true\`) — when \`false\`, returns ID-only results.

\`\`\`bash
# Vector query
curl -X POST http://localhost:8080/query \\
  -H 'content-type: application/json' \\
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'

# ID-only response (hydrate in your existing DB)
curl -X POST http://localhost:8080/query \\
  -H 'content-type: application/json' \\
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5,"hydrate":false}'
\`\`\`

### \`POST /query/batch\`

Batch multiple vector queries in a single request. Supports \`"hydrate": false\`.

### \`POST /ingest\`

Appends embedding-level records to the index.

### \`POST /ingest-text\`

Ingest raw text with automatic chunking and embedding.

## Production Features

- **CORS**: Enabled by default. Restrict with \`--cors-origins\`.
- **Metrics**: Hit \`GET /metrics\` for latency histograms.
- **Request tracking**: Pass \`X-Request-Id\` header.
- **Structured logging**: Use \`--log-format json\`.
- **Multi-worker**: Use \`--workers N\`.
- **Strict hydration**: Use \`--require-hydrated\` to drop any hit that cannot be hydrated.`
  },
  'current-rag-rollout': {
    title: 'Current RAG Rollout',
    content: `# Current RAG Rollout

This document explains exactly how TurboRAG fits into an existing RAG system.

The key point is simple: **TurboRAG does not need to replace the current database. It can sit beside it.**

## What Stays The Same

- the chunk table or document store,
- the source-of-truth metadata database,
- the chunk IDs,
- the answer-generation LLM call,
- existing auth, tenancy, and business logic.

## What TurboRAG Replaces

TurboRAG replaces only the retrieval engine that ranks chunk IDs from embeddings.

## Flow 1: Embedded In The Existing App

### One-time setup

1. Export existing embeddings and chunk IDs.
2. Build a TurboRAG sidecar index.
3. Keep the current database untouched.

### Runtime query flow

1. The user asks a question.
2. The app embeds the query.
3. TurboRAG returns ranked \`chunk_id\` values.
4. The app fetches those from the existing database.
5. Normal answer generation continues.

## Flow 2: HTTP Sidecar

### One-time setup

1. Export current embeddings and chunk IDs.
2. Build a TurboRAG sidecar index.
3. Configure adapter binding from environment (optional):

\`\`\`bash
turborag adapt --index ./turborag_sidecar
\`\`\`

4. Start the service:

\`\`\`bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
\`\`\`

Memory-minimal startup (no snapshot hydration load):

\`\`\`bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080 --no-load-snapshot
\`\`\`

### Query examples

\`\`\`bash
# Full hydrated response
curl -X POST http://localhost:8080/query \\
  -H 'content-type: application/json' \\
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'

# ID-only response (hydrate in your existing DB)
curl -X POST http://localhost:8080/query \\
  -H 'content-type: application/json' \\
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5,"hydrate":false}'
\`\`\`

## Rollout Phases

### Phase A: Evaluate offline
Build a sidecar and run benchmarks.

### Phase B: Shadow mode
Run TurboRAG in parallel with current retriever.

### Phase C: Partial cutover
Route small percentage of traffic to TurboRAG.

### Phase D: Full retrieval cutover
Make TurboRAG the retrieval layer.`
  },
  'import-existing': {
    title: 'Import Existing Data',
    content: `# Import Existing RAG Data

## Goal

Let a team with an already-running RAG stack build a TurboRAG sidecar from the embeddings and chunk records they already have.

## Install

\`\`\`bash
# CLI import and query
pip install turborag

# With HTTP sidecar serving after import
pip install turborag[serve]

# With plug-and-play known DB adapters
pip install turborag[adapters]
\`\`\`

## Supported Import Formats

### JSONL

Each line should contain:

- \`chunk_id\` or \`id\`
- \`text\`, \`page_content\`, or \`content\`
- \`embedding\`

Optional: \`source_doc\`, \`page_num\`, \`section\`, \`metadata\`

### NPZ

The \`.npz\` file must contain:

- \`embeddings\`
- \`ids\`

## CLI Flow

### Import

\`\`\`bash
turborag import-existing-index \\
  --input ./dataset.jsonl \\
  --index ./turborag_sidecar \\
  --bits 3
\`\`\`

### Query by vector

\`\`\`bash
turborag query \\
  --index ./turborag_sidecar \\
  --query-vector '[0.1, 0.2, 0.3]' \\
  --top-k 5
\`\`\`

### Configure existing DB adapter (one-time)

\`\`\`bash
turborag adapt --index ./turborag_sidecar
\`\`\`

### Query IDs only (no local hydration)

\`\`\`bash
turborag query \\
  --index ./turborag_sidecar \\
  --query-vector '[0.1, 0.2, 0.3]' \\
  --top-k 5 \\
  --ids-only
\`\`\`

### Query by text

\`\`\`bash
turborag query \\
  --index ./turborag_sidecar \\
  --query 'What changed in capex guidance?' \\
  --model sentence-transformers/all-MiniLM-L6-v2 \\
  --top-k 5
\`\`\`

## Python Flow

\`\`\`python
from turborag.ingest import build_sidecar_index, load_dataset

dataset = load_dataset("./dataset.jsonl")
result = build_sidecar_index(dataset, "./turborag_sidecar", bits=3)
print(result.index_path)
\`\`\`

## What Gets Written

The sidecar directory contains:

- the TurboRAG index files,
- \`records.jsonl\` for local hydration when self-contained.

When no snapshot is loaded, TurboRAG can still run as an ID-only sidecar.`
  },
  'spec-status': {
    title: 'Spec Status',
    content: `# Spec Status

## Source Papers

The implementation is built from:

- [TurboQuant](https://arxiv.org/abs/2504.19874) — ICLR 2026
- [Quantized Johnson-Lindenstrauss (QJL)](https://arxiv.org/abs/2406.03482) — AAAI 2025
- [PolarQuant](https://arxiv.org/abs/2502.02617) — AISTATS 2026

## Fully Implemented

- Core package scaffolding and packaging
- Deterministic rotation generation
- Bit-packed scalar quantization and LUT-based scoring
- **C scoring kernel** with fused byte-triplet acceleration
- \`TurboIndex\` with memmap-backed persistence, batch search, delete, update
- \`GraphBuilder\` with Leiden community detection and graph persistence
- \`HybridRetriever\` for dense + graph retrieval
- **Token-aware document chunking** for PDF, markdown, and plain text
- **Production HTTP service** with CORS, metrics, batch queries
- **MCP server** with query, describe, and ingest tools
- **Dockerfile** for production deployment
- **104+ tests**

## Performance Summary

### Small Scale (1K x 128, 4-bit)

| | Recall@10 | QPS |
|---|---|---|
| **TurboRAG 4-bit** | 1.000 | 6,209 |
| Exact float32 | 1.000 | 26,774 |

### Large Scale (100K x 384, 3-bit)

| | Recall@10 | QPS |
|---|---|---|
| **TurboRAG exact** | 1.000 | 67 |
| **TurboRAG fast** | 0.975 | 274 |
| Exact float32 | 1.000 | 240 |
| FAISS HNSW | 0.645 | 1,928 |

TurboRAG achieves perfect recall at both scales with 8x memory compression.`
  },
  'llm-request-model': {
    title: 'LLM Request Model',
    content: `# LLM Request Model

## Short Answer

TurboQuant-style compression by itself does not inherently reduce the number of LLM requests in a standard RAG pipeline.

It changes the retrieval layer, not the final generation call.

## Standard RAG Flow

1. Embed query.
2. Search vector store.
3. Fetch chunks.
4. Send context to the LLM.

If you switch only the retrieval engine to TurboRAG, the LLM request count usually stays the same.

## Where TurboRAG Can Change LLM Usage

### Index Time

If you enable the graph layer, TurboRAG can require more LLM calls during indexing for:

- entity extraction,
- relationship extraction,
- community summarisation.

### Query Time

Query-time stays close to a normal RAG pattern:

- one embedding step,
- one retrieval step,
- one final LLM answer step.

## When Overall LLM Spend Can Go Down

Even if request count stays the same, total spend can still go down if TurboRAG helps you:

- retrieve more accurate chunks,
- avoid retries and fallback prompts,
- send less irrelevant context,
- get acceptable quality with smaller prompts.`
  },
  'spec-decisions': {
    title: 'Spec Decisions',
    content: `# Spec Decisions

This document records the places where the TurboQuant/QJL/PolarQuant papers left room for interpretation and the exact decisions used in this implementation.

## Reference Papers

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [Quantized Johnson-Lindenstrauss (QJL)](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

## 1. Quantization Strategy

### Spec Tension

The papers describe QJL, SRHT projection, sign quantization, and scalar min/max quantization with different tradeoffs.

### Decision

Implement rotated scalar quantization with fixed symmetric bounds.

### Why

- It is internally consistent.
- It supports incremental indexing without having to re-encode older shards.
- It keeps query-time and index-time calibration identical.
- It is easy to test and reason about.

## 2. Value Range

### Spec Tension

The pseudocode calibrates each batch with per-dimension min/max values, but query vectors need the same calibration to be comparable with stored vectors.

### Decision

Normalize vectors by default and quantize into a fixed range of \`[-1.0, 1.0]\`.

### Why

- Rotated unit vectors remain bounded.
- The representation is stable across batches.
- Incremental adds work naturally.

## 3. Rotation Persistence

### Spec Tension

The narrative suggests storing only a seed and algorithm identifier, while the storage layout explicitly includes \`rotation.npy\`.

### Decision

Store both the seed and the concrete \`rotation.npy\` matrix.

### Why

- Loading is deterministic and exact.
- The index file format is explicit.
- It avoids regeneration differences across SciPy versions.

## 4. Search Kernel

### Spec Tension

The PDF points toward a future SIMD or popcount-optimised kernel but also shows a dequantize-then-dot prototype.

### Decision

Start with the prototype dequantize-then-dot implementation, now upgraded to a LUT-based C kernel with fused byte-triplet acceleration.

### Why

- It is portable.
- It establishes a correctness baseline.
- The C kernel provides production-grade performance.

## 5. Graph Dependencies

### Spec Tension

The graph layer is central to the long-term product, but the spec also argues for a lightweight core install.

### Decision

Keep graph dependencies optional and degrade gracefully when they are not installed.

### Why

- The core package remains light.
- Dense retrieval remains useful on its own.
- The architecture cleanly accommodates the graph layer.`
  },
  'implementation-plan': {
    title: 'Implementation Plan',
    content: `# Implementation Plan

## Status Snapshot

The exact mapping from the PDF to current implementation status is documented in the Spec Status page.

### Implemented

- Core package and packaging metadata.
- Rotation generation.
- Bit-packing quantization utilities.
- LUT-based C scoring kernel with fused byte-triplet acceleration.
- In-memory and on-disk compressed vector indexing.
- Binary sketch head and adaptive two-stage search (auto/exact/fast modes).
- Batch search and threaded shard scanning.
- Token-aware chunking for PDF, markdown, and plain text.
- Document ingestion with metadata propagation.
- Graph construction hooks with caching and persistence.
- Hybrid retrieval scaffolding.
- Compatibility adapters for sidecar adoption with existing RAG stacks.
- Known backend helpers for Postgres/Neon/Supabase, Pinecone, Qdrant, and Chroma.
- Adapter config persistence and CLI (\`turborag adapt\`) for plug-and-play backend binding.
- Existing-embedding import/sync flow and sidecar CLI commands.
- Benchmark harness, side-by-side baseline comparison, and local artifact generation.
- HTTP sidecar service with CORS, metrics, request tracking, batch queries, and text ingestion.
- Low-memory sidecar mode with optional ID-only query payloads.
- MCP query, describe, and ingest tools over stdio.
- Domain-specific exception hierarchy.
- Docker packaging with multi-stage production build.
- Core documentation and test coverage (104+ tests).

### Not Yet Implemented

- Richer embedding model wrappers and provider integrations.
- Cross-encoder reranking.
- Deeper LangChain integration and a full LlamaIndex adapter.
- Persisted graph/community storage and graph API surfaces.
- Reproducible published benchmark artifacts on real external datasets.

## Recommended Build Sequence

### Phase 1 (Done)
Stabilise the core numeric path: batch search, C scoring kernel, top-k optimisation.

### Phase 2 (Done)
Make ingestion real: token-aware chunking, PDF extraction, metadata store.

### Phase 3 (Done)
Production hardening: CORS, metrics, request ID tracking, multi-worker support, Docker.

### Phase 4 (Mostly Done)
Deepen graph retrieval and distribution: raw-text ingest, MCP ingest tool, adapters.

### Phase 5 (Done)
Adaptive search and sketch head: binary SimHash, POPCNT pre-filtering, auto/exact/fast modes.

### Phase 6
Proof and positioning: publish real-world benchmarks, example notebooks, TestPyPI release.`
  },
  'client-sdks': {
    title: 'Client SDKs',
    content: `# Client SDKs

Language-agnostic clients for the TurboRAG HTTP API. Each is a thin typed wrapper with zero external dependencies.

## Available Clients

| Language | Package | Install | Dependencies |
|---|---|---|---|
| **Python** | \`turborag\` | \`pip install turborag\` | numpy, scipy |
| **Node.js / TypeScript** | \`turborag\` | \`npm install turborag\` | None (native fetch) |
| **Go** | \`turborag\` | \`go get github.com/ratnam1510/turborag/clients/go\` | None (net/http) |
| **Ruby** | \`turborag\` | \`gem install turborag\` | None (net/http) |

All clients talk to the same HTTP API — start a server with:

\`\`\`bash
# Docker (no Python needed)
docker run -p 8080:8080 -v ./my_index:/data/index turborag \\
  turborag serve --index /data/index --host 0.0.0.0

# Or pip
pip install turborag[serve]
turborag serve --index ./my_index --port 8080
\`\`\`

## Node.js / TypeScript

\`\`\`bash
npm install turborag
\`\`\`

\`\`\`typescript
import { TurboRAG } from "turborag";

const client = new TurboRAG("http://localhost:8080");

// Query by vector
const { results } = await client.query({
  vector: [0.1, 0.2, 0.3],
  topK: 5,
});

// Query IDs only (application hydrates from existing DB)
const idOnly = await client.queryIds({
  vector: [0.1, 0.2, 0.3],
  topK: 5,
});

// Query by text (requires --model on the server)
const textResults = await client.queryText({
  text: "What changed in capex guidance?",
  topK: 5,
});

// Batch query
const batch = await client.queryBatch({
  queries: [
    { vector: [0.1, 0.2, 0.3] },
    { vector: [0.4, 0.5, 0.6] },
  ],
  topK: 5,
});

// Ingest records
await client.ingest({
  records: [
    {
      chunk_id: "c1",
      text: "Capital expenditure guidance increased.",
      embedding: [0.1, 0.2, 0.3],
      source_doc: "q3_call.pdf",
    },
  ],
});

// Health & metrics
const health = await client.health();
const info = await client.index();
const metrics = await client.metrics();
\`\`\`

### TypeScript API

| Method | Description |
|---|---|
| \`query({ vector, topK, hydrate })\` | Search by embedding vector |
| \`queryIds({ vector, topK })\` | Search returning IDs only |
| \`queryText({ text, topK })\` | Search by text (requires \`--model\`) |
| \`queryBatch({ queries, topK })\` | Batch vector search |
| \`queryBatchIds({ queries, topK })\` | Batch search with ID-only hits |
| \`ingest({ records })\` | Add records with embeddings |
| \`ingestText({ text, sourceDoc })\` | Ingest raw text |
| \`health()\` | Health check |
| \`index()\` | Index config and stats |
| \`metrics()\` | Latency and error metrics |

## Go

\`\`\`bash
go get github.com/ratnam1510/turborag/clients/go
\`\`\`

\`\`\`go
package main

import (
	"context"
	"fmt"
	"log"

	turborag "github.com/ratnam1510/turborag/clients/go"
)

func main() {
	client := turborag.New("http://localhost:8080")
	ctx := context.Background()

	// Query by vector
	results, err := client.Query(ctx, []float64{0.1, 0.2, 0.3}, 5)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range results.Results {
		fmt.Printf("%s  %.4f  %s\\n", r.ChunkID, r.Score, r.Text)
	}

	// Query by text (requires --model)
	textResults, err := client.QueryText(ctx, "What changed?", 5)

	// Batch query
	batch, err := client.QueryBatch(ctx, [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}, 5)

	// Health & info
	health, _ := client.Health(ctx)
	info, _ := client.Index(ctx)
	fmt.Println(health.Status, info.IndexSize)
}
\`\`\`

### Go API

| Method | Description |
|---|---|
| \`Query(ctx, vector, topK)\` | Search by embedding vector |
| \`QueryText(ctx, text, topK)\` | Search by text (requires \`--model\`) |
| \`QueryBatch(ctx, vectors, topK)\` | Batch vector search |
| \`Ingest(ctx, records)\` | Add records with embeddings |
| \`IngestText(ctx, text, sourceDoc)\` | Ingest raw text |
| \`Health(ctx)\` | Health check |
| \`Index(ctx)\` | Index config and stats |

## Ruby

\`\`\`bash
gem install turborag
\`\`\`

\`\`\`ruby
require "turborag"

client = TurboRAG::Client.new("http://localhost:8080")

# Query by vector
results = client.query(vector: [0.1, 0.2, 0.3], top_k: 5)
results["results"].each do |r|
  puts "#{r["chunk_id"]}  #{r["score"]}  #{r["text"]}"
end

# Query by text (requires --model)
results = client.query_text(text: "What changed?", top_k: 5)

# Batch query
batch = client.query_batch(
  queries: [{ vector: [0.1, 0.2, 0.3] }, { vector: [0.4, 0.5, 0.6] }],
  top_k: 5
)

# Ingest
client.ingest(records: [
  {
    chunk_id: "c1",
    text: "Capital expenditure guidance increased.",
    embedding: [0.1, 0.2, 0.3],
    source_doc: "q3_call.pdf",
  }
])

# Health & metrics
health = client.health
info = client.index
metrics = client.metrics
\`\`\`

### Ruby API

| Method | Description |
|---|---|
| \`query(vector:, top_k:)\` | Search by embedding vector |
| \`query_text(text:, top_k:)\` | Search by text (requires \`--model\`) |
| \`query_batch(queries:, top_k:)\` | Batch vector search |
| \`ingest(records:)\` | Add records with embeddings |
| \`ingest_text(text:, source_doc:)\` | Ingest raw text |
| \`health\` | Health check |
| \`index\` | Index config and stats |
| \`metrics\` | Latency and error metrics |

## Benchmarks

Performance benchmarks against a **100K vector index** (384 dimensions, 3-bit quantization).

### Memory Usage (The Key Metric)

| Metric | Value |
|---|---|
| **Float32 baseline** | 146.5 MB |
| **TurboRAG index** | 14.3 MB |
| **Compression ratio** | **10.7x** |
| **Memory savings** | **91%** |

The TurboRAG server runs at ~73 MB total RSS (includes Python/uvicorn overhead). The actual compressed index uses only **14.3 MB** for 100K vectors.

### Client Memory Footprint

| Client | Max Heap | Max RSS |
|---|---|---|
| **Node.js / TypeScript** | 11.3 MB | 96.8 MB |
| **Ruby** | — | 30.6 MB |

### Query Performance

| Client | Single Query | Batch (50) | Ingest |
|---|---|---|---|
| **Node.js** | 30.7 QPS (32.6ms) | 5.7 QPS | 1,307 rec/s |
| **Ruby** | 25.8 QPS (38.8ms) | 6.3 QPS | 1,013 rec/s |

> **Why this matters**: TurboRAG achieves **10x memory compression** while maintaining high recall (0.975+). Client choice has minimal impact — the server handles the heavy lifting. All clients use native HTTP with zero external dependencies.

### Run Your Own Benchmarks

\`\`\`bash
# TypeScript (includes memory tracking)
cd clients/typescript && npm run benchmark

# Ruby (includes memory tracking)
cd clients/ruby && ruby benchmark.rb
\`\`\`

## Adding A New Language

Each client is ~100 lines wrapping \`POST\`/\`GET\` with JSON serialization. To add a new language:

1. Create \`clients/<language>/\`
2. Implement the 7 API methods above
3. Use native HTTP — no external dependencies
4. Match the type signatures from the TypeScript client`
  },
  'cli-reference': {
    title: 'CLI Reference',
    content: `# CLI Reference

Complete reference for all TurboRAG command-line tools. Install with \`pip install turborag\`.

## Global Options

\`\`\`bash
turborag [OPTIONS] COMMAND [ARGS]...
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--log-level\` | \`INFO\` | Logging verbosity: \`debug\`, \`info\`, \`warning\`, \`error\`, \`critical\` |
| \`--log-format\` | \`text\` | Output format: \`text\` or \`json\` |
| \`--help\` | — | Show help and exit |

## Commands

### turborag serve

Start the HTTP API server for TurboRAG retrieval.

\`\`\`bash
turborag serve --index ./turborag_sidecar [OPTIONS]
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--index\` | *required* | Path to TurboRAG index directory |
| \`--host\` | \`127.0.0.1\` | Bind address |
| \`--port\` | \`8080\` | Port number |
| \`--workers\` | \`1\` | Number of uvicorn workers |
| \`--model\` | — | Sentence-transformers model for text queries |
| \`--cors-origins\` | \`*\` | Comma-separated allowed CORS origins |
| \`--adapter-config\` | — | Path to adapter config JSON from \`turborag adapt\` |
| \`--no-load-snapshot\` | — | Don't load records.jsonl into memory (use with external DBs) |
| \`--require-hydrated\` | — | Drop hits that can't be hydrated |

**Examples:**

\`\`\`bash
# Basic server
turborag serve --index ./my_index

# Production with multiple workers
turborag serve --index ./my_index --host 0.0.0.0 --port 8080 --workers 4

# With text query support
turborag serve --index ./my_index --model sentence-transformers/all-MiniLM-L6-v2

# Low-memory mode with external database
turborag serve --index ./my_index --no-load-snapshot --adapter-config ./adapter.json
\`\`\`

---

### turborag query

Query a TurboRAG index from the command line.

\`\`\`bash
turborag query --index ./turborag_sidecar [OPTIONS]
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--index\` | *required* | Path to TurboRAG index directory |
| \`--query\` | — | Query text to embed locally |
| \`--model\` | — | Sentence-transformers model for \`--query\` |
| \`--query-vector\` | — | Inline JSON array query vector |
| \`--query-vector-file\` | — | Path to JSON file with query vector |
| \`--top-k\` | \`5\` | Number of results |
| \`--ids-only\` | — | Return only chunk_id and score (minimal memory) |

**Examples:**

\`\`\`bash
# Query by vector
turborag query --index ./my_index --query-vector '[0.1, 0.2, 0.3, ...]' --top-k 10

# Query by text (requires model)
turborag query --index ./my_index --query "What is TurboRAG?" --model all-MiniLM-L6-v2

# ID-only results (lowest memory)
turborag query --index ./my_index --query-vector-file ./query.json --ids-only
\`\`\`

---

### turborag import-existing-index

Build a TurboRAG sidecar index from existing embeddings.

\`\`\`bash
turborag import-existing-index --input ./data.jsonl --index ./turborag_sidecar [OPTIONS]
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--input\` | *required* | Embeddings file (JSONL or NPZ) |
| \`--index\` | *required* | Output index directory |
| \`--format\` | \`auto\` | Input format: \`auto\`, \`jsonl\`, or \`npz\` |
| \`--bits\` | \`3\` | Quantization bits (1-8) |
| \`--shard-size\` | \`100000\` | Vectors per shard |
| \`--seed\` | \`42\` | Random seed for reproducibility |
| \`--no-normalize\` | — | Disable vector normalization |
| \`--no-record-snapshot\` | — | Skip writing records.jsonl |

**Input Format (JSONL):**

\`\`\`json
{"chunk_id": "c1", "embedding": [0.1, 0.2, ...], "text": "...", "source_doc": "file.pdf"}
{"chunk_id": "c2", "embedding": [0.3, 0.4, ...], "text": "...", "source_doc": "file.pdf"}
\`\`\`

**Examples:**

\`\`\`bash
# Basic import
turborag import-existing-index --input ./embeddings.jsonl --index ./my_index

# With higher precision
turborag import-existing-index --input ./embeddings.jsonl --index ./my_index --bits 4

# From NumPy archive
turborag import-existing-index --input ./embeddings.npz --index ./my_index --format npz
\`\`\`

---

### turborag adapt

Configure plug-and-play adapters for existing databases.

\`\`\`bash
turborag adapt [OPTIONS] COMMAND
\`\`\`

| Option | Description |
|---|---|
| \`--index\` | Index path for auto mode |
| \`--config\` | Config target path for auto mode |
| \`--backend\` | Force backend: \`postgres\`, \`neon\`, \`supabase\`, \`pinecone\`, \`qdrant\`, \`chroma\` |
| \`--option\` | Key=value overrides (can be repeated) |

**Subcommands:**

| Command | Description |
|---|---|
| \`supabase\` | Configure Supabase adapter |
| \`pinecone\` | Configure Pinecone adapter |
| \`qdrant\` | Configure Qdrant adapter |
| \`chroma\` | Configure Chroma adapter |
| \`neon\` | Configure Neon/Postgres adapter |
| \`set\` | Create/update adapter config |
| \`show\` | Show current adapter config |
| \`remove\` | Remove adapter config |
| \`demo\` | Print example command for a backend |

**Examples:**

\`\`\`bash
# Auto-detect and configure from environment
turborag adapt --index ./my_index

# Configure Supabase explicitly
turborag adapt supabase --index ./my_index \\
  --url "$SUPABASE_URL" \\
  --key "$SUPABASE_KEY" \\
  --table chunks

# Configure Pinecone
turborag adapt pinecone --index ./my_index \\
  --api-key "$PINECONE_API_KEY" \\
  --index-name my-pinecone-index

# Configure Qdrant
turborag adapt qdrant --index ./my_index \\
  --url "http://localhost:6333" \\
  --collection-name chunks

# Configure Chroma
turborag adapt chroma --index ./my_index \\
  --path ./chroma_db \\
  --collection-name chunks

# Show current config
turborag adapt show --index ./my_index

# Remove adapter config
turborag adapt remove --index ./my_index
\`\`\`

---

### turborag mcp

Start an MCP (Model Context Protocol) server over stdio.

\`\`\`bash
turborag mcp --index ./turborag_sidecar
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--index\` | *required* | Path to TurboRAG index directory |

Requires: \`pip install turborag[mcp]\`

**Usage with Claude Desktop:**

Add to your Claude Desktop config:

\`\`\`json
{
  "mcpServers": {
    "turborag": {
      "command": "turborag",
      "args": ["mcp", "--index", "/path/to/index"]
    }
  }
}
\`\`\`

---

### turborag benchmark

Run retrieval benchmarks against a TurboRAG index.

\`\`\`bash
turborag benchmark --index ./turborag_sidecar --queries ./queries.jsonl [OPTIONS]
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--index\` | *required* | Path to TurboRAG index directory |
| \`--queries\` | *required* | JSONL file with test queries |
| \`--dataset\` | — | Corpus embeddings for baseline comparison |
| \`--baseline\` | — | Baseline backend: \`exact\`, \`faiss-flat\`, \`faiss-hnsw\`, \`faiss-ivfpq\` |
| \`--output\` | — | Path to write JSON benchmark results |
| \`--k\` | \`10\` | Number of results for recall/MRR |
| \`--json-output\` | — | Emit full JSON report |

**Query File Format (JSONL):**

\`\`\`json
{"query_id": "q1", "query_vector": [0.1, 0.2, ...], "relevant_ids": ["c1", "c5"]}
{"query_id": "q2", "query_vector": [0.3, 0.4, ...], "relevant_ids": ["c2", "c8"]}
\`\`\`

**Examples:**

\`\`\`bash
# Basic benchmark
turborag benchmark --index ./my_index --queries ./test_queries.jsonl

# With baseline comparison
turborag benchmark --index ./my_index --queries ./test_queries.jsonl \\
  --dataset ./corpus.jsonl \\
  --baseline exact \\
  --baseline faiss-hnsw

# Export results
turborag benchmark --index ./my_index --queries ./test_queries.jsonl \\
  --output ./benchmark_results.json --json-output
\`\`\`

---

### turborag describe-index

Show configuration of an existing TurboRAG index.

\`\`\`bash
turborag describe-index --index ./turborag_sidecar
\`\`\`

| Option | Default | Description |
|---|---|---|
| \`--index\` | *required* | Path to TurboRAG index directory |

**Output includes:**

- Vector dimensions
- Quantization bits
- Number of vectors
- Shard configuration
- Search mode (auto/exact/fast)
- Index size on disk

---

## Quick Reference

\`\`\`bash
# Create index from existing embeddings
turborag import-existing-index --input data.jsonl --index ./my_index

# Configure database adapter
turborag adapt supabase --index ./my_index

# Start HTTP server
turborag serve --index ./my_index --port 8080

# Start MCP server (for Claude Desktop)
turborag mcp --index ./my_index

# Query from command line
turborag query --index ./my_index --query "search text" --model all-MiniLM-L6-v2

# Run benchmarks
turborag benchmark --index ./my_index --queries queries.jsonl

# Inspect index
turborag describe-index --index ./my_index
\`\`\`

## Environment Variables

| Variable | Used By | Description |
|---|---|---|
| \`SUPABASE_URL\` | \`turborag adapt supabase\` | Supabase project URL |
| \`SUPABASE_KEY\` | \`turborag adapt supabase\` | Supabase API key |
| \`PINECONE_API_KEY\` | \`turborag adapt pinecone\` | Pinecone API key |
| \`QDRANT_URL\` | \`turborag adapt qdrant\` | Qdrant server URL |
| \`QDRANT_API_KEY\` | \`turborag adapt qdrant\` | Qdrant API key (optional) |
| \`DATABASE_URL\` | \`turborag adapt neon\` | Postgres/Neon connection string |`
  },
  'database-adapters': {
    title: 'Database Adapters',
    content: `# Database Adapters

TurboRAG integrates with your existing vector database as a sidecar — no migration required. Your database stays the source of truth for records; TurboRAG handles fast compressed retrieval.

## Supported Databases

| Database | Builder Function | Install |
|---|---|---|
| **PostgreSQL** | \`build_postgres_fetch_records()\` | \`pip install turborag[adapters]\` |
| **Neon** | \`build_neon_fetch_records()\` | \`pip install turborag[adapters]\` |
| **Supabase** | \`build_supabase_fetch_records()\` | \`pip install turborag[adapters]\` |
| **Pinecone** | \`build_pinecone_fetch_records()\` | \`pip install turborag[adapters]\` |
| **Qdrant** | \`build_qdrant_fetch_records()\` | \`pip install turborag[adapters]\` |
| **Chroma** | \`build_chroma_fetch_records()\` | \`pip install turborag[adapters]\` |

## Quick Setup (CLI)

The fastest way to configure a database adapter:

\`\`\`bash
# Auto-detect from environment variables
turborag adapt --index ./my_index

# Or explicitly specify backend
turborag adapt supabase --index ./my_index
turborag adapt pinecone --index ./my_index
turborag adapt qdrant --index ./my_index
turborag adapt chroma --index ./my_index
turborag adapt neon --index ./my_index
\`\`\`

---

## Supabase

### CLI Setup

\`\`\`bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-or-service-key"

turborag adapt supabase --index ./my_index \\
  --table chunks \\
  --id-column chunk_id \\
  --text-column text
\`\`\`

### Python API

\`\`\`python
from supabase import create_client
from turborag.adapters.backends import build_supabase_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

# Create Supabase client
supabase = create_client(
    "https://your-project.supabase.co",
    "your-anon-or-service-key"
)

# Build the fetch function
fetch_records = build_supabase_fetch_records(
    client=supabase,
    table="chunks",
    id_column="chunk_id",        # Your ID column
    text_column="text",          # Your text content column
    source_doc_column="source_doc",
    page_num_column="page_num",
    section_column="section",
    metadata_column="metadata",  # Optional JSON metadata
)

# Create adapter
adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=your_embeddings_matrix,  # numpy array
    ids=your_chunk_ids,                 # list of strings
    query_embedder=your_embedder,
    fetch_records=fetch_records,
    bits=3,
)

# Query
results = adapter.query("What is the revenue?", k=10)
for hit in results.hits:
    print(f"{hit.chunk_id}: {hit.text[:100]}...")
\`\`\`

### Table Schema

\`\`\`sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384),  -- pgvector
    source_doc TEXT,
    page_num INTEGER,
    section TEXT,
    metadata JSONB
);
\`\`\`

---

## Pinecone

### CLI Setup

\`\`\`bash
export PINECONE_API_KEY="your-api-key"

turborag adapt pinecone --index ./my_index \\
  --index-name your-pinecone-index \\
  --namespace prod
\`\`\`

### Python API

\`\`\`python
from pinecone import Pinecone
from turborag.adapters.backends import build_pinecone_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

# Create Pinecone client
pc = Pinecone(api_key="your-api-key")
index = pc.Index("your-index-name")

# Build the fetch function
fetch_records = build_pinecone_fetch_records(
    index=index,
    namespace="prod",           # Optional namespace
    text_key="text",            # Metadata key for text content
    source_doc_key="source_doc",
    page_num_key="page_num",
    section_key="section",
)

# Create adapter
adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=your_embeddings_matrix,
    ids=your_chunk_ids,
    query_embedder=your_embedder,
    fetch_records=fetch_records,
    bits=3,
)

# Query
results = adapter.query("What changed in Q3?", k=10)
\`\`\`

### Pinecone Metadata

Pinecone stores text and metadata in the vector record's metadata field:

\`\`\`python
# When upserting to Pinecone
index.upsert([
    {
        "id": "chunk-1",
        "values": [0.1, 0.2, 0.3, ...],
        "metadata": {
            "text": "The quarterly revenue increased...",
            "source_doc": "earnings.pdf",
            "page_num": 5,
            "section": "Financial Overview"
        }
    }
])
\`\`\`

---

## Qdrant

### CLI Setup

\`\`\`bash
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-api-key"  # Optional

turborag adapt qdrant --index ./my_index \\
  --collection-name chunks
\`\`\`

### Python API

\`\`\`python
from qdrant_client import QdrantClient
from turborag.adapters.backends import build_qdrant_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

# Create Qdrant client
client = QdrantClient(
    url="http://localhost:6333",
    api_key="your-api-key",  # Optional
)

# Build the fetch function
fetch_records = build_qdrant_fetch_records(
    client=client,
    collection_name="chunks",
    text_key="text",            # Payload key for text
    source_doc_key="source_doc",
    page_num_key="page_num",
    section_key="section",
)

# Create adapter
adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=your_embeddings_matrix,
    ids=your_chunk_ids,
    query_embedder=your_embedder,
    fetch_records=fetch_records,
    bits=3,
)
\`\`\`

### Qdrant Payload

\`\`\`python
# When upserting to Qdrant
client.upsert(
    collection_name="chunks",
    points=[
        {
            "id": "chunk-1",
            "vector": [0.1, 0.2, 0.3, ...],
            "payload": {
                "text": "The quarterly revenue increased...",
                "source_doc": "earnings.pdf",
                "page_num": 5,
                "section": "Financial Overview"
            }
        }
    ]
)
\`\`\`

---

## ChromaDB

### CLI Setup

\`\`\`bash
turborag adapt chroma --index ./my_index \\
  --path ./chroma_db \\
  --collection-name chunks
\`\`\`

### Python API

\`\`\`python
import chromadb
from turborag.adapters.backends import build_chroma_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

# Create Chroma client
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("chunks")

# Build the fetch function
fetch_records = build_chroma_fetch_records(collection=collection)

# Create adapter
adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=your_embeddings_matrix,
    ids=your_chunk_ids,
    query_embedder=your_embedder,
    fetch_records=fetch_records,
    bits=3,
)
\`\`\`

### Chroma Collection

\`\`\`python
# When adding to Chroma
collection.add(
    ids=["chunk-1", "chunk-2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["The quarterly revenue...", "Operating expenses..."],
    metadatas=[
        {"source_doc": "earnings.pdf", "page_num": 5},
        {"source_doc": "earnings.pdf", "page_num": 6},
    ]
)
\`\`\`

---

## PostgreSQL / Neon

### CLI Setup

\`\`\`bash
export DATABASE_URL="postgresql://user:pass@host/db"

turborag adapt neon --index ./my_index \\
  --table chunks \\
  --id-column chunk_id \\
  --text-column text
\`\`\`

### Python API

\`\`\`python
from turborag.adapters.backends import build_postgres_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

# Option 1: Using DSN
fetch_records = build_postgres_fetch_records(
    dsn="postgresql://user:pass@host/db",
    table="public.chunks",
    id_column="chunk_id",
    text_column="text",
    source_doc_column="source_doc",
    page_num_column="page_num",
    section_column="section",
    metadata_column="metadata",
)

# Option 2: Using existing connection
import psycopg
conn = psycopg.connect("postgresql://user:pass@host/db")
fetch_records = build_postgres_fetch_records(
    connection=conn,
    table="chunks",
)

# Create adapter
adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=your_embeddings_matrix,
    ids=your_chunk_ids,
    query_embedder=your_embedder,
    fetch_records=fetch_records,
    bits=3,
)
\`\`\`

### Table Schema

\`\`\`sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384),  -- pgvector (optional, TurboRAG doesn't need it)
    source_doc TEXT,
    page_num INTEGER,
    section TEXT,
    metadata JSONB
);

-- Index on chunk_id for fast lookups
CREATE INDEX idx_chunks_id ON chunks(chunk_id);
\`\`\`

---

## Adapter Configuration File

The \`turborag adapt\` CLI generates an \`adapter.json\` file in your index directory:

\`\`\`json
{
  "version": 1,
  "backend": "supabase",
  "url": "\${SUPABASE_URL}",
  "key": "\${SUPABASE_KEY}",
  "table": "chunks",
  "id_column": "chunk_id",
  "text_column": "text"
}
\`\`\`

Environment variables are referenced with \`\${VAR_NAME}\` syntax — secrets stay in your environment, not in config files.

## Using With HTTP Server

\`\`\`bash
# Configure adapter
turborag adapt supabase --index ./my_index

# Start server (auto-loads adapter.json)
turborag serve --index ./my_index --port 8080 --no-load-snapshot
\`\`\`

The \`--no-load-snapshot\` flag tells the server to fetch records from your database instead of loading them into memory.

## Custom Record Resolver

For databases not listed above, implement a custom \`fetch_records\` function:

\`\`\`python
from turborag.adapters.compat import ExistingRAGAdapter

def fetch_records(ids: list[str]) -> list[dict]:
    """Fetch records from your custom database."""
    records = []
    for chunk_id in ids:
        # Your database query here
        row = your_db.get(chunk_id)
        records.append({
            "chunk_id": chunk_id,
            "text": row["content"],
            "source_doc": row.get("file"),
            "page_num": row.get("page"),
            "section": row.get("heading"),
            "metadata": row.get("meta", {}),
        })
    return records

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings,
    ids=ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
)
\`\`\``
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Code Block with Copy for Docs
// ─────────────────────────────────────────────────────────────────────────────

function DocsCodeBlock({ children, className }: { children: string; className?: string }) {
  const language = className?.replace('language-', '') || 'text'
  const code = String(children).replace(/\n$/, '')
  
  return (
    <div className="docs-code-block">
      <div className="docs-code-block__header">
        <div className="docs-code-block__traffic-lights">
          <span className="docs-code-block__traffic-light docs-code-block__traffic-light--red" />
          <span className="docs-code-block__traffic-light docs-code-block__traffic-light--yellow" />
          <span className="docs-code-block__traffic-light docs-code-block__traffic-light--green" />
        </div>
        <span className="docs-code-block__title">{language}</span>
        <CopyButton text={code} className="docs-code-block__copy" />
      </div>
      <pre className="docs-code-block__pre">
        <code>{code}</code>
      </pre>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Documentation Page
// ─────────────────────────────────────────────────────────────────────────────

const docsSidebar = [
  { section: 'Getting Started', items: [
    { title: 'Installation', slug: 'installation' },
    { title: 'CLI Reference', slug: 'cli-reference' },
    { title: 'Client SDKs', slug: 'client-sdks' },
    { title: 'Architecture', slug: 'architecture' },
  ]},
  { section: 'Guides', items: [
    { title: 'Adoption Guide', slug: 'adoption' },
    { title: 'Database Adapters', slug: 'database-adapters' },
    { title: 'RAG Rollout', slug: 'current-rag-rollout' },
    { title: 'Import Existing Data', slug: 'import-existing' },
  ]},
  { section: 'Reference', items: [
    { title: 'Service API', slug: 'service' },
    { title: 'Benchmarking', slug: 'benchmarking' },
    { title: 'LLM Request Model', slug: 'llm-request-model' },
    { title: 'Spec Status', slug: 'spec-status' },
    { title: 'Spec Decisions', slug: 'spec-decisions' },
    { title: 'Implementation Plan', slug: 'implementation-plan' },
  ]},
]

function DocsSidebar({ currentSlug }: { currentSlug?: string }) {
  return (
    <aside className="docs-sidebar">
      {docsSidebar.map(group => (
        <div key={group.section} className="docs-sidebar__group">
          <span className="docs-sidebar__section">{group.section}</span>
          <ul className="docs-sidebar__list">
            {group.items.map(item => (
              <li key={item.slug}>
                <Link 
                  to={`/docs/${item.slug}`} 
                  className={`docs-sidebar__link ${currentSlug === item.slug ? 'docs-sidebar__link--active' : ''}`}
                >
                  {item.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </aside>
  )
}

function DocsIndex() {
  return (
    <div className="docs-layout">
      <DocsSidebar />
      <main className="docs-main">
        <h1 className="docs-title">Documentation</h1>
        <p className="docs-intro">
          Welcome to TurboRAG documentation. Select a topic from the sidebar to get started.
        </p>
        
        <div className="docs-overview">
          {docsSidebar.map(group => (
            <div key={group.section} className="docs-overview__group">
              <h2 className="docs-overview__section">{group.section}</h2>
              <div className="docs-overview__items">
                {group.items.map(item => (
                  <Link 
                    key={item.slug} 
                    to={`/docs/${item.slug}`} 
                    className="docs-overview__card"
                  >
                    <span className="docs-overview__card-title">{item.title}</span>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}

function DocsPage() {
  const { slug } = useParams<{ slug: string }>()
  const doc = slug ? docsContent[slug] : null

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [slug])

  if (!doc) {
    return (
      <div className="docs-layout">
        <DocsSidebar currentSlug={slug} />
        <main className="docs-main">
          <h1>Page not found</h1>
          <p>The requested documentation page does not exist.</p>
          <Link to="/docs">Back to Documentation</Link>
        </main>
      </div>
    )
  }

  return (
    <div className="docs-layout">
      <DocsSidebar currentSlug={slug} />
      <main className="docs-main">
        <article className="docs-article">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
              code: ({ inline, className, children, ...props }: { inline?: boolean; className?: string; children?: React.ReactNode }) => {
                // Check if this is a fenced code block (has language-* class) and not inline
                const isCodeBlock = !inline && className && className.startsWith('language-')
                
                if (isCodeBlock) {
                  return (
                    <DocsCodeBlock className={className}>
                      {String(children)}
                    </DocsCodeBlock>
                  )
                }
                
                // Inline code - just render as styled code element
                return <code className={className} {...props}>{children}</code>
              },
              pre: ({ children }) => {
                // If the child is already a DocsCodeBlock, just return it without wrapping
                return <>{children}</>
              }
            }}
          >
            {doc.content}
          </ReactMarkdown>
        </article>
      </main>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Pages
// ─────────────────────────────────────────────────────────────────────────────

function HomePage() {
  return (
    <>
      <Hero />
      <AIAgentPrompt />
      <Features />
      <Performance />
      <Pipeline />
      <Quickstart />
      <Architecture />
      <DocumentationLinks />
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// App
// ─────────────────────────────────────────────────────────────────────────────

function AppContent() {
  const location = useLocation()
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [displayLocation, setDisplayLocation] = useState(location)

  useEffect(() => {
    if (location !== displayLocation) {
      setIsTransitioning(true)
      const timer = setTimeout(() => {
        setDisplayLocation(location)
        setIsTransitioning(false)
      }, 150)
      return () => clearTimeout(timer)
    }
  }, [location, displayLocation])

  return (
    <>
      <Header />
      <main className={`page-content ${isTransitioning ? 'page-content--exiting' : 'page-content--entering'}`}>
        <Routes location={displayLocation}>
          <Route path="/" element={<HomePage />} />
          <Route path="/docs" element={<DocsIndex />} />
          <Route path="/docs/:slug" element={<DocsPage />} />
        </Routes>
      </main>
      <Footer />
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AppContent />
      </ThemeProvider>
    </BrowserRouter>
  )
}
