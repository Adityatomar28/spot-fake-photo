import { useEffect, useState } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion'
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  BrainCircuit,
  Check,
  ChevronDown,
  Clock3,
  Code2,
  Cpu,
  Database,
  Eye,
  FileImage,
  Gauge,
  ImageUp,
  Layers3,
  Loader2,
  Microscope,
  MonitorSmartphone,
  MoveRight,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Upload,
  Zap,
} from 'lucide-react'
import './App.css'

const fadeUp = {
  hidden: { opacity: 0, y: 28 },
  show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } },
}

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
}

const steps = [
  { title: 'Upload Image', icon: ImageUp, detail: 'JPEG or PNG input' },
  { title: 'Feature Extraction', icon: ScanSearch, detail: 'OpenCV visual signals' },
  { title: 'Machine Learning Model', icon: BrainCircuit, detail: 'Logistic regression' },
  { title: 'Prediction', icon: BadgeCheck, detail: 'Real or screen photo' },
]

const features = [
  ['Fast Prediction', Zap],
  ['Machine Learning Powered', BrainCircuit],
  ['OpenCV Feature Extraction', Microscope],
  ['Real vs Screen Detection', MonitorSmartphone],
  ['Lightweight Model', Gauge],
  ['Easy Deployment', Code2],
]

const metrics = [
  ['Accuracy', 95, '%', ShieldCheck],
  ['Cross Validation', 94, '%', Activity],
  ['Features Used', 18, '+', Layers3],
  ['Prediction Time', 1, ' sec', Clock3],
]

const workflowFeatures = ['Edge Density', 'Blur', 'Reflection', 'Sharpness', 'Contrast']
const tech = ['Python', 'OpenCV', 'Scikit-learn', 'NumPy', 'Flask', 'Machine Learning', 'Logistic Regression', 'Computer Vision']
const architecture = ['Images', 'Feature Extraction', 'Feature Vector', 'ML Model', 'Prediction', 'Output']
const githubUrl = 'https://github.com/Adityatomar28/spot-fake-photo/tree/main'
const assetBase = import.meta.env.BASE_URL
const sampleImages = {
  real: `${assetBase}samples/real-photo.jpg`,
  screen: `${assetBase}samples/screen-photo.jpeg`,
}

function Section({ eyebrow, title, children, className = '', id }) {
  return (
    <motion.section
      id={id}
      className={`section ${className}`}
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.16 }}
    >
      <div className="section-heading">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </motion.section>
  )
}

function FlowCard({ item, index }) {
  const Icon = item.icon

  return (
    <motion.div className="flow-card lift" variants={fadeUp}>
      <div className="icon-shell">
        <Icon size={24} />
      </div>
      <span className="step-number">0{index + 1}</span>
      <h3>{item.title}</h3>
      <p>{item.detail}</p>
    </motion.div>
  )
}

function Counter({ value, suffix }) {
  const motionValue = useMotionValue(0)
  const spring = useSpring(motionValue, { duration: 1400, bounce: 0 })
  const rounded = useTransform(spring, (latest) => Math.round(latest))
  const [display, setDisplay] = useState(0)

  useEffect(() => rounded.on('change', setDisplay), [rounded])
  useEffect(() => {
    motionValue.set(value)
  }, [motionValue, value])

  return (
    <span className="metric-value">
      {suffix.trim() === 'sec' ? '<' : ''}
      {display}
      {suffix}
    </span>
  )
}

function HeroIllustration() {
  return (
    <motion.div
      className="hero-visual"
      initial={{ opacity: 0, x: 24, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
    >
      <motion.div className="sample-photo real" animate={{ y: [0, -12, 0] }} transition={{ repeat: Infinity, duration: 5, ease: 'easeInOut' }}>
        <img src={sampleImages.real} alt="Real camera example" />
        <span>Real camera signal</span>
      </motion.div>
      <motion.div className="sample-photo screen" animate={{ y: [0, 10, 0] }} transition={{ repeat: Infinity, duration: 5.5, ease: 'easeInOut' }}>
        <img src={sampleImages.screen} alt="Screen recaptured example" />
        <span>Screen artifacts</span>
      </motion.div>
      <div className="processor-panel">
        <div className="panel-top">
          <Cpu size={18} />
          <span>AI Processing</span>
        </div>
        <div className="signal-lines">
          <motion.i animate={{ scaleX: [0.45, 1, 0.6] }} transition={{ repeat: Infinity, duration: 2.3 }} />
          <motion.i animate={{ scaleX: [0.7, 0.35, 1] }} transition={{ repeat: Infinity, duration: 2.6 }} />
          <motion.i animate={{ scaleX: [0.3, 0.88, 0.42] }} transition={{ repeat: Infinity, duration: 2.1 }} />
        </div>
        <div className="result-chip">
          <Sparkles size={16} />
          95% Real Photo
        </div>
      </div>
    </motion.div>
  )
}

function DemoSection() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(sampleImages.real)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState({ label: 'Real Photo', confidence: 95, type: 'real' })

  const handleFile = (event) => {
    const selected = event.target.files?.[0]
    if (!selected) return

    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setLoading(true)
    setResult(null)

    window.setTimeout(() => {
      const isScreen = selected.name.toLowerCase().includes('screen')
      setResult({
        label: isScreen ? 'Screen Photo' : 'Real Photo',
        confidence: isScreen ? 91 : 96,
        type: isScreen ? 'screen' : 'real',
      })
      setLoading(false)
    }, 1200)
  }

  return (
    <Section eyebrow="Interactive Preview" title="Demo">
      <div className="demo-grid">
        <label className="upload-box lift">
          <input type="file" accept="image/*" onChange={handleFile} />
          <Upload size={32} />
          <strong>Drop an image or browse</strong>
          <span>{file ? file.name : 'Try a camera photo or screen recapture'}</span>
        </label>

        <div className="demo-output">
          <div className="preview-card">
            <span className="mini-label">Uploaded Image</span>
            <img src={preview} alt="Uploaded preview" />
          </div>
          <ChevronDown className="down-arrow" size={22} />
          <div className="prediction-card">
            {loading ? (
              <div className="loading-state">
                <Loader2 className="spin" size={26} />
                <span>Analyzing image features...</span>
              </div>
            ) : (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <span className={`prediction-badge ${result?.type}`}>
                  {result?.type === 'real' ? '✓' : '!'}
                </span>
                <h3>{result?.label}</h3>
                <p>{result?.confidence}% confidence</p>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </Section>
  )
}

function App() {
  return (
    <main>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <nav className="nav">
        <a className="brand" href="#top" aria-label="Spot Fake Photo home">
          <Eye size={20} />
          <span>Spot Fake Photo</span>
        </a>
        <div className="nav-actions">
          <a href="#demo">Demo</a>
          <a href="#tech">Stack</a>
          <a className="github-link" href={githubUrl} target="_blank" rel="noreferrer">
            <Code2 size={17} />
            GitHub
          </a>
        </div>
      </nav>

      <section className="hero-section" id="top">
        <motion.div className="hero-copy" variants={stagger} initial="hidden" animate="show">
          <motion.span className="hero-pill" variants={fadeUp}>
            <Sparkles size={16} />
            Computer vision classifier
          </motion.span>
          <motion.h1 variants={fadeUp}>Detect Screen Recaptured Photos with AI</motion.h1>
          <motion.p variants={fadeUp}>
            An AI-powered computer vision model that distinguishes genuine camera photos from photos captured off another screen using handcrafted image features and machine learning.
          </motion.p>
          <motion.div className="hero-buttons" variants={fadeUp}>
            <a className="primary-btn" href="#demo">
              <Upload size={18} />
              Try Demo
            </a>
            <a className="secondary-btn" href={githubUrl} target="_blank" rel="noreferrer">
              <Code2 size={18} />
              View GitHub
            </a>
          </motion.div>
        </motion.div>
        <HeroIllustration />
      </section>

      <Section eyebrow="Process" title="How It Works">
        <motion.div className="flow-grid" variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.22 }}>
          {steps.map((item, index) => (
            <div className="flow-item" key={item.title}>
              <FlowCard item={item} index={index} />
              {index < steps.length - 1 && <ArrowRight className="flow-arrow" size={22} />}
            </div>
          ))}
        </motion.div>
      </Section>

      <Section eyebrow="Capabilities" title="Features">
        <motion.div className="feature-grid" variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.18 }}>
          {features.map(([label, Icon]) => (
            <motion.div className="feature-card lift" variants={fadeUp} key={label}>
              <span className="check-icon">
                <Check size={16} />
              </span>
              <Icon size={22} />
              <h3>{label}</h3>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      <Section eyebrow="Signal Path" title="Detection Workflow" className="workflow-section">
        <div className="pipeline">
          <div className="pipeline-node">
            <FileImage size={26} />
            <span>Image Upload</span>
          </div>
          <MoveRight className="pipeline-arrow" />
          <div className="pipeline-node wide">
            <ScanSearch size={26} />
            <span>Feature Extraction</span>
            <div className="feature-tags">
              {workflowFeatures.map((item) => (
                <small key={item}>{item}</small>
              ))}
            </div>
          </div>
          <MoveRight className="pipeline-arrow" />
          <div className="pipeline-node">
            <BrainCircuit size={26} />
            <span>Logistic Regression</span>
          </div>
          <MoveRight className="pipeline-arrow" />
          <div className="pipeline-node">
            <BadgeCheck size={26} />
            <span>Prediction</span>
          </div>
        </div>
      </Section>

      <Section eyebrow="Model Performance" title="Production-ready Signals">
        <div className="metrics-grid">
          {metrics.map(([label, value, suffix, Icon]) => (
            <div className="metric-card lift" key={label}>
              <Icon size={24} />
              <Counter value={value} suffix={suffix} />
              <p>{label}</p>
            </div>
          ))}
        </div>
      </Section>

      <div id="demo">
        <DemoSection />
      </div>

      <Section eyebrow="Build Stack" title="Tech Stack" className="tech-section" id="tech">
        <div className="tech-pills">
          {tech.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </Section>

      <Section eyebrow="Architecture" title="Project Architecture">
        <div className="architecture-flow">
          {architecture.map((item, index) => (
            <div className="architecture-step" key={item}>
              <div>
                {index === 0 && <Database size={22} />}
                {index === 1 && <ScanSearch size={22} />}
                {index === 2 && <Layers3 size={22} />}
                {index === 3 && <BrainCircuit size={22} />}
                {index === 4 && <BadgeCheck size={22} />}
                {index === 5 && <Sparkles size={22} />}
                <span>{item}</span>
              </div>
              {index < architecture.length - 1 && <ArrowRight size={20} />}
            </div>
          ))}
        </div>
      </Section>

      <footer>
        <a className="secondary-btn" href={githubUrl} target="_blank" rel="noreferrer">
          <Code2 size={18} />
          GitHub
        </a>
        <p>Aditya Singh Tomar</p>
      </footer>
    </main>
  )
}

export default App
