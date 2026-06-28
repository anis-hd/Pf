import './App.css';
import { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Hiro from './components/Hiro';
import Skills from './components/Skills'
import Education from './components/Ed';
import Projects from './components/Projects';
import Footer from './components/Footer';

import Experience from './components/Experience';
import HyperRaftPage from './components/HyperRaftPage';

// Home page component
function HomePage() {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 relative">
      {/* Left side pattern bar */}
      <div className="side-pattern side-pattern-left" />
      {/* Right side pattern bar */}
      <div className="side-pattern side-pattern-right" />

      <div className="max-w-6xl mx-auto bg-white min-h-screen border-x border-slate-200/80 shadow-md relative z-10 px-6 md:px-12 lg:px-16">
        <Navbar />
        <Hiro />
        <Skills />
        <Education />
        <Experience />
        <Projects />
        <Footer />
      </div>
    </div>
  );
}

function App() {
  useEffect(() => {
    document.title = 'Anis Houidi';
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/hyperraft" element={<HyperRaftPage />} />
      </Routes>
    </Router>
  );
}

export default App;