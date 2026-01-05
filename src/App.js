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
    <div className="px-6 lg:px-20 xl:px-36 bg-gradient-to-b from-dark-500 to-dark-600 min-h-screen">
      <Navbar />
      <Hiro />
      <Skills />
      <Education />
      <Experience />
      <Projects />
      <Footer />
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