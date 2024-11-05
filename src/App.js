import './App.css';
import {useEffect} from 'react'
import Navbar from './components/Navbar';
import Hiro from './components/Hiro';
import Skills from './components/Skills'
import Education from './components/Ed';
import Projects from './components/Projects';
import Footer from './components/Footer';
import AOS from 'aos';
import 'aos/dist/aos.css';
import Experience from './components/Experience';


function App() {
    useEffect(() => {
      document.title = 'Anis Houidi';
      AOS.init();
    }, []);
  return (
    <div className="px-6 lg:px-20 xl:px-36 bg-dark-500">
      <Navbar />
      <Hiro />
      <Skills />
      <Education />
      <Experience/>
      <Projects />
      <Footer />
    </div>
  );
}

export default App;