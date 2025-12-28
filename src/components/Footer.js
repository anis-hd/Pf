import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHeart, faArrowUp } from '@fortawesome/free-solid-svg-icons';
import { faGithub, faLinkedinIn } from '@fortawesome/free-brands-svg-icons';

export default function Footer() {
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      setMousePos({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const navLinks = [
    { name: 'About', href: '#home' },
    { name: 'Skills', href: '#skills' },
    { name: 'Education', href: '#honors' },
    { name: 'Experience', href: '#experience' },
    { name: 'Projects', href: '#certs' }
  ];

  const socialLinks = [
    { icon: faGithub, href: 'https://github.com/anis-hd', label: 'GitHub' },
    { icon: faLinkedinIn, href: 'https://www.linkedin.com/in/anis-ben-houidi/', label: 'LinkedIn' }
  ];

  return (
    <footer className="relative mt-20 border-t border-white/10">
      {/* Background Gradient */}
      <div
        className="absolute inset-0 pointer-events-none overflow-hidden"
        style={{
          background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, rgba(99, 102, 241, 0.05) 0%, transparent 50%)`
        }}
      />

      <div className="relative z-10 py-12">
        {/* Main Footer Content */}
        <div className="max-w-6xl mx-auto px-6">
          {/* Top Section */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-8 mb-10">
            {/* Logo/Name */}
            <div className="text-center md:text-left">
              <h3
                className="text-3xl font-bold bg-clip-text text-transparent"
                style={{
                  backgroundImage: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, #6366f1, #d946ef, #0ea5e9)`
                }}
              >
                Anis Ben Houidi
              </h3>
              <p className="text-gray-500 mt-2 text-sm">
                Computer Science Engineer • AI Engineer
              </p>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-4">
              {socialLinks.map((social, idx) => (
                <a
                  key={idx}
                  href={social.href}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={social.label}
                  className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-gray-400 hover:text-white hover:border-purple-500/50 hover:bg-white/10 transition-all duration-300 group"
                >
                  <FontAwesomeIcon
                    icon={social.icon}
                    className="text-xl group-hover:scale-110 transition-transform duration-300"
                  />
                </a>
              ))}
            </div>
          </div>

          {/* Divider with Gradient */}
          <div className="h-px w-full bg-gradient-to-r from-transparent via-purple-500/50 to-transparent mb-10" />

          {/* Navigation Links */}
          <nav className="flex flex-wrap justify-center gap-x-8 gap-y-4 mb-10">
            {navLinks.map((link, idx) => (
              <a
                key={idx}
                href={link.href}
                className="text-gray-400 hover:text-white transition-colors duration-300 relative group"
              >
                {link.name}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-purple-500 to-pink-500 group-hover:w-full transition-all duration-300" />
              </a>
            ))}
          </nav>

          {/* Bottom Section */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {/* Copyright */}
            <p className="text-gray-500 text-sm text-center md:text-left">
              © {new Date().getFullYear()} Anis Ben Houidi. All rights reserved.
            </p>


            {/* Scroll to Top */}
            <button
              onClick={scrollToTop}
              className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center text-white hover:shadow-lg hover:shadow-purple-500/30 hover:scale-110 transition-all duration-300"
              aria-label="Scroll to top"
            >
              <FontAwesomeIcon icon={faArrowUp} />
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
}