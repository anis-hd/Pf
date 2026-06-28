import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowUp } from '@fortawesome/free-solid-svg-icons';
import { faGithub, faLinkedinIn } from '@fortawesome/free-brands-svg-icons';

export default function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const navLinks = [
    { name: 'About', href: '#home' },
    { name: 'Tech Stack', href: '#skills' },
    { name: 'Education', href: '#honors' },
    { name: 'Experience', href: '#experience' },
    { name: 'Projects', href: '#certs' }
  ];

  const socialLinks = [
    { icon: faGithub, href: 'https://github.com/anis-hd', label: 'GitHub' },
    { icon: faLinkedinIn, href: 'https://www.linkedin.com/in/anis-ben-houidi/', label: 'LinkedIn' }
  ];

  return (
    <footer className="relative mt-20 border-t border-slate-200">
      <div className="relative z-10 py-12">
        {/* Main Footer Content */}
        <div className="max-w-6xl mx-auto px-6">
          {/* Top Section */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-8 mb-10">
            {/* Logo/Name */}
            <div className="text-center md:text-left">
              <h3 className="text-3xl font-bold text-slate-900">
                Anis Ben Houidi
              </h3>
              <p className="text-slate-500 mt-2 text-sm">
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
                  className="w-12 h-12 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500 hover:text-slate-900 hover:border-slate-300 hover:bg-slate-200 transition-all duration-300 group"
                >
                  <FontAwesomeIcon
                    icon={social.icon}
                    className="text-xl group-hover:scale-110 transition-transform duration-300"
                  />
                </a>
              ))}
            </div>
          </div>

          {/* Divider */}
          <div className="h-px w-full bg-slate-200 mb-10" />

          {/* Navigation Links */}
          <nav className="flex flex-wrap justify-center gap-x-8 gap-y-4 mb-10">
            {navLinks.map((link, idx) => (
              <a
                key={idx}
                href={link.href}
                className="text-slate-500 hover:text-slate-900 transition-colors duration-300 relative group"
              >
                {link.name}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-600 group-hover:w-full transition-all duration-300" />
              </a>
            ))}
          </nav>

          {/* Bottom Section */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {/* Copyright */}
            <p className="text-slate-500 text-sm text-center md:text-left">
              © {new Date().getFullYear()} Anis Ben Houidi. All rights reserved.
            </p>

            {/* Scroll to Top */}
            <button
              onClick={scrollToTop}
              className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white hover:bg-blue-700 hover:scale-110 shadow-sm transition-all duration-300"
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