import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faTimes } from '@fortawesome/free-solid-svg-icons';

export default function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };

        window.addEventListener('scroll', handleScroll);
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    const navLinks = [
        { name: 'About', href: '#home' },
        { name: 'Tech Stack', href: '#skills' },
        { name: 'Education', href: '#honors' },
        { name: 'Experience', href: '#experience' },
        { name: 'Projects', href: '#certs' }
    ];

    return (
        <>
            <nav className={`fixed z-50 max-w-5xl w-full top-0 left-1/2 -translate-x-1/2 px-6 md:px-12 lg:px-16 py-4 transition-all duration-300 border-x border-transparent ${isScrolled ? 'bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-sm !border-slate-200/80' : 'bg-transparent'}`}>
                <div className="flex justify-between items-center text-slate-800 max-w-7xl mx-auto">
                    {/* Logo */}
                    <a
                        href="#home"
                        className="font-bold text-2xl text-slate-900 hover:scale-105 transition-all duration-300"
                    >
                        Anis.
                    </a>

                    {/* Desktop Navigation */}
                    <ul className="hidden md:flex items-center gap-1">
                        {navLinks.map((link, idx) => (
                            <li key={idx}>
                                <a
                                    href={link.href}
                                    className="relative px-4 py-2 text-slate-600 hover:text-slate-900 transition-colors duration-300 group"
                                >
                                    {link.name}
                                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-0 h-0.5 bg-blue-600 group-hover:w-3/4 transition-all duration-300 rounded-full" />
                                </a>
                            </li>
                        ))}
                    </ul>

                    {/* Mobile Menu Button */}
                    <button
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                        className="md:hidden w-10 h-10 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500 hover:text-slate-900 hover:border-slate-300 hover:bg-slate-200 transition-all duration-300"
                    >
                        <FontAwesomeIcon icon={isMobileMenuOpen ? faTimes : faBars} />
                    </button>
                </div>
            </nav>

            {/* Mobile Menu */}
            <div className={`fixed inset-0 z-40 md:hidden transition-all duration-300 ${isMobileMenuOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
                {/* Backdrop */}
                <div
                    className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
                    onClick={() => setIsMobileMenuOpen(false)}
                />

                {/* Menu Content */}
                <div className={`absolute top-20 left-6 right-6 bg-white/95 backdrop-blur-md border border-slate-200 shadow-xl rounded-2xl p-6 transition-all duration-300 ${isMobileMenuOpen ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'}`}>
                    <ul className="space-y-2">
                        {navLinks.map((link, idx) => (
                            <li key={idx}>
                                <a
                                    href={link.href}
                                    onClick={() => setIsMobileMenuOpen(false)}
                                    className="block px-4 py-3 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition-all duration-300"
                                >
                                    {link.name}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </>
    );
}