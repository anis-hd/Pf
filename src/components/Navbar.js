import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faTimes } from '@fortawesome/free-solid-svg-icons';

export default function Navbar() {
    const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleMouseMove = (e) => {
            const x = (e.clientX / window.innerWidth) * 100;
            const y = (e.clientY / window.innerHeight) * 100;
            setMousePos({ x, y });
        };

        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('scroll', handleScroll);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    const navLinks = [
        { name: 'About', href: '#home' },
        { name: 'Skills', href: '#skills' },
        { name: 'Education', href: '#honors' },
        { name: 'Experience', href: '#experience' },
        { name: 'Projects', href: '#certs' }
    ];

    return (
        <>
            <nav className={`fixed z-50 w-full top-0 left-0 px-6 lg:px-20 xl:px-36 py-4 transition-all duration-300 ${isScrolled ? 'bg-dark-500/95 backdrop-blur-md border-b border-white/10 shadow-lg' : 'bg-transparent'}`}>
                <div className="flex justify-between items-center text-white max-w-7xl mx-auto">
                    {/* Logo */}
                    <a
                        href="#home"
                        className="font-bold text-2xl text-white hover:scale-105 transition-all duration-300"
                    >
                        Anis.
                    </a>

                    {/* Desktop Navigation */}
                    <ul className="hidden md:flex items-center gap-1">
                        {navLinks.map((link, idx) => (
                            <li key={idx}>
                                <a
                                    href={link.href}
                                    className="relative px-4 py-2 text-gray-300 hover:text-white transition-colors duration-300 group"
                                >
                                    {link.name}
                                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-0 h-0.5 bg-gradient-to-r from-purple-500 to-pink-500 group-hover:w-3/4 transition-all duration-300 rounded-full" />
                                </a>
                            </li>
                        ))}
                    </ul>

                    {/* Mobile Menu Button */}
                    <button
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                        className="md:hidden w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-gray-400 hover:text-white hover:border-purple-500/50 transition-all duration-300"
                    >
                        <FontAwesomeIcon icon={isMobileMenuOpen ? faTimes : faBars} />
                    </button>
                </div>
            </nav>

            {/* Mobile Menu */}
            <div className={`fixed inset-0 z-40 md:hidden transition-all duration-300 ${isMobileMenuOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
                {/* Backdrop */}
                <div
                    className="absolute inset-0 bg-black/80 backdrop-blur-sm"
                    onClick={() => setIsMobileMenuOpen(false)}
                />

                {/* Menu Content */}
                <div className={`absolute top-20 left-6 right-6 bg-dark-500/95 backdrop-blur-md border border-white/10 rounded-2xl p-6 transition-all duration-300 ${isMobileMenuOpen ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'}`}>
                    <ul className="space-y-2">
                        {navLinks.map((link, idx) => (
                            <li key={idx}>
                                <a
                                    href={link.href}
                                    onClick={() => setIsMobileMenuOpen(false)}
                                    className="block px-4 py-3 rounded-xl text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-300"
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