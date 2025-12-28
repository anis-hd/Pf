export default function Navbar() {
    return (
        <div className='fixed z-50 bg-dark-500/80 backdrop-blur-md w-full top-0 left-0 px-8 py-4 lg:px-20 xl:px-36 border-b border-white/10'>
            <div className="flex justify-between items-center text-white">
                <p className="font-bold text-2xl">Anis.</p>
                <ul className="hidden md:flex gap-6">
                    <li className="p-0"><a href="#home" className="hover:text-primary transition-colors duration-300">About</a></li>
                    <li className="p-0"><a href="#skills" className="hover:text-primary transition-colors duration-300">Skills</a></li>
                    <li className="p-0"><a href="#honors" className="hover:text-primary transition-colors duration-300">Education</a></li>
                    <li className="p-0"><a href="#experience" className="hover:text-primary transition-colors duration-300">Experience</a></li>
                    <li className="p-0"><a href="#certs" className="hover:text-primary transition-colors duration-300">Projects</a></li>
                </ul>
            </div>
        </div>
    )
}