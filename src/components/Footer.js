import hr from '../assets/curve-hr.svg'

export default function Footer() {
  return (
    <div className="mt-12 bg-dark-200 text-white px-8 py-8">
      <ul className="flex flex-wrap justify-center gap-6 md:gap-10 text-center font-medium text-gray-300">
        <li><a href="#home" className="hover:text-primary transition-colors duration-300">About</a></li>
        <li><a href="#skills" className="hover:text-primary transition-colors duration-300">Skills</a></li>
        <li><a href="#honors" className="hover:text-primary transition-colors duration-300">Education</a></li>
        <li><a href="#experience" className="hover:text-primary transition-colors duration-300">Experience</a></li>
        <li><a href="#certs" className="hover:text-primary transition-colors duration-300">Projects</a></li>
      </ul>

      <p className="text-center text-gray-500 mt-6 text-sm">
        &copy; 2025 All rights reserved.
      </p>
    </div>
  )
}