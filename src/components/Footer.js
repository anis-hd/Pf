import hr from '../assets/curve-hr.svg'

export default function Footer(){
  return (
    <div className="mt-4 bg-dark-200 rounded-md text-white px-8 py-4">
      <ul className="text-center">
        <li><a href="#home" className="hover:underline">About</a></li>
        <li><a href="#skills" className="hover:underline">Skills</a></li>
        <li><a href="#honors" className="hover:underline">Education</a></li>
        <li><a href="#honors" className="hover:underline">Experience</a></li>

        <li><a href="#certs" className="hover:underline">Projects</a></li>
      </ul>



    </div>
  )
}