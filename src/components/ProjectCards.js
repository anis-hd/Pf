
// src/components/ProjectCard.js
export default function ProjectCard({ project, onClick }) {
    return (
        <div
            data-aos="fade-up"
            data-aos-duration="500"
            data-aos-offset="100"
            className="hover:bg-dark-300 w-full h-full bg-dark-200 py-4 px-4 cursor-pointer transform hover:-translate-y-2 transition-transform duration-300"
            onClick={() => onClick(project)}
        >
            <img src={project.img} className="w-full h-56 mx-auto object-cover" alt={project.name}></img>
            <div className="mt-2">
                <h1 className="font-bold md:text-xl">{project.name}</h1>
                <p className="font-light md:text-lg">{project.issued}</p>
                <p className="font-light text-gray-400 truncate">{project.desc}</p>
                <p className="font-light text-gray-400 mt-2">{project.date}</p>
            </div>
        </div>
    );
}