import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

export default function Modal({ project, onClose }) {
    if (!project) {
        return null;
    }

    return (
        // The main overlay
        <div
            className="fixed inset-0 bg-black bg-opacity-70 backdrop-blur-md flex justify-center items-center z-50 p-4"
            onClick={onClose}
        >
            {/* The Modal itself */}
            <div
                className="bg-dark-200 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-y-auto"
                onClick={e => e.stopPropagation()}
            >
                {/* --- Image Section: Full Width on Top --- */}
                <div className="w-full h-64 md:h-96 bg-dark-200 relative flex-shrink-0">
                    <img
                        src={project.img}
                        className="w-full h-full object-cover"
                        alt={project.name}
                    />
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 text-white hover:text-gray-300 text-4xl font-bold leading-none bg-black/50 w-10 h-10 flex items-center justify-center"
                    >
                        &times;
                    </button>
                </div>

                {/* --- Text Content --- */}
                <div className="w-full p-8 flex flex-col">
                    {/* --- Modal Header --- */}
                    <div className="mb-2">
                        <h2 className="text-3xl font-bold text-white">{project.name}</h2>
                    </div>

                    <p className="text-lg text-gray-300 mt-2 border-b border-gray-700 pb-4">{project.issued}</p>

                    {/* --- UPDATED Scrollable Content Area --- */}
                    {/* This section now dynamically renders project details */}
                    <div className="mt-4 text-gray-400 space-y-4">
                        {/* Renders the short description first */}
                        <p>{project.desc}</p>

                        {/* Renders additional paragraphs if they exist */}
                        {project.details && project.details.map((paragraph, index) => (
                            <p key={index}>
                                {paragraph}
                            </p>
                        ))}
                    </div>

                    {/* --- Links Footer --- */}
                    <div className="mt-6 pt-4 border-t border-gray-700 flex-shrink-0 flex justify-between items-center">
                        <div>
                            {project.liveLink && (
                                <a href={project.liveLink} target="_blank" rel="noopener noreferrer" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 transition-colors duration-300">
                                    Live Demo
                                </a>
                            )}
                        </div>
                        <div>
                            {project.repoLink && (
                                <a href={project.repoLink} rel="noreferrer" target="_blank" className="text-gray-400 hover:text-white transition-colors duration-300">
                                    <FontAwesomeIcon size='3x' icon={faGithub} />
                                </a>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}