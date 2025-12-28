import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown, faGraduationCap, faCalendar } from '@fortawesome/free-solid-svg-icons';

export default function Education() {
    const [isCollapsed, setIsCollapsed] = useState(false);
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

    const educationData = [
        {
            school: "National School of Computer Science",
            degree: "M.Eng Computer Science, AI & Decision Systems",
            description: "Computer science, software engineering and applied mathematics.",
            year: "2021 - 2025",
            gradient: "from-purple-500 to-pink-500"
        },
        {
            school: "Preparatory Institute for Engineering Studies of Nabeul",
            degree: "Physics and Chemistry",
            description: "Two years of intensive studies in Mathematics, Physics and Industrial Sciences for the national engineering contest.",
            year: "2019 - 2021",
            gradient: "from-blue-500 to-cyan-500"
        }
    ];

    return (
        <section id="honors" className="py-20 text-white relative">
            {/* Background Gradient Orb */}
            <div
                className="absolute inset-0 pointer-events-none overflow-hidden"
                style={{
                    background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, rgba(217, 70, 239, 0.06) 0%, transparent 50%)`
                }}
            />

            <div className="relative z-10">
                {/* Section Header */}
                <div
                    className="flex items-center justify-between cursor-pointer group mb-8"
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    data-aos="fade-up"
                >
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-white">
                            Education
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 rounded-full" />
                    </div>
                    <div className={`w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center group-hover:bg-white/10 group-hover:border-purple-500/50 transition-all duration-300 ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}>
                        <FontAwesomeIcon
                            icon={faChevronDown}
                            className="text-gray-400 group-hover:text-purple-400 transition-colors"
                        />
                    </div>
                </div>

                {/* Collapsible Content */}
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[2000px] opacity-100'}`}>
                    <p className="text-gray-400 text-lg mb-10" data-aos="fade-up" data-aos-delay="100">
                        My academic journey
                    </p>

                    {/* Timeline */}
                    <div className="relative">
                        {/* Timeline Line */}
                        <div className="absolute left-8 md:left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-purple-500 via-pink-500 to-blue-500 hidden md:block" />

                        {/* Education Cards */}
                        <div className="space-y-8">
                            {educationData.map((edu, index) => (
                                <div
                                    key={index}
                                    className={`relative flex flex-col md:flex-row gap-8 ${index % 2 === 0 ? 'md:flex-row-reverse' : ''}`}
                                    data-aos={index % 2 === 0 ? "fade-left" : "fade-right"}
                                    data-aos-delay={index * 100}
                                >
                                    {/* Timeline Dot */}
                                    <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 z-10" />

                                    {/* Content */}
                                    <div className={`flex-1 ${index % 2 === 0 ? 'md:text-right md:pr-12' : 'md:pl-12'}`}>
                                        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 group">
                                            {/* Year Badge */}
                                            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r ${edu.gradient} bg-opacity-20 text-sm font-medium mb-4`}>
                                                <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                                {edu.year}
                                            </div>

                                            {/* School Name */}
                                            <h3 className="text-xl md:text-2xl font-bold mb-2 group-hover:text-purple-400 transition-colors">
                                                {edu.school}
                                            </h3>

                                            {/* Degree */}
                                            <p className={`text-lg font-medium bg-gradient-to-r ${edu.gradient} bg-clip-text text-transparent mb-3`}>
                                                {edu.degree}
                                            </p>

                                            {/* Description */}
                                            <p className="text-gray-400 leading-relaxed">
                                                {edu.description}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Spacer for timeline alignment */}
                                    <div className="hidden md:block flex-1" />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}