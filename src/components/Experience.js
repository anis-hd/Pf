import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown, faBriefcase, faCalendar, faBuilding } from '@fortawesome/free-solid-svg-icons';
import imgTalan from "../assets/talan.png";
import imgCogno from "../assets/cognoriseInfo.png";
import imgGrift from "../assets/grift.jpg";

export default function Experience() {
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

    const experienceData = [
        {
            img: imgGrift,
            title: "Computer Vision Intern",
            company: "GRIFT Group of CRISTAL Laboratory",
            date: "July 2025",
            desc: "Introducing Complex Moments and Fourier-Mellin Transform to deep learning Architectures for geometrically invariant semantic drone image segmentation in low-data regimes.",
            keywords: ["UNet", "Image Segmentation", "PyTorch", "Geometric Invariance", "Deep Learning"],
            gradient: "from-green-500 to-emerald-500"
        },
        {
            img: imgTalan,
            title: "End of Studies AI Research Intern",
            company: "Talan - Tunisie",
            date: "Feb 2025 - Jun 2025",
            desc: "Developed a video compression framework using RAFT for motion prediction, Hyperprior Entropy Coding, and Quantum Fourier Transform (QFT).",
            keywords: ["Video Compression", "RAFT", "Hyperprior Entropy Coding", "QFT", "PyTorch"],
            gradient: "from-purple-500 to-pink-500"
        },
        {
            img: imgTalan,
            title: "Artificial Intelligence Intern",
            company: "Talan - Tunisie",
            date: "Summer 2024",
            desc: "Collaborated on a predictive digital twin for CRISPR-Cas9 applications using multi-modal RAG to assess mutated protein impacts and BLAST for off-target searches.",
            keywords: ["Tensorflow", "Multiomics", "PyBio", "Llama 3.1", "Multi-modal RAG", "BLAST"],
            gradient: "from-pink-500 to-orange-500"
        },
        {
            img: imgCogno,
            title: "Machine Learning Intern",
            company: "CognoRise Infotech",
            date: "Summer 2023",
            desc: "Performed EDA, feature engineering, and implemented different ML models for various tasks including sentiment analysis and regression.",
            keywords: ["Python", "Data mining", "Scikit-learn", "EDA", "SVMs", "Random forests"],
            gradient: "from-blue-500 to-cyan-500"
        }
    ];

    return (
        <section id="experience" className="py-20 text-white relative">
            {/* Background Gradient Orb */}
            <div
                className="absolute inset-0 pointer-events-none overflow-hidden"
                style={{
                    background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, rgba(99, 102, 241, 0.06) 0%, transparent 50%)`
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
                            Experience
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
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[4000px] opacity-100'}`}>
                    <p className="text-gray-400 text-lg mb-10" data-aos="fade-up" data-aos-delay="100">
                        My professional journey in tech
                    </p>

                    {/* Experience Cards Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {experienceData.map((exp, index) => (
                            <div
                                key={index}
                                className="group relative"
                                data-aos="fade-up"
                                data-aos-delay={index * 100}
                            >
                                {/* Glow Effect */}
                                <div className={`absolute -inset-0.5 bg-gradient-to-r ${exp.gradient} rounded-2xl blur opacity-0 group-hover:opacity-30 transition-opacity duration-500`} />

                                {/* Card */}
                                <div className="relative p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 transition-all duration-300 h-full">
                                    <div className="flex items-start gap-4">
                                        {/* Company Logo */}
                                        <div className="flex-shrink-0 w-16 h-16 rounded-xl bg-white/10 p-2 overflow-hidden">
                                            <img
                                                src={exp.img}
                                                alt={exp.company}
                                                className="w-full h-full object-contain"
                                            />
                                        </div>

                                        {/* Content */}
                                        <div className="flex-1 min-w-0">
                                            {/* Date Badge */}
                                            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r ${exp.gradient} bg-opacity-20 text-xs font-medium mb-2`}>
                                                <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                                {exp.date}
                                            </div>

                                            {/* Title */}
                                            <h3 className="text-xl font-bold mb-1 group-hover:text-purple-400 transition-colors">
                                                {exp.title}
                                            </h3>

                                            {/* Company */}
                                            <p className="text-gray-400 text-sm flex items-center gap-2 mb-3">
                                                <FontAwesomeIcon icon={faBuilding} className="text-xs" />
                                                {exp.company}
                                            </p>

                                            {/* Description */}
                                            <p className="text-gray-400 text-sm leading-relaxed mb-4">
                                                {exp.desc}
                                            </p>

                                            {/* Keywords */}
                                            <div className="flex flex-wrap gap-2">
                                                {exp.keywords.map((keyword, idx) => (
                                                    <span
                                                        key={idx}
                                                        className="px-2 py-1 text-xs rounded-md bg-white/5 border border-white/10 text-gray-300 hover:border-purple-500/50 transition-colors"
                                                    >
                                                        {keyword}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}