import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCalendar, faBuilding } from '@fortawesome/free-solid-svg-icons';
import imgTalan from "../assets/talan.png";
import imgCogno from "../assets/cognoriseInfo.png";
import imgGrift from "../assets/grift.jpg";
import imgEnsi from "../assets/skills/ensi.jpg";
import cntxtai_logo from "../assets/skills/cntxtai_logo.jpg";

export default function Experience() {
    const experienceData = [
        {
            img: cntxtai_logo,
            title: "Data Specialist - Freelance",
            company: "CNTXT.AI",
            date: "Jan 2026 - April 2026",
            desc: "Curating Tunisian dialect datasets and developing AI powered apps.",
            keywords: ["Generative AI", "Data cleaning", "Data Annotation"]
        },
        {
            img: imgGrift,
            title: "ML Engineer - Freelance",
            company: "GRIFT Group of CRISTAL Laboratory",
            date: "July 2025",
            desc: "Introducing Complex Moments and Fourier-Mellin Transform to deep learning Architectures for geometrically invariant semantic drone image segmentation in low-data regimes.",
            keywords: ["UNet", "Image Segmentation", "PyTorch", "Geometric Invariance", "Deep Learning"]
        },
        {
            img: imgTalan,
            title: "End of Studies AI Research Intern",
            company: "Talan - Tunisie",
            date: "Feb 2025 - Jun 2025",
            desc: "Developed a video compression framework using RAFT for motion prediction, Hyperprior Entropy Coding, and Quantum Fourier Transform (QFT).",
            keywords: ["Video Compression", "RAFT", "Hyperprior Entropy Coding", "QFT", "PyTorch"]
        },
        {
            img: imgTalan,
            title: "Artificial Intelligence Intern",
            company: "Talan - Tunisie",
            date: "Summer 2024",
            desc: "Collaborated on a predictive digital twin for CRISPR-Cas9 applications using multi-modal RAG to assess mutated protein impacts and BLAST for off-target searches.",
            keywords: ["Tensorflow", "Multiomics", "PyBio", "Llama 3.1", "Multi-modal RAG", "BLAST"]
        },
        {
            img: imgCogno,
            title: "Machine Learning Intern",
            company: "CognoRise Infotech",
            date: "Summer 2023",
            desc: "Performed EDA, feature engineering, and implemented different ML models for various tasks including sentiment analysis and regression.",
            keywords: ["Python", "Data mining", "Scikit-learn", "EDA", "SVMs", "Random forests"]
        },
        {
            img: imgEnsi,
            title: "Software development Intern",
            company: "ENSI",
            date: "Summer 2023",
            desc: "Developed serious games in Android Studio & Java",
            keywords: ["Java", "Android Studio", "Game Development"]
        }
    ];

    return (
        <section id="experience" className="py-20 text-slate-900 relative">
            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-slate-900">
                            Experience
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-blue-600 rounded-full" />
                    </div>
                </div>

                {/* Content */}
                <div className="opacity-100 transition-all duration-500 ease-in-out">
                    <p className="text-slate-500 text-lg mb-10">
                        My professional journey in tech
                    </p>

                    {/* Experience Cards Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {experienceData.map((exp, index) => (
                            <div
                                key={index}
                                className="group relative"
                            >
                                {/* Card */}
                                <div className="relative p-6 rounded-2xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 h-full">
                                    <div className="flex items-start gap-4">
                                        {/* Company Logo */}
                                        <div className="flex-shrink-0 w-16 h-16 rounded-xl bg-slate-50 border border-slate-100 p-2 overflow-hidden">
                                            <img
                                                src={exp.img}
                                                alt={exp.company}
                                                className="w-full h-full object-contain"
                                            />
                                        </div>

                                        {/* Content */}
                                        <div className="flex-1 min-w-0">
                                            {/* Date Badge */}
                                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-medium mb-2">
                                                <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                                {exp.date}
                                            </div>

                                            {/* Title */}
                                            <h3 className="text-xl font-bold mb-1 text-slate-900 group-hover:text-blue-600 transition-colors">
                                                {exp.title}
                                            </h3>

                                            {/* Company */}
                                            <p className="text-slate-500 text-sm flex items-center gap-2 mb-3">
                                                <FontAwesomeIcon icon={faBuilding} className="text-xs" />
                                                {exp.company}
                                            </p>

                                            {/* Description */}
                                            <p className="text-slate-600 text-sm leading-relaxed mb-4">
                                                {exp.desc}
                                            </p>

                                            {/* Keywords */}
                                            <div className="flex flex-wrap gap-2">
                                                {exp.keywords.map((keyword, idx) => (
                                                    <span
                                                        key={idx}
                                                        className="px-2 py-1 text-xs rounded-md bg-slate-50 border border-slate-200 text-slate-600 hover:border-blue-500/50 hover:bg-slate-100 transition-colors"
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