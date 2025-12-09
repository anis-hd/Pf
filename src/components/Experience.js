import { useState } from 'react';
import ExpCard from "./ExpCard";
import imgTalan from "../assets/talan.png";
import imgCogno from "../assets/cognoriseInfo.png";
import imgGrift from "../assets/grift.jpg";

export default function Experience() {
    const [isCollapsed, setIsCollapsed] = useState(false);

    return (
        <div id="experience" className="mt-12 text-white transition-all duration-300">
            <div
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                <h1 className="text-4xl font-bold mb-6 border-b-4 border-primary w-fit pb-2">Professional Experience</h1>
                <div className={`transform transition-transform duration-300 ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400 group-hover:text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>

            <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[2000px] opacity-100'}`}>
                <p className="font-light text-gray-400">My journey in the professional world.</p>

                {/* The layout is now a responsive 2x2 grid */}
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-5">
                    <ExpCard
                        img={imgGrift}
                        name="Computer Vision Intern"
                        issued="GRIFT Group of CRISTAL Laboratory"
                        date="July 2025"
                        desc="Introducing Complex Moments and Fourier-Mellin Transform to deep learning Architectures for geometrically invariant semantic drone image segmentation in low-data regimes."
                        keywords={["UNet", "Image Segmentation", "PyTorch", "Geometric Invariance", "Deep Learning"]}
                    />
                    <ExpCard
                        img={imgTalan}
                        name="End of Studies AI Research Intern"
                        issued="Talan - Tunisie"
                        date="Feb 2025 - Jun 2025"
                        desc="Developed a video compression framework using RAFT for motion prediction, Hyperprior Entropy Coding, and Quantum Fourier Transform (QFT)."
                        keywords={["Video Compression", "RAFT", "Hyperprior Entropy Coding", "QFT", "PyTorch"]}
                    />
                    <ExpCard
                        img={imgTalan}
                        name="Artificial Intelligence Intern"
                        issued="Talan - Tunisie"
                        date="[Date]" // Add date
                        desc="Collaborated on a predictive digital twin for CRISPR-Cas9 applications using multi-modal RAG to assess mutated protein impacts and BLAST for off-target searches."
                        keywords={["Tensorflow", "Multiomics", "PyBio", "Llama 3.1", "Multi-modal RAG", "BLAST"]}
                    />
                    <ExpCard
                        img={imgCogno}
                        name="Machine Learning Intern"
                        issued="CognoRise Infotech"
                        date="[Date]" // Add date
                        desc="Performed EDA, feature engineering, and implemented different ML models for various tasks including sentiment analysis and regression."
                        keywords={["Python", "Data mining", "Scikit-learn", "EDA", "SVMs", "Random forests"]}
                    />
                </div>
            </div>
        </div>
    )
}