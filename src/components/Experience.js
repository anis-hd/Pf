import ExpCard from "./ExpCard";
import imgTalan from "../assets/talan.png";
import imgCogno from "../assets/cognoriseInfo.png";
import imgGrift from "../assets/grift.jpg";

export default function Experience() {
    return (
        <div id="experience" className="mt-12 text-white">
            <h1 className="text-2xl font-bold">Professional Experience</h1>
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
                    name="Research Intern"
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
    )
}