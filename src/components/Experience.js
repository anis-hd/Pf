import ExpCard from "./ExpCard";
import hr from "../assets/curve-hr.svg";
import img1 from "../assets/talan.png"; // Example image import
import img2 from "../assets/cognoriseInfo.png"; // Example image import

export default function Experience() {
    return (
        <div id="honors" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Professional Experience</h1>
            <p className="font-light text-gray-400"></p>

            <div className="flex flex-col md:flex-row mt-4 gap-5">
                <ExpCard 
                    img={img1} 
                    name="Artificial Intelligence Intern" 
                    issued="Talan Tunisie" 
                    desc="Collaborated on a predictive digital twin for CRISPR-Cas9 applications. Managed reference genomes, generated gRNAs, optimized off-target searches with BLAST, and utilized multi-modal RAG to assess mutated protein impacts." 
                    keywords={["Tensorflow","Multiomics","PyBio", "LLama 3.1", "Multi-modal RAG", "BLAST sequence search","Selenium scraping"]} 

                />
                <ExpCard 
                    img={img2} 
                    name="Machine Learning Intern" 
                    issued="CognoRise Infotech" 
                    desc="Performed EDA, feature engineering, and implemented different ML models for various tasks" 
                    keywords={["Python","Data mining","Sentiment analysis", "Scikit-learn","EDA","Linear regression","Logistic regression","SVMs","Random forests"]} 

                />
            </div>
        </div>
    )
}
