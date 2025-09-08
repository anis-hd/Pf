import ProjectCard from "./ProjectCards.js"

import hr from "../assets/curve-hr.svg"
import RadAi from "../assets/certs/RadAi.png"
import portfolio from "../assets/certs/portfolio.jpg"
import arabic from "../assets/certs/arabic.jpg"
import mcode from "../assets/certs/mcode.jpg"
import wordle from "../assets/certs/wordle.webp"
import Ptable from "../assets/certs/Ptable.png"
import crypt from "../assets/certs/crypt.jpg"
import bigdata from "../assets/certs/bigdata.jpg"
import bi from "../assets/certs/bi.webp"
// import 

export default function Projects(){
    return (
        <div id="certs" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Projects</h1>
            <p className="font-light text-gray-400">Here are some of my projects</p>

            <div className="grid grid-cols-1 md:grid-cols-3 justify-center mt-4 gap-5">
                {/* --- New Projects Added Here --- */}
                <ProjectCard name="AI Recruitment Test Generator" img={RadAi} issued="Developed an LLM-powered solution using Llama 3.1 and RAG to automatically generate recruitment tests for the EYxUIK AI Hackathon." date="[Date]" />
                <ProjectCard name="Quantum Farm Weather Prediction" img={RadAi} issued="Used QLSTM for long-term prediction and QSVM for short-term prediction and detecting equipment malfunction." date="[Date]" />
                <ProjectCard name="Ophthalmic Disease Diagnosis" img={RadAi} issued="Manual data collection, cleaning, and implementation of different computer vision models for classification, alongside rule-based methods for diagnosis." date="[Date]" />
                <ProjectCard name="SMILES Drug Sequence Prediction" img={RadAi} issued="Predicted SMILES drug sequences from protein sequences using explainable and generative AI via fine-tuning." date="[Date]" />
                <ProjectCard name="Movement Detection from WiFi Signals" img={RadAi} issued="Detected movement from the fluctuations of WiFi signals using RNNs and CNNs." date="[Date]" />
                <ProjectCard name="AI Guidance Map for Travellers" img={RadAi} issued="Used LLMs and Dijkstra's algorithm for a guidance map system for foreign travellers in Tunis." date="[Date]" />
                <ProjectCard name="Image Reconstruction from Sketches" img={RadAi} issued="Used a Pix2Pix architecture for the reconstruction of images from sketches." date="[Date]" />

                {/* --- Existing Projects --- */}
                <ProjectCard name="Big data user behavior analysis" img={bigdata} issued="Set up a distributed, multi-node HDFS environment in Docker. Used MapReduce to filter raw data, followed by data cleaning with Pig and Spark to create structured datasets. Leveraged Hive for complex queries, extracting insights on buying patterns." date="November 2024" />
                <ProjectCard name="Business Intelligence data pipeline" img={bi} issued="Used Apache NiFi with Power BI to automate the ingestion, processing, and visualization of data from a CSV file, enabling real-time insights through dashboards." date="October 2024" />
                <ProjectCard name="Cryptographic key rotation optimization" img={crypt} issued="This was a proof of concept for DeepFlow AI Hackathon, winning first place from the technical jury. Keywords: Deep Q-Networks, Tensorflow, Gym" date="October 2024" />
                <ProjectCard name="MERN Stack Arabic NLP Platform" img={arabic} issued="Personal Project, Technologies used: React, Express, Node, MongoDB, Flask APIs, LLMs, Generative AI, NLP" date="Feb 2024 - Ongoing" />
                <ProjectCard name="AI Radiology platform" img={RadAi} issued="ENSI PCD, Technologies used: Django, SQL, Image Processing, CNNs" date="Jan-May 2023" />
                <ProjectCard name="Personal Portfolio" img={portfolio} issued="React, TailwindCSS, Firebase" date="Sept - Dec 2023" />
                <ProjectCard name="Language Specific Compiler" img={mcode} issued="Compilation Theory, Lexical Analysis, Syntaxical Analysis, Semantic Analysis, C" date="Nov - Dec 2023" />
                <ProjectCard name="Wordle Clone in Android Studio" img={wordle} issued="Android Studio, Java, OOP" date="June - July 2022" />
                <ProjectCard name="Essence: A chemistry Companion App" img={Ptable} issued="Flutter, Dart, OOP" date="March - May 2022" />
            </div>
        </div>
    )
}