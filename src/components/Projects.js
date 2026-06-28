import { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCalendar, faTimes, faGlobe } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

// Import assets
import RadAi from "../assets/certs/RadAi.png";
import portfolio from "../assets/certs/portfolio.jpg";
import arabic from "../assets/certs/arabic.jpg";
import mcode from "../assets/certs/mcode.jpg";
import wordle from "../assets/certs/wordle.webp";
import Ptable from "../assets/certs/Ptable.png";
import crypt from "../assets/certs/crypt.jpg";
import bigdata from "../assets/certs/bigdata.jpg";
import bi from "../assets/certs/bi.webp";
import ophthalmo from "../assets/projects/ophthalmo.jpg";
import farm from "../assets/projects/farm.jpg";
import hr from "../assets/projects/hr.jpg";
import smiles from "../assets/projects/smiles.jpg";
import wifi from "../assets/projects/wifi.jpg";
import map from "../assets/projects/map.jpg";
import sketch from "../assets/projects/sketch.jpg";
import cnn from "../assets/skills/cnn.webp";
import tunibot from "../assets/skills/tunibot.png";
const projectData = [
    {
        name: "TuniBot: Multi-Agent AI that Speaks Tunisian",
        img: tunibot,
        issued: "Talan Tunisie Bootcamp",
        desc: "Engineered a multi-agent framework powered by TuniSpeak (an internal dialectal LLM) featuring specialized agents for Translation, Music Recommendation, and Cultural Insights.",
        date: "feb - 2025",
        repoLink: "",
        category: "AI/ML",
        details: [
            "Engineered a multi-agent framework powered by TuniSpeak (an internal dialectal LLM) featuring specialized agents for Translation, Music Recommendation, and Cultural Insights.",
            "Integrated Vosk ASR for speech-to-text, PyTesseract for vision OCR, and dynamic context-based routing for seamless inter-agent coordination."
        ]
    },
    {
        name: "AI Recruitment Test Generator for EY x IKU Hackathon",
        img: hr,
        issued: "EYxUIK AI Hackathon",
        desc: "Architected and engineered an LLM-powered multi-modal pipeline utilizing Llama 3.1 and RAG to automatically generate comprehensive recruitment tests, winning 2nd place.",
        date: "dec - 2024",
        repoLink: "https://github.com/4nisHd/Recruitment-Test-Generator",
        category: "AI/ML",
        details: [
            "This project was my submission for the EYxUIK AI Hackathon. The core challenge was to create a tool that could help recruiters by automating the creation of technical tests based on a job description.",
            "I implemented a Retrieval-Augmented Generation (RAG) pipeline. The system first retrieves relevant technical concepts from a vector database and then feeds this context to a fine-tuned Llama 3.1 model to generate high-quality, relevant questions and coding challenges."
        ]
    },

    {
        name: "Ophthalmic Eye Disease Diagnosis",
        img: ophthalmo,
        issued: "Talan Tunisie Bootcamp",
        desc: "Collaborated directly with medical professionals to preprocess clinical eye region data and develop computer vision models utilizing VGG16.",
        date: "March 2024",
        repoLink: "https://github.com/4nisHd/Ophthalmic-Disease-Diagnosis-and-Grading",
        category: "AI/ML",
        details: [
            "Collaborated directly with medical professionals to aggregate, clean, and preprocess clinical eye region data.",
            "Developed advanced computer vision models, combining deep learning (VGG16) and geometric models utilizing clinical metrics (MRD1/MRD2) for automated disease diagnosis."
        ]
    },
    {
        name: "SMILES Drug Sequence Prediction",
        img: smiles,
        issued: "Personal Project",
        desc: "Predicted SMILES drug sequences from protein sequences using explainable and generative AI via fine-tuning.",
        date: "February 2024",
        repoLink: "https://github.com/4nisHd/SMILES-sequence-prediction",
        category: "AI/ML",
        details: [
            "Led a computational drug discovery initiative during the Talan PFE Bootcamp to predict SMILES molecular structures directly from protein target sequences.",
            "Fine-tuned and benchmarked multiple LLMs (BioBERT, Mistral, DeepSeek) utilizing LoRA to optimize generation accuracy."
        ]
    },
    {
        name: "Human Movement Prediction via WiFi Channel State Information (CSI)",
        img: wifi,
        issued: "Talan Tunisie Bootcamp",
        desc: "Processed and filtered noisy WiFi CSI amplitude and phase data to extract spatial-temporal features representing human presence and motion.",
        date: "December 2023",
        category: "AI/ML",
        details: [
            "Processed and filtered noisy WiFi CSI (Channel State Information) amplitude and phase data to extract spatial-temporal features representing human presence and motion.",
            "Engineered a hybrid deep learning architecture combining CNNs to extract spatial patterns from subcarriers, and LSTM networks to capture the temporal sequence of movements."
        ]
    },
    {
        name: "AI Guidance Map for Travelers",
        img: map,
        issued: "Projet Technologique Encadré",
        desc: "Integrated the Qwen 2.5 LLM with Dijkstra's algorithm and Google Maps APIs to engineer an intelligent routing system in Tunis.",
        date: "November 2023",
        category: "AI/ML",
        details: [
            "Integrated the Qwen 2.5 LLM with Dijkstra's algorithm and Google Maps APIs to engineer an intelligent, context-aware routing and guidance system for foreign travelers in Tunis."
        ]
    },


    {
        name: "Radiology Platform for AI-assisted Disease Diagnosis",
        img: RadAi,
        issued: "ENSI PCD",
        desc: "A full-stack platform enabling healthcare practitioners to collaborate and automate medical scan diagnoses with secure 2FA and CNN classifiers.",
        date: "Jan-May 2023",
        category: "Full Stack",
        details: [
            "Developed a full-stack platform enabling healthcare practitioners to collaborate and automate medical scan diagnoses.",
            "Built a secure Django backend with a MySQL database, integrating a robust multi-user 2FA system using OTPs.",
            "Trained and deployed a Convolutional Neural Network (CNN) in TensorFlow for accurate image classification."
        ]
    },
    {
        name: "Image Reconstruction from Sketches",
        img: sketch,
        issued: "Personal Project",
        desc: "Used a Pix2Pix architecture for the reconstruction of images from sketches.",
        date: "October 2023",
        repoLink: "https://github.com/4nisHd/Image-reconstruction-from-sketches",
        category: "AI/ML",
        details: [
            "This project focused on the creative application of Generative Adversarial Networks (GANs). The goal was to train a model that could colorize and add texture to a simple line-art sketch, effectively turning it into a photorealistic image.",
            "A Pix2Pix model, which is a type of Conditional GAN, was used. It was trained on a large dataset of paired images (sketches and their corresponding photos) to learn the complex mapping from the sketch domain to the realistic image domain."
        ]
    },
    {
        name: "Lambda Architecture Weather Big Data Analysis",
        img: bigdata,
        issued: "Academic Project",
        desc: "Designed a highly available Lambda architecture to ingest, stream, and analyze real-time OpenWeather API data using Kafka, Spark, HDFS, and Cassandra.",
        date: "November 2024",
        category: "Data Engineering",
        details: [
            "Designed a highly available Lambda architecture to ingest and analyze real-time OpenWeather API data.",
            "Streamed data via Apache Kafka and processed it in real-time using Apache Spark.",
            "Configured a containerized HDFS cluster and Cassandra NoSQL database for robust batch processing and analytics."
        ]
    },
    {
        name: "Data Warehousing of Student Performance Assessments",
        img: bi,
        issued: "Academic Project",
        desc: "Architected an automated ETL pipeline using Apache NiFi to ingest, transform, and load multidimensional data into a MySQL star schema.",
        date: "October 2024",
        category: "Data Engineering",
        details: [
            "Architected an automated ETL pipeline using Apache NiFi to ingest and cleanse multifaceted CSV data.",
            "Transformed and loaded multidimensional data into a MySQL star schema.",
            "Designed interactive Power BI dashboards to track, visualize, and assess student performance metrics."
        ]
    },
    {
        name: "Dynamic Cryptographic Key Rotation with Deep Reinforcement Learning",
        img: crypt,
        issued: "DeepFlow AI Hackathon",
        desc: "Won 1st place among 30 teams at the DeepFlow AI Hackathon by developing a quantum-resilient cyber defense strategy.",
        date: "October 2024",
        category: "AI/ML",
        details: [
            "Won 1st place among 30 teams at the DeepFlow AI Hackathon by developing a quantum-resilient cyber defense strategy.",
            "Evaluated diverse SVM kernels on the KDD dataset to detect network intrusions.",
            "Engineered a custom OpenAI Gym environment and trained a Deep Q-Network (DQN) agent with a Boltzmann policy to dynamically optimize cryptographic key rotation based on threat patterns."
        ]
    },
    {
        name: "MERN Stack Arabic NLP Platform",
        img: arabic,
        issued: "Personal Project",
        desc: "A full-stack platform for various NLP tasks on Arabic text, using React, Node.js, Express, MongoDB, and Flask.",
        date: "Feb 2024 - Ongoing",
        repoLink: "https://github.com/4nisHd/Arabic-NLP-Platform",
        category: "Full Stack",
        details: [
            "This is an ongoing personal project to build a comprehensive web platform for Arabic Natural Language Processing. The frontend is a responsive single-page application built with React.",
            "The backend follows a microservice-oriented architecture. A primary Node.js server with Express handles user authentication and API routing, while a separate Python Flask API serves the computationally intensive NLP models, including fine-tuned LLMs for tasks like summarization and sentiment analysis."
        ]
    },




    {
        name: "Quantum Farm Weather Prediction",
        img: farm,
        issued: "Quantum Challenge",
        desc: "Used QLSTM for long-term prediction and QSVM for short-term prediction and detecting equipment malfunction.",
        date: "June 2024",
        category: "Quantum",
        details: [
            "This project explored the application of quantum machine learning to agricultural technology. The primary goal was to create a robust weather prediction model for optimizing farm operations.",
            "A Quantum Long Short-Term Memory (QLSTM) network was implemented for forecasting long-term weather patterns, while a Quantum Support Vector Machine (QSVM) was used for more immediate, short-term predictions. The QSVM was also effective in detecting anomalies in sensor data, indicating potential equipment malfunctions."
        ]
    },



    {
        name: "Static React Portfolio Deployed on Vercel",
        img: portfolio,
        issued: "Personal Project",
        desc: "Designed and developed a responsive personal portfolio using React and Tailwind CSS to showcase projects and experience.",
        date: "Sept - Dec 2023",
        repoLink: "https://github.com/4nisHd/your-portfolio-repo",
        liveLink: "https://anis-benhd.vercel.app",
        category: "Full Stack",
        details: [
            "Designed and developed a responsive personal portfolio using React and Tailwind CSS to showcase projects and experience.",
            "Deployed on Vercel, leveraging clean responsive architecture."
        ]
    },
    {
        name: "Language Specific Compiler",
        img: mcode,
        issued: "Academic Project",
        desc: "Designed and implemented a complete compiler for a custom programming language using C.",
        date: "Nov - Dec 2023",
        category: "Systems",
        details: [
            "This project was a deep dive into compilation theory. The goal was to build a fully functional compiler for a new, C-like programming language.",
            "The implementation, written entirely in C, covered all the major phases of compilation: Lexical Analysis to convert code into tokens, Syntactical Analysis to build a parse tree, and Semantic Analysis to check for logical and type errors in the source code."
        ]
    },
    {
        name: "Wordle Clone in Android Studio",
        img: wordle,
        issued: "Personal Project",
        desc: "A mobile version of the popular word game Wordle, developed for Android using Java.",
        date: "June - July 2022",
        repoLink: "https://github.com/4nisHd/Wordle-Clone",
        category: "Mobile",
        details: [
            "This project was an exercise in mobile application development and game logic. I recreated the popular game Wordle as a native Android application.",
            "The app was built in Android Studio using Java. Object-Oriented Programming (OOP) principles were heavily utilized to structure the code cleanly, with distinct classes for the game board, keyboard state, and word validation logic."
        ]
    },
    {
        name: "Essence: A chemistry Companion App",
        img: Ptable,
        issued: "Personal Project",
        desc: "A cross-platform mobile app for chemistry students built with Flutter and Dart.",
        date: "March - May 2022",
        category: "Mobile",
        details: [
            "Essence is a mobile app designed to be a helpful tool for chemistry students. It features an interactive periodic table, a molar mass calculator, and other useful utilities.",
            "The application was built using Flutter and the Dart programming language, allowing for a single codebase to be deployed on both Android and iOS. This project was a great introduction to cross-platform development and state management in Flutter."
        ]
    },
    {
        name: "Comparative Analysis of Convolutional vs Recurrent Models in NLP",
        img: cnn,
        issued: "Academic Project",
        desc: "Engineered an NLP pipeline and benchmarked varied CNN and RNN architectures in TensorFlow to evaluate data processing efficiency.",
        date: "October 2023",
        category: "AI/ML",
        details: [
            "Engineered a comprehensive NLP pipeline encompassing text preprocessing, stemming, embedding, and vectorization.",
            "Designed and benchmarked varied CNN and RNN architectures (incorporating LSTM and Attention mechanisms) in TensorFlow to evaluate sequential data data processing efficiency."
        ]
    },
    {
        name: "Multi-Agent Energy Distribution System",
        img: mcode,
        issued: "Academic Project",
        desc: "Programmed a distributed multi-agent system using JADE to optimize power grid energy distribution and load balancing.",
        date: "May 2023",
        category: "Systems",
        details: [
            "Programmed a distributed multi-agent system using JADE to optimize power grid energy distribution and load balancing.",
            "Implemented Source, Consumer, and Utility agents communicating via standard ACL for real-time state synchronization."
        ]
    }
];

const categories = ["All", "AI/ML", "Data Engineering", "Full Stack", "Systems", "Mobile", "Quantum"];

export default function Projects() {
    const [selectedProject, setSelectedProject] = useState(null);
    const [activeCategory, setActiveCategory] = useState("All");

    const handleCardClick = (project) => {
        setSelectedProject(project);
    };

    const handleCloseModal = () => {
        setSelectedProject(null);
    };

    const filteredProjects = activeCategory === "All"
        ? projectData
        : projectData.filter(p => p.category === activeCategory);

    return (
        <section id="certs" className="py-20 text-slate-900 relative">
            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-slate-900">
                            Projects
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-blue-600 rounded-full" />
                    </div>
                </div>

                {/* Content */}
                <div className="opacity-100 transition-all duration-500 ease-in-out">
                    <p className="text-slate-500 text-lg mb-6">
                        A showcase of my work across different domains
                    </p>

                    <div className="flex flex-wrap gap-3 mb-10">
                        {categories.map((cat, idx) => (
                            <button
                                key={idx}
                                onClick={() => setActiveCategory(cat)}
                                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${activeCategory === cat
                                    ? 'bg-blue-600 text-white shadow-sm'
                                    : 'bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-200 hover:border-slate-300'
                                    }`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>

                    {/* Projects Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {filteredProjects.map((project, index) => (
                            <div
                                key={index}
                                className="group relative cursor-pointer"
                                onClick={() => handleCardClick(project)}
                            >
                                {/* Card */}
                                <div className="relative rounded-2xl bg-white border border-slate-200 overflow-hidden shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 h-full flex flex-col">
                                    {/* Image */}
                                    <div className="relative h-48 overflow-hidden">
                                        <img
                                            src={project.img}
                                            alt={project.name}
                                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />

                                        {/* Category Badge */}
                                        <div className="absolute top-4 left-4">
                                            <span className="px-3 py-1 rounded-full bg-slate-900/80 backdrop-blur-sm text-xs font-medium text-white border border-slate-800">
                                                {project.category}
                                            </span>
                                        </div>

                                        {/* Date */}
                                        <div className="absolute bottom-4 left-4 flex items-center gap-2 text-sm text-slate-100">
                                            <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                            {project.date}
                                        </div>
                                    </div>

                                    {/* Content */}
                                    <div className="p-5 flex-1 flex flex-col">
                                        <h3 className="text-lg font-bold mb-2 text-slate-900 group-hover:text-blue-600 transition-colors line-clamp-1">
                                            {project.name}
                                        </h3>
                                        <p className="text-slate-500 text-sm mb-2">
                                            {project.issued}
                                        </p>
                                        <p className="text-slate-600 text-sm line-clamp-2 flex-1">
                                            {project.desc}
                                        </p>

                                        {/* Links */}
                                        <div className="flex items-center gap-3 mt-4 pt-4 border-t border-slate-100">
                                            {project.repoLink && (
                                                <a
                                                    href={project.repoLink}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="text-slate-400 hover:text-blue-600 transition-colors"
                                                >
                                                    <FontAwesomeIcon icon={faGithub} />
                                                </a>
                                            )}
                                            {project.liveLink && (
                                                <a
                                                    href={project.liveLink}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="text-slate-400 hover:text-blue-600 transition-colors"
                                                >
                                                    <FontAwesomeIcon icon={faGlobe} />
                                                </a>
                                            )}
                                            <span className="ml-auto text-xs text-slate-400 group-hover:text-blue-600 transition-colors">
                                                Click for details →
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Modal */}
            {selectedProject && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
                    onClick={handleCloseModal}
                >
                    <div
                        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl bg-white border border-slate-200 shadow-xl animate-in fade-in zoom-in-95 duration-200"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Close Button */}
                        <button
                            onClick={handleCloseModal}
                            className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-slate-100/90 backdrop-blur-sm flex items-center justify-center text-slate-600 hover:text-slate-900 transition-colors border border-slate-200"
                        >
                            <FontAwesomeIcon icon={faTimes} />
                        </button>

                        {/* Modal Image */}
                        <div className="relative h-64">
                            <img
                                src={selectedProject.img}
                                alt={selectedProject.name}
                                className="w-full h-full object-cover"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-white via-transparent to-transparent" />
                        </div>

                        {/* Modal Content */}
                        <div className="p-6 -mt-16 relative z-10">
                            {/* Category & Date */}
                            <div className="flex items-center gap-3 mb-4">
                                <span className="px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-800 text-xs font-medium">
                                    {selectedProject.category}
                                </span>
                                <span className="text-slate-500 text-sm flex items-center gap-2">
                                    <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                    {selectedProject.date}
                                </span>
                            </div>

                            {/* Title */}
                            <h2 className="text-2xl md:text-3xl font-bold mb-2 text-slate-900">
                                {selectedProject.name}
                            </h2>

                            {/* Issued */}
                            <p className="text-slate-500 mb-6">
                                {selectedProject.issued}
                            </p>

                            {/* Description */}
                            <div className="space-y-4 mb-6">
                                {selectedProject.details.map((paragraph, idx) => (
                                    <p key={idx} className="text-slate-700 leading-relaxed">
                                        {paragraph}
                                    </p>
                                ))}
                            </div>

                            {/* Links */}
                            <div className="flex flex-wrap gap-4">
                                {selectedProject.repoLink && (
                                    <a
                                        href={selectedProject.repoLink}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inline-flex items-center gap-2 px-6 py-3 bg-slate-100 border border-slate-200 rounded-lg font-medium text-slate-700 hover:bg-slate-200 transition-all duration-300"
                                    >
                                        <FontAwesomeIcon icon={faGithub} />
                                        View Code
                                    </a>
                                )}
                                {selectedProject.liveLink && (
                                    <a
                                        href={selectedProject.liveLink}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 shadow-sm hover:shadow transition-all duration-300"
                                    >
                                        <FontAwesomeIcon icon={faGlobe} />
                                        Live Demo
                                    </a>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}