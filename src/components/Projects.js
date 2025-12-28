import { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown, faExternalLink, faCode, faCalendar, faTimes, faGlobe } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

// Import your assets
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

const projectData = [
    {
        name: "AI Recruitment Test Generator",
        img: hr,
        issued: "EYxUIK AI Hackathon",
        desc: "Developed an LLM-powered solution using Llama 3.1 and RAG to automatically generate recruitment tests for the EYxUIK AI Hackathon.",
        date: "July 2024",
        repoLink: "https://github.com/4nisHd/Recruitment-Test-Generator",
        category: "AI/ML",
        details: [
            "This project was my submission for the EYxUIK AI Hackathon. The core challenge was to create a tool that could help recruiters by automating the creation of technical tests based on a job description.",
            "I implemented a Retrieval-Augmented Generation (RAG) pipeline. The system first retrieves relevant technical concepts from a vector database and then feeds this context to a fine-tuned Llama 3.1 model to generate high-quality, relevant questions and coding challenges."
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
        name: "Ophthalmic Disease Diagnosis",
        img: ophthalmo,
        issued: "Talan Tunisie Bootcamp",
        desc: "Manual data collection, cleaning, and implementation of different computer vision models for classification, alongside rule-based methods for diagnosis.",
        date: "March 2024",
        repoLink: "https://github.com/4nisHd/Ophthalmic-Disease-Diagnosis-and-Grading",
        category: "AI/ML",
        details: [
            "As part of a bootcamp with Talan Tunisie, this project focused on diagnosing eye diseases from retinal scans. A significant part of the project involved the manual collection and meticulous cleaning of a diverse dataset of ophthalmic images.",
            "Several computer vision models, including ResNet and VGG, were trained and evaluated for classification accuracy. To enhance diagnostic reliability, a rule-based system was integrated to cross-reference model outputs with established medical criteria, providing a more robust final diagnosis."
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
            "This project delves into computational drug discovery by predicting SMILES strings (a textual representation of a molecule's structure) from protein target sequences. This can significantly accelerate the initial stages of drug development.",
            "A transformer-based sequence-to-sequence model was fine-tuned for this task. A key focus was on explainable AI (XAI) techniques to understand which parts of the protein sequence were most influential in determining the final drug structure, adding a layer of interpretability to the generative process."
        ]
    },
    {
        name: "Movement Detection from WiFi Signals",
        img: wifi,
        issued: "Talan Tunisie Bootcamp",
        desc: "Detected movement from the fluctuations of WiFi signals using RNNs and CNNs.",
        date: "December 2023",
        category: "AI/ML",
        details: [
            "This project explored a device-free approach to motion detection by analyzing Channel State Information (CSI) from standard WiFi signals. The presence of a person in a room subtly alters these signals, creating patterns that can be learned.",
            "Recurrent Neural Networks (RNNs) were employed to capture the temporal dependencies in the WiFi signal fluctuations over time, while Convolutional Neural Networks (CNNs) were used to extract spatial features from the signal data, creating a powerful hybrid model for accurate movement detection."
        ]
    },
    {
        name: "AI Guidance Map for Travellers",
        img: map,
        issued: "Projet Technologique Encadré",
        desc: "Used LLMs and Dijkstra's algorithm for a guidance map system for foreign travellers in Tunis.",
        date: "November 2023",
        category: "AI/ML",
        details: [
            "The goal of this project was to create an intelligent map for tourists that combines optimal pathfinding with rich, contextual information. The system was designed specifically for travellers navigating the city of Tunis.",
            "Dijkstra's algorithm formed the core of the navigation system, ensuring the shortest path between two points. This was enhanced by a Large Language Model (LLM) that interpreted natural language queries and provided detailed, interesting descriptions of landmarks and points of interest along the calculated route."
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
        name: "Real-Time Weather Data Pipeline",
        img: bigdata,
        issued: "Academic Project",
        desc: "Engineered a multi-node big data pipeline using HDFS, Spark, and Cassandra to process and store real-time weather data.",
        date: "November 2024",
        category: "Data Engineering",
        details: [
            "This project involved architecting a complete big data ecosystem. The foundation was a multi-node cluster running the Hadoop Distributed File System (HDFS), which provided a scalable and fault-tolerant storage layer for the raw, incoming weather data from the OpenWeather API.",
            "Apache Spark's Structured Streaming was used as the real-time processing engine. It ingested the data, performed transformations and aggregations in-memory, and then implemented a dual-storage strategy: the processed, query-ready results were written to Apache Cassandra for low-latency access, while the raw historical data was persisted in HDFS for long-term storage and potential batch analysis."
        ]
    },
    {
        name: "Business Intelligence data pipeline",
        img: bi,
        issued: "Academic Project",
        desc: "Used Apache NiFi with Power BI to automate the ingestion, processing, and visualization of data from a CSV file.",
        date: "October 2024",
        category: "Data Engineering",
        details: [
            "The objective was to build an end-to-end automated data pipeline for business intelligence. The system was designed to pull data from a source, process it, and present it in an interactive dashboard.",
            "Apache NiFi was used to manage the data flow, automatically ingesting raw data from a CSV file, performing necessary cleaning and transformation steps. The processed data was then fed directly into Microsoft Power BI, enabling the creation of dynamic, real-time dashboards for insightful data visualization."
        ]
    },
    {
        name: "Cryptographic key rotation optimization",
        img: crypt,
        issued: "DeepFlow AI Hackathon",
        desc: "A proof of concept for DeepFlow AI Hackathon, winning first place from the technical jury.",
        date: "October 2024",
        category: "AI/ML",
        details: [
            "This project, which won first place from the technical jury, aimed to find the optimal policy for rotating cryptographic keys to balance security with computational cost. The problem was framed as a reinforcement learning challenge.",
            "A Deep Q-Network (DQN) agent was built using Tensorflow. The agent was trained in a custom environment, created with the OpenAI Gym toolkit, that simulated security threats and system overhead. The trained agent learned a dynamic rotation policy that proved more effective than traditional, static schedules."
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
        name: "AI Radiology platform",
        img: RadAi,
        issued: "ENSI PCD",
        desc: "A Django-based web platform for analyzing medical images using CNNs for diagnostic assistance.",
        date: "Jan-May 2023",
        category: "Full Stack",
        details: [
            "This academic project involved creating a web-based platform to assist radiologists. The system allows users to upload medical scans and receive an AI-powered analysis.",
            "The platform was built using the Django framework, with a SQL database to manage patient and image data. The core diagnostic feature was powered by Convolutional Neural Networks (CNNs) trained to classify and identify anomalies in radiological images."
        ]
    },
    {
        name: "Personal Portfolio",
        img: portfolio,
        issued: "Personal Project",
        desc: "My personal portfolio website to showcase my skills and projects, built with modern web technologies.",
        date: "Sept - Dec 2023",
        repoLink: "https://github.com/4nisHd/your-portfolio-repo",
        liveLink: "#",
        category: "Full Stack",
        details: [
            "This website was built from scratch using React and TailwindCSS, demonstrating my ability to create responsive and aesthetically pleasing user interfaces.",
            "It is deployed on Firebase, leveraging its hosting capabilities for fast content delivery. The project cards and modal system you are interacting with now are key features of this implementation."
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
    }
];

const categories = ["All", "AI/ML", "Data Engineering", "Full Stack", "Systems", "Mobile", "Quantum"];

export default function Projects() {
    const [selectedProject, setSelectedProject] = useState(null);
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [activeCategory, setActiveCategory] = useState("All");
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
        <section id="certs" className="py-20 text-white relative">
            {/* Background Gradient Orb */}
            <div
                className="absolute inset-0 pointer-events-none overflow-hidden"
                style={{
                    background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, rgba(14, 165, 233, 0.06) 0%, transparent 50%)`
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
                        <h2
                            className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent transition-all duration-75 ease-out"
                            style={{
                                backgroundImage: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, #6366f1, #d946ef, #0ea5e9)`
                            }}
                        >
                            Projects
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
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[8000px] opacity-100'}`}>
                    <p className="text-gray-400 text-lg mb-6" data-aos="fade-up" data-aos-delay="100">
                        A showcase of my work across different domains
                    </p>

                    {/* Category Filter */}
                    <div className="flex flex-wrap gap-3 mb-10" data-aos="fade-up" data-aos-delay="150">
                        {categories.map((cat, idx) => (
                            <button
                                key={idx}
                                onClick={() => setActiveCategory(cat)}
                                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${activeCategory === cat
                                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                                        : 'bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:border-purple-500/50'
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
                                data-aos="fade-up"
                                data-aos-delay={index * 50}
                            >
                                {/* Glow Effect */}
                                <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-2xl blur opacity-0 group-hover:opacity-30 transition-opacity duration-500" />

                                {/* Card */}
                                <div className="relative rounded-2xl bg-white/5 border border-white/10 overflow-hidden hover:border-white/20 transition-all duration-300 h-full flex flex-col">
                                    {/* Image */}
                                    <div className="relative h-48 overflow-hidden">
                                        <img
                                            src={project.img}
                                            alt={project.name}
                                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />

                                        {/* Category Badge */}
                                        <div className="absolute top-4 left-4">
                                            <span className="px-3 py-1 rounded-full bg-black/50 backdrop-blur-sm text-xs font-medium border border-white/20">
                                                {project.category}
                                            </span>
                                        </div>

                                        {/* Date */}
                                        <div className="absolute bottom-4 left-4 flex items-center gap-2 text-sm text-gray-300">
                                            <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                            {project.date}
                                        </div>
                                    </div>

                                    {/* Content */}
                                    <div className="p-5 flex-1 flex flex-col">
                                        <h3 className="text-lg font-bold mb-2 group-hover:text-purple-400 transition-colors line-clamp-1">
                                            {project.name}
                                        </h3>
                                        <p className="text-gray-500 text-sm mb-2">
                                            {project.issued}
                                        </p>
                                        <p className="text-gray-400 text-sm line-clamp-2 flex-1">
                                            {project.desc}
                                        </p>

                                        {/* Links */}
                                        <div className="flex items-center gap-3 mt-4 pt-4 border-t border-white/10">
                                            {project.repoLink && (
                                                <a
                                                    href={project.repoLink}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="text-gray-400 hover:text-purple-400 transition-colors"
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
                                                    className="text-gray-400 hover:text-purple-400 transition-colors"
                                                >
                                                    <FontAwesomeIcon icon={faGlobe} />
                                                </a>
                                            )}
                                            <span className="ml-auto text-xs text-gray-500 group-hover:text-purple-400 transition-colors">
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
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
                    onClick={handleCloseModal}
                >
                    <div
                        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl bg-dark-500 border border-white/20"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Close Button */}
                        <button
                            onClick={handleCloseModal}
                            className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-gray-400 hover:text-white transition-colors"
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
                            <div className="absolute inset-0 bg-gradient-to-t from-dark-500 via-transparent to-transparent" />
                        </div>

                        {/* Modal Content */}
                        <div className="p-6 -mt-16 relative z-10">
                            {/* Category & Date */}
                            <div className="flex items-center gap-3 mb-4">
                                <span className="px-3 py-1 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 text-xs font-medium">
                                    {selectedProject.category}
                                </span>
                                <span className="text-gray-400 text-sm flex items-center gap-2">
                                    <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                    {selectedProject.date}
                                </span>
                            </div>

                            {/* Title */}
                            <h2 className="text-2xl md:text-3xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                                {selectedProject.name}
                            </h2>

                            {/* Issued */}
                            <p className="text-gray-400 mb-6">
                                {selectedProject.issued}
                            </p>

                            {/* Description */}
                            <div className="space-y-4 mb-6">
                                {selectedProject.details.map((paragraph, idx) => (
                                    <p key={idx} className="text-gray-300 leading-relaxed">
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
                                        className="inline-flex items-center gap-2 px-6 py-3 bg-white/10 border border-white/20 rounded-lg font-medium hover:bg-white/20 transition-all duration-300"
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
                                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/30 transition-all duration-300"
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