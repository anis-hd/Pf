import { useState } from 'react';
import ProjectCard from "./ProjectCards.js";
import Modal from "./Modal.js";

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
        details: [
            "This project focused on the creative application of Generative Adversarial Networks (GANs). The goal was to train a model that could colorize and add texture to a simple line-art sketch, effectively turning it into a photorealistic image.",
            "A Pix2Pix model, which is a type of Conditional GAN, was used. It was trained on a large dataset of paired images (sketches and their corresponding photos) to learn the complex mapping from the sketch domain to the realistic image domain."
        ]
    },
    {
        name: "Big data user behavior analysis",
        img: bigdata,
        issued: "Academic Project",
        desc: "Set up a distributed, multi-node HDFS environment in Docker. Used MapReduce, Pig, and Spark for data processing, and Hive for querying.",
        date: "November 2024",
        details: [
            "This project involved analyzing a massive dataset of user behavior logs to extract actionable insights. A multi-node Hadoop Distributed File System (HDFS) was simulated using Docker to handle the sheer volume of data.",
            "The data processing pipeline began with MapReduce for initial filtering. Apache Pig was used for scripting complex data transformations, followed by Apache Spark for faster, in-memory computations. Finally, Apache Hive was layered on top, allowing for complex SQL-like queries to uncover user buying patterns and engagement metrics."
        ]
    },
    {
        name: "Business Intelligence data pipeline",
        img: bi,
        issued: "Academic Project",
        desc: "Used Apache NiFi with Power BI to automate the ingestion, processing, and visualization of data from a CSV file.",
        date: "October 2024",
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
        details: [
            "Essence is a mobile app designed to be a helpful tool for chemistry students. It features an interactive periodic table, a molar mass calculator, and other useful utilities.",
            "The application was built using Flutter and the Dart programming language, allowing for a single codebase to be deployed on both Android and iOS. This project was a great introduction to cross-platform development and state management in Flutter."
        ]
    }
];

export default function Projects() {
    const [selectedProject, setSelectedProject] = useState(null);

    const handleCardClick = (project) => {
        setSelectedProject(project);
    };

    const handleCloseModal = () => {
        setSelectedProject(null);
    };

    return (
        <div id="certs" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Projects</h1>
            <p className="font-light text-gray-400">Here are some of my projects</p>

            <div className="grid grid-cols-1 md:grid-cols-3 justify-center mt-4 gap-5">
                {projectData.map((project, index) => (
                    <ProjectCard 
                        key={index} 
                        project={project}
                        onClick={handleCardClick}
                    />
                ))}
            </div>
            
            <Modal project={selectedProject} onClose={handleCloseModal} />
        </div>
    );
}