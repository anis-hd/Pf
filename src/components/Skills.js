import React, { useState, useEffect } from 'react';
import SkillCard from "./SkillCard.js";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';


// Import your skill logos
import javascript from "../assets/skills/javascript.svg";
import bash from "../assets/skills/bash.svg";
import linux from "../assets/skills/linux.svg";
import python from "../assets/skills/python.svg";
import reactIcon from "../assets/skills/react.svg";
import tailwind from "../assets/skills/tailwind.svg";
import git from "../assets/skills/git.svg";
import tensorflow from "../assets/skills/tensorflow.png";
import pytorch from "../assets/skills/pytorch.svg";
import scikitlearn from "../assets/skills/scikitlearn.png";
import django from "../assets/skills/dj.svg";
import flask from "../assets/skills/flask.svg";
import nodejsicon from "../assets/skills/nodejs-icon.svg";
import mongodb from "../assets/skills/mongodb.svg";
import java from "../assets/skills/java-4-logo.svg";
import cee from "../assets/skills/c.png";
import cpp from "../assets/skills/cpp.png";
import flutter from "../assets/skills/flutter.svg";

import docker from "../assets/skills/docker.svg";
import express from "../assets/skills/express.svg";
import fastapi from "../assets/skills/fastapi.png";
import gitlab from "../assets/skills/gitlab.svg";
import html from "../assets/skills/html.svg";
import huggingface from "../assets/skills/huggingface.png";

import langchain from "../assets/skills/langchain.png";

import ollama from "../assets/skills/ollama.webp";
import unsloth from "../assets/skills/unsloth-logo.webp";



export default function Skills() {
    // An array of skill images with names for better accessibility
    const skills = [
        { img: python, name: "Python" },
        { img: pytorch, name: "PyTorch" },
        { img: tensorflow, name: "TensorFlow" },
        { img: huggingface, name: "Hugging Face" },
        { img: langchain, name: "LangChain" },
        { img: ollama, name: "Ollama" },
        { img: unsloth, name: "Unsloth" },
        { img: scikitlearn, name: "Scikit-learn" },
        { img: javascript, name: "JavaScript" },
        { img: reactIcon, name: "React" },
        { img: html, name: "HTML" },
        { img: flutter, name: "Flutter" },
        { img: nodejsicon, name: "Node.js" },
        { img: express, name: "Express" },
        { img: django, name: "Django" },
        { img: flask, name: "Flask" },
        { img: fastapi, name: "FastAPI" },
        { img: mongodb, name: "MongoDB" },
        { img: tailwind, name: "Tailwind" },
        { img: git, name: "Git" },
        { img: gitlab, name: "GitLab" },
        { img: docker, name: "Docker" },

        { img: linux, name: "Linux" },
        { img: bash, name: "Bash" },

        { img: java, name: "Java" },
        { img: cee, name: "C" },
        { img: cpp, name: "C++" }
    ];

    const [mousePosition, setMousePosition] = useState({ x: null, y: null });
    const [mousePos, setMousePos] = useState({ x: 50, y: 50 });

    useEffect(() => {
        const handleMouseMove = (event) => {
            setMousePosition({ x: event.clientX, y: event.clientY });
            // For gradient effect
            const x = (event.clientX / window.innerWidth) * 100;
            const y = (event.clientY / window.innerHeight) * 100;
            setMousePos({ x, y });
        };

        window.addEventListener('mousemove', handleMouseMove);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);



    return (
        <section id="skills" className="py-20 text-white relative">
            {/* Background Gradient Orb */}
            <div
                className="absolute inset-0 pointer-events-none overflow-hidden"
                style={{
                    background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, rgba(99, 102, 241, 0.08) 0%, transparent 50%)`
                }}
            />

            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-white">
                            Tech Stack
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 rounded-full" />
                    </div>
                </div>

                {/* Content */}
                <div className="opacity-100 transition-all duration-500 ease-in-out">
                    {/* Subtitle */}
                    <p className="text-gray-400 text-lg mb-2">
                        Technologies I work with
                    </p>


                    {/* Skills Grid */}
                    <div className="flex justify-center">
                        <div className="inline-grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-9 gap-4">
                            {skills.map((skill, index) => (
                                <SkillCard
                                    key={index}
                                    img={skill.img}
                                    name={skill.name}
                                    mousePosition={mousePosition}
                                />
                            ))}
                        </div>
                    </div>

                    <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                AI/ML
                            </div>
                            <p className="text-gray-500 text-sm mt-1">PyTorch, TensorFlow, Hugging Face, LangChain</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-pink-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                Frontend
                            </div>
                            <p className="text-gray-500 text-sm mt-1">React, Flutter, Tailwind, HTML</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-blue-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                Backend
                            </div>
                            <p className="text-gray-500 text-sm mt-1">Django, FastAPI, Node.js, Express</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-green-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                DevOps
                            </div>
                            <p className="text-gray-500 text-sm mt-1">Docker, Git, GitLab, Linux</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}