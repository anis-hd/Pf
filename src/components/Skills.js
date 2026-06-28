import React from 'react';
import SkillCard from "./SkillCard.js";


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

    return (
        <section id="skills" className="py-20 text-slate-900 relative">
            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-slate-900">
                            Tech Stack
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-blue-600 rounded-full" />
                    </div>
                </div>

                {/* Content */}
                <div className="opacity-100 transition-all duration-500 ease-in-out">
                    {/* Subtitle */}
                    <p className="text-slate-500 text-lg mb-2">
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
                                />
                            ))}
                        </div>
                    </div>

                    <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 group">
                            <div className="text-2xl font-bold text-slate-900">
                                AI/ML
                            </div>
                            <p className="text-slate-600 text-sm mt-1">PyTorch, TensorFlow, Hugging Face, Agentic Workflows</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 group">
                            <div className="text-2xl font-bold text-slate-900">
                                Frontend
                            </div>
                            <p className="text-slate-600 text-sm mt-1">React, NextJs, Flutter, Tailwind </p>
                        </div>
                        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 group">
                            <div className="text-2xl font-bold text-slate-900">
                                Backend
                            </div>
                            <p className="text-slate-600 text-sm mt-1">Django, FastAPI, Node.js, Express</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 group">
                            <div className="text-2xl font-bold text-slate-900">
                                DevOps
                            </div>
                            <p className="text-slate-600 text-sm mt-1">Docker, Git, GitLab, Linux</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}