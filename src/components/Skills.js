import React, { useState, useEffect } from 'react';
import SkillCard from "./SkillCard.js";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown } from '@fortawesome/free-solid-svg-icons';

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


export default function Skills() {
    // An array of skill images with names for better accessibility
    const skills = [
        { img: python, name: "Python" },
        { img: pytorch, name: "PyTorch" },
        { img: tensorflow, name: "TensorFlow" },
        { img: scikitlearn, name: "Scikit-learn" },
        { img: javascript, name: "JavaScript" },
        { img: reactIcon, name: "React" },
        { img: nodejsicon, name: "Node.js" },
        { img: django, name: "Django" },
        { img: flask, name: "Flask" },
        { img: mongodb, name: "MongoDB" },
        { img: tailwind, name: "Tailwind" },
        { img: git, name: "Git" },
        { img: linux, name: "Linux" },
        { img: bash, name: "Bash" },
        { img: java, name: "Java" },
        { img: cee, name: "C" },
        { img: cpp, name: "C++" },
        { img: flutter, name: "Flutter" }
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

    const [isCollapsed, setIsCollapsed] = useState(false);

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
                <div
                    className="flex items-center justify-between cursor-pointer group mb-8"
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    data-aos="fade-up"
                >
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-white">
                            Tech Stack
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
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[2000px] opacity-100'}`}>
                    {/* Subtitle */}
                    <p className="text-gray-400 text-lg mb-2" data-aos="fade-up" data-aos-delay="100">
                        Technologies I work with
                    </p>


                    {/* Skills Grid */}
                    <div className="flex justify-center" data-aos="fade-up" data-aos-delay="200">
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

                    {/* Categories Summary */}
                    <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4" data-aos="fade-up" data-aos-delay="300">
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                AI/ML
                            </div>
                            <p className="text-gray-500 text-sm mt-1">PyTorch, TensorFlow, Scikit-learn</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-pink-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                Frontend
                            </div>
                            <p className="text-gray-500 text-sm mt-1">React, JavaScript, Tailwind</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-blue-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                Backend
                            </div>
                            <p className="text-gray-500 text-sm mt-1">Django, Flask, Node.js</p>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-green-500/30 transition-all duration-300 group">
                            <div className="text-2xl font-bold text-white">
                                DevOps
                            </div>
                            <p className="text-gray-500 text-sm mt-1">Linux, Bash, Git</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}