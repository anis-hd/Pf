import React, { useState, useEffect } from 'react';
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


export default function Skills() {
    // An array of skill images
    const skills = [
        django, flask, linux, bash, python, javascript, reactIcon, tailwind,
        git, tensorflow, pytorch, scikitlearn, nodejsicon, mongodb, java,
        cee, cpp, flutter
    ];

    const [mousePosition, setMousePosition] = useState({ x: null, y: null });

    useEffect(() => {
        const handleMouseMove = (event) => {
            setMousePosition({ x: event.clientX, y: event.clientY });
        };

        window.addEventListener('mousemove', handleMouseMove);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    const [isCollapsed, setIsCollapsed] = useState(false);

    return (
        <div id="skills" className="mt-12 text-white transition-all duration-300">
            <div
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                <h1 className="text-4xl font-bold mb-6 border-b-4 border-primary w-fit pb-2">Skills</h1>
                <div className={`transform transition-transform duration-300 ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400 group-hover:text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>

            <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[1000px] opacity-100'}`}>
                <p className="font-light text-gray-400">Hover over the skills to see the effect</p>

                {/* Centering wrapper */}
                <div className="mt-8 flex justify-center">
                    {/* Use inline-grid to make the container only as wide as its content */}
                    <div className="inline-grid grid-cols-4 md:grid-cols-6 lg:grid-cols-9 gap-[10px]">
                        {skills.map((skillImg, index) => (
                            <SkillCard key={index} img={skillImg} mousePosition={mousePosition} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}