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

    return (
        <div id="skills" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Skills</h1>
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
    );
}