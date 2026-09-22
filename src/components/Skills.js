import React, { useState } from 'react';
import SkillCard from "./SkillCard.js";

// Import your skill logos
import javascript from "../assets/skills/javascript.svg";
import python from "../assets/skills/python.svg";
import reactIcon from "../assets/skills/react.svg";
import git from "../assets/skills/git.svg";
import tensorflow from "../assets/skills/tensorflow.png";
import pytorch from "../assets/skills/pytorch.svg";
import scikitlearn from "../assets/skills/scikitlearn.png";
import django from "../assets/skills/dj.svg";
import flask from "../assets/skills/flask.svg";
import mongodb from "../assets/skills/mongodb.svg";
import java from "../assets/skills/java-4-logo.svg";
import cee from "../assets/skills/c.png";
import cpp from "../assets/skills/cpp.png";
import docker from "../assets/skills/docker.svg";
import fastapi from "../assets/skills/fastapi.png";
import gitlab from "../assets/skills/gitlab.svg";

// Data Engineering logos
import spark from "../assets/skills/apache-spark-logo-official-png-1.png";
import kafka from "../assets/skills/apache-kafka-logo-official-png-1.png";
import cassandra from "../assets/skills/apache-cassandra-logo-official-png-2.png";
import postgresql from "../assets/skills/postgresql-logo-official-png-1.png";
import powerbi from "../assets/skills/microsoft-power-bi-logo-official-png-1.jpeg";
import talend from "../assets/skills/talend-logo-official-png-1.png";

export default function Skills() {
    const [activeCategory, setActiveCategory] = useState('all');

    // Categorized skills list
    const skills = [
        // AI / ML
        { img: python, name: "Python", category: "aiml" },
        { img: pytorch, name: "PyTorch", category: "aiml" },
        { img: tensorflow, name: "TensorFlow", category: "aiml" },
        { img: scikitlearn, name: "Scikit-learn", category: "aiml" },

        // Data Engineering
        { img: spark, name: "Apache Spark", category: "data" },
        { img: kafka, name: "Apache Kafka", category: "data" },
        { img: postgresql, name: "PostgreSQL", category: "data" },
        { img: cassandra, name: "Cassandra", category: "data" },
        { img: talend, name: "Talend", category: "data" },
        { img: powerbi, name: "Power BI", category: "data" },

        // Backend & APIs
        { img: fastapi, name: "FastAPI", category: "backend" },
        { img: django, name: "Django", category: "backend" },
        { img: flask, name: "Flask", category: "backend" },
        { img: mongodb, name: "MongoDB", category: "backend" },

        // Frontend & Mobile
        { img: reactIcon, name: "React", category: "frontend" },
        { img: javascript, name: "JavaScript", category: "frontend" },

        // DevOps & Systems
        { img: docker, name: "Docker", category: "devops" },
        { img: git, name: "Git", category: "devops" },
        { img: gitlab, name: "GitLab", category: "devops" },
        { img: java, name: "Java", category: "devops" },
        { img: cee, name: "C", category: "devops" },
        { img: cpp, name: "C++", category: "devops" }
    ];

    const categories = [
        { id: 'all', label: 'All', count: skills.length },
        { id: 'aiml', label: 'AI & ML', count: skills.filter(s => s.category === 'aiml').length },
        { id: 'data', label: 'Data Engineering', count: skills.filter(s => s.category === 'data').length },
        { id: 'backend', label: 'Backend & APIs', count: skills.filter(s => s.category === 'backend').length },
        { id: 'frontend', label: 'Frontend', count: skills.filter(s => s.category === 'frontend').length },
        { id: 'devops', label: 'DevOps & Systems', count: skills.filter(s => s.category === 'devops').length }
    ];

    const filteredSkills = activeCategory === 'all'
        ? skills
        : skills.filter(skill => skill.category === activeCategory);

    const toggleCategory = (categoryId) => {
        setActiveCategory(prev => prev === categoryId ? 'all' : categoryId);
    };

    return (
        <section id="skills" className="py-20 text-slate-900 relative">
            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight">
                            Tech Stack
                        </h2>
                        <div className="hidden md:block h-1 w-20 bg-blue-600 rounded-full" />
                    </div>

                    {/* Active Count Pill */}
                    <span className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200/80 w-fit">
                        Showing {filteredSkills.length} of {skills.length} tools
                    </span>
                </div>

                {/* Category Filter Pills */}
                <div className="flex flex-wrap items-center gap-2 sm:gap-2.5 mb-8">
                    {categories.map((cat) => {
                        const isActive = activeCategory === cat.id;
                        return (
                            <button
                                key={cat.id}
                                type="button"
                                onClick={() => setActiveCategory(cat.id)}
                                className={`px-3.5 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition-all duration-200 flex items-center gap-2 ${
                                    isActive
                                        ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25 scale-[1.02]'
                                        : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 hover:text-slate-900 hover:border-slate-300'
                                }`}
                            >
                                <span>{cat.label}</span>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                                    isActive ? 'bg-white/20 text-white' : 'bg-slate-100 text-slate-500'
                                }`}>
                                    {cat.count}
                                </span>
                            </button>
                        );
                    })}
                </div>

                {/* Skills Grid */}
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 xl:grid-cols-11 gap-3 sm:gap-3.5">
                    {filteredSkills.map((skill, index) => (
                        <SkillCard
                            key={`${skill.name}-${index}`}
                            img={skill.img}
                            name={skill.name}
                        />
                    ))}
                </div>

                {/* Interactive Domain Overview Cards */}
                <div className="mt-14 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                    {/* AI/ML */}
                    <div
                        onClick={() => toggleCategory('aiml')}
                        className={`p-5 rounded-2xl bg-white border transition-all duration-300 cursor-pointer group ${
                            activeCategory === 'aiml'
                                ? 'border-blue-600 ring-2 ring-blue-500/20 shadow-md bg-blue-50/20'
                                : 'border-slate-200/90 shadow-xs hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                                AI / ML
                            </span>
                            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100">
                                4 Tools
                            </span>
                        </div>
                        <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
                            Python, PyTorch, TensorFlow, Scikit-learn.
                        </p>
                    </div>

                    {/* Data Engineering */}
                    <div
                        onClick={() => toggleCategory('data')}
                        className={`p-5 rounded-2xl bg-white border transition-all duration-300 cursor-pointer group ${
                            activeCategory === 'data'
                                ? 'border-blue-600 ring-2 ring-blue-500/20 shadow-md bg-blue-50/20'
                                : 'border-slate-200/90 shadow-xs hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                                Data Engineering
                            </span>
                            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100">
                                6 Tools
                            </span>
                        </div>
                        <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
                            Apache Spark, Apache Kafka, PostgreSQL, Cassandra, Talend, Power BI.
                        </p>
                    </div>

                    {/* Backend */}
                    <div
                        onClick={() => toggleCategory('backend')}
                        className={`p-5 rounded-2xl bg-white border transition-all duration-300 cursor-pointer group ${
                            activeCategory === 'backend'
                                ? 'border-blue-600 ring-2 ring-blue-500/20 shadow-md bg-blue-50/20'
                                : 'border-slate-200/90 shadow-xs hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                                Backend &amp; APIs
                            </span>
                            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100">
                                4 Tools
                            </span>
                        </div>
                        <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
                            FastAPI, Django, Flask, MongoDB.
                        </p>
                    </div>

                    {/* Frontend */}
                    <div
                        onClick={() => toggleCategory('frontend')}
                        className={`p-5 rounded-2xl bg-white border transition-all duration-300 cursor-pointer group ${
                            activeCategory === 'frontend'
                                ? 'border-blue-600 ring-2 ring-blue-500/20 shadow-md bg-blue-50/20'
                                : 'border-slate-200/90 shadow-xs hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                                Frontend
                            </span>
                            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100">
                                2 Tools
                            </span>
                        </div>
                        <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
                            React, JavaScript, Web Interfaces.
                        </p>
                    </div>

                    {/* DevOps & Systems */}
                    <div
                        onClick={() => toggleCategory('devops')}
                        className={`p-5 rounded-2xl bg-white border transition-all duration-300 cursor-pointer group ${
                            activeCategory === 'devops'
                                ? 'border-blue-600 ring-2 ring-blue-500/20 shadow-md bg-blue-50/20'
                                : 'border-slate-200/90 shadow-xs hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                                DevOps &amp; Systems
                            </span>
                            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100">
                                6 Tools
                            </span>
                        </div>
                        <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
                            Docker, Git, GitLab, Java, C, C++.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
