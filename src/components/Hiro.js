import React from "react";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleArrowRight } from "@fortawesome/free-solid-svg-icons";
import { faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import pfpImage from '../assets/pfp.jpg';

export default function Hiro() {
    return (
        <div
            id="home"
            className="relative flex w-full min-h-[calc(100vh-5rem)] flex-col items-center justify-center text-slate-900 pt-28 pb-16 lg:pt-32 lg:pb-24"
        >
            {/* Subtle Ambient Decorative Glows */}
            <div className="absolute top-1/4 -left-12 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none -z-10" />
            <div className="absolute bottom-1/4 -right-12 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl pointer-events-none -z-10" />

            {/* Main Intro - centered now that featured highlights live in Projects */}
            <div className="w-full max-w-3xl flex flex-col justify-center items-center text-center mx-auto">
                <div className="flex flex-col w-full items-center text-center">
                    {/* Profile Picture */}
                    <img
                        src={pfpImage}
                        alt="Anis Houidi"
                        className="w-28 h-28 sm:w-36 sm:h-36 lg:w-44 lg:h-44 rounded-full object-cover object-[center_15%] mb-6 mx-auto"
                    />

                    {/* Large Name */}
                    <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold text-black tracking-tight leading-[1.08] mb-3">
                        Anis Houidi
                    </h1>

                    {/* Dynamic Typewriter Title */}
                    <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-800 mb-4 flex items-center justify-center gap-2.5">
                        <span className="w-6 h-1 bg-blue-600 rounded-full inline-block" />
                        <span className="text-blue-600 min-h-[34px] flex items-center">
                            <Typewriter
                                texts={["Software Engineer", "AI Engineer", "Data Science Enthusiast"]}
                                delay={80}
                                infinite
                            />
                        </span>
                    </div>

                    {/* Bio Description */}
                    <p className="text-base sm:text-lg font-normal text-slate-600 leading-relaxed max-w-xl mb-6 mx-auto">
                        Computer Science Engineer with a strong foundation in <span className="font-semibold text-slate-800">software development</span>, <span className="font-semibold text-slate-800">machine learning</span>, and <span className="font-semibold text-slate-800">data engineering</span>. Experienced in building and deploying end-to-end applications, optimizing models, and developing clean, scalable solutions.
                    </p>

                    {/* Specialization Badges */}
                    <div className="flex flex-wrap items-center justify-center gap-2 mb-8">
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-100/90 text-slate-700 border border-slate-200/80">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                            Full-Stack Systems
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-100/90 text-slate-700 border border-slate-200/80">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                            Deep Learning &amp; AI
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-100/90 text-slate-700 border border-slate-200/80">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                            End-to-End Delivery
                        </span>
                    </div>
                </div>

                {/* Call To Action Buttons & Socials */}
                <div className="flex flex-wrap items-center justify-center gap-4">
                    <a
                        href="https://www.linkedin.com/in/anis-ben-houidi/"
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2.5 px-6 py-3.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/35 hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 group"
                    >
                        <span>Let's connect!</span>
                        <FontAwesomeIcon icon={faCircleArrowRight} className="group-hover:translate-x-1 transition-transform duration-200" />
                    </a>

                    <a
                        href="#certs"
                        className="inline-flex items-center gap-2 px-5 py-3.5 bg-white border border-slate-200/90 text-slate-700 rounded-xl font-medium shadow-xs hover:border-slate-300 hover:text-blue-600 hover:bg-slate-50 hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200"
                    >
                        <span>View Projects</span>
                    </a>

                    <div className="flex gap-2.5 items-center pl-1">
                        <a
                            href="https://github.com/anis-hd"
                            rel="noreferrer"
                            target="_blank"
                            aria-label="GitHub Profile"
                            className="w-12 h-12 rounded-xl bg-white border border-slate-200/90 shadow-xs flex items-center justify-center text-slate-600 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-300 hover:-translate-y-0.5 transition-all duration-200 group"
                        >
                            <FontAwesomeIcon icon={faGithub} className="text-xl group-hover:scale-110 transition-transform duration-200" />
                        </a>
                        <a
                            href="https://www.linkedin.com/in/anis-ben-houidi/"
                            rel="noreferrer"
                            target="_blank"
                            aria-label="LinkedIn Profile"
                            className="w-12 h-12 rounded-xl bg-white border border-slate-200/90 shadow-xs flex items-center justify-center text-slate-600 hover:text-blue-600 hover:bg-slate-50 hover:border-slate-300 hover:-translate-y-0.5 transition-all duration-200 group"
                        >
                            <FontAwesomeIcon icon={faLinkedinIn} className="text-xl group-hover:scale-110 transition-transform duration-200" />
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
}

