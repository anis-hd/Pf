import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleArrowRight, faPlay, faPause, faVolumeUp, faVolumeMute, faExpand } from "@fortawesome/free-solid-svg-icons";
import { faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import hyperraftVideo from '../assets/hyperraft.mp4';
import walkVideo from '../assets/walk.mp4';
import pfpImage from '../assets/pfp.jpg';

function FeaturedVideo({ src, badge, badgeClass, description, to, githubUrl }) {
    const [isPlaying, setIsPlaying] = useState(true);
    const [isMuted, setIsMuted] = useState(true);
    const [progress, setProgress] = useState(0);
    const [isHovered, setIsHovered] = useState(false);
    const videoRef = useRef(null);

    useEffect(() => {
        const video = videoRef.current;
        if (video) {
            const updateProgress = () => {
                if (video.duration) {
                    const prog = (video.currentTime / video.duration) * 100;
                    setProgress(prog);
                }
            };
            video.addEventListener('timeupdate', updateProgress);
            return () => video.removeEventListener('timeupdate', updateProgress);
        }
    }, []);

    const togglePlay = (e) => {
        if (e) e.stopPropagation();
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause();
            } else {
                videoRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const toggleMute = (e) => {
        if (e) e.stopPropagation();
        if (videoRef.current) {
            videoRef.current.muted = !isMuted;
            setIsMuted(!isMuted);
        }
    };

    const handleFullscreen = (e) => {
        if (e) e.stopPropagation();
        if (videoRef.current) {
            if (videoRef.current.requestFullscreen) {
                videoRef.current.requestFullscreen();
            }
        }
    };

    const handleProgressClick = (e) => {
        e.stopPropagation();
        const rect = e.currentTarget.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        if (videoRef.current && videoRef.current.duration) {
            videoRef.current.currentTime = pos * videoRef.current.duration;
        }
    };

    return (
        <div
            className="relative w-full group rounded-2xl border border-slate-200/90 bg-white shadow-sm hover:shadow-xl hover:border-blue-300/80 transition-all duration-300 overflow-hidden flex flex-col"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Video Container */}
            <div className="relative overflow-hidden bg-slate-950 aspect-video">
                <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    autoPlay
                    loop
                    muted
                    playsInline
                >
                    <source src={src} type="video/mp4" />
                </video>

                {/* Play/Pause Overlay Button */}
                <div
                    className={`absolute inset-0 flex items-center justify-center bg-black/30 transition-opacity duration-300 cursor-pointer ${!isPlaying || isHovered ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                    onClick={togglePlay}
                >
                    <div className="w-14 h-14 rounded-full bg-white/25 backdrop-blur-md flex items-center justify-center border border-white/30 hover:bg-white/35 transition-all duration-200 hover:scale-110 shadow-lg">
                        <FontAwesomeIcon
                            icon={isPlaying ? faPause : faPlay}
                            className="text-white text-lg ml-0.5"
                        />
                    </div>
                </div>

                {/* Custom Overlay Controls */}
                <div className={`absolute bottom-0 left-0 right-0 p-3.5 bg-gradient-to-t from-black/85 via-black/40 to-transparent transition-opacity duration-300 ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                    {/* Progress Bar */}
                    <div
                        className="w-full h-1.5 bg-white/25 rounded-full mb-2.5 cursor-pointer group/progress relative overflow-hidden"
                        onClick={handleProgressClick}
                    >
                        <div
                            className="h-full bg-blue-600 rounded-full relative"
                            style={{ width: `${progress}%` }}
                        >
                            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-white rounded-full opacity-0 group-hover/progress:opacity-100 transition-opacity" />
                        </div>
                    </div>

                    {/* Controls Row */}
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <button
                                type="button"
                                onClick={togglePlay}
                                aria-label={isPlaying ? "Pause video" : "Play video"}
                                className="w-7 h-7 rounded-full bg-white/15 hover:bg-white/25 backdrop-blur-sm flex items-center justify-center text-white transition-colors"
                            >
                                <FontAwesomeIcon icon={isPlaying ? faPause : faPlay} className="text-xs" />
                            </button>
                            <button
                                type="button"
                                onClick={toggleMute}
                                aria-label={isMuted ? "Unmute audio" : "Mute audio"}
                                className="w-7 h-7 rounded-full bg-white/15 hover:bg-white/25 backdrop-blur-sm flex items-center justify-center text-white transition-colors"
                            >
                                <FontAwesomeIcon icon={isMuted ? faVolumeMute : faVolumeUp} className="text-xs" />
                            </button>
                        </div>
                        <button
                            type="button"
                            onClick={handleFullscreen}
                            aria-label="Fullscreen"
                            className="w-7 h-7 rounded-full bg-white/15 hover:bg-white/25 backdrop-blur-sm flex items-center justify-center text-white transition-colors"
                        >
                            <FontAwesomeIcon icon={faExpand} className="text-xs" />
                        </button>
                    </div>
                </div>

                {/* Badge Label - On Video */}
                <div className="absolute top-3 left-3 pointer-events-none">
                    <span className={`px-2.5 py-1 text-white rounded-full text-[11px] font-semibold uppercase tracking-wider shadow-md backdrop-blur-sm ${badgeClass}`}>
                        {badge}
                    </span>
                </div>

                {/* Live Demo Pill Indicator */}
                <div className="absolute top-3 right-3 pointer-events-none">
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-medium tracking-wide bg-black/60 text-white/90 backdrop-blur-md flex items-center gap-1.5 shadow-sm">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                        DEMO
                    </span>
                </div>
            </div>

            {/* Description & Action Footer */}
            <div className="p-4 sm:p-5 flex flex-col justify-between flex-1 bg-white">
                <div>
                    <h4 className="text-base font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                        {badge}
                    </h4>
                    <p className="mt-1.5 text-xs sm:text-sm text-slate-600 font-normal leading-relaxed">
                        {description}
                    </p>
                </div>

                {/* Explore & Source Links */}
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between gap-2 text-xs sm:text-sm font-semibold">
                    <Link
                        to={to}
                        className="flex items-center gap-1.5 text-blue-600 hover:text-blue-700 transition-colors group/link"
                    >
                        <span>Explore Project</span>
                        <FontAwesomeIcon
                            icon={faCircleArrowRight}
                            className="text-xs group-hover/link:translate-x-0.5 transition-transform duration-200"
                        />
                    </Link>

                    {githubUrl && (
                        <a
                            href={githubUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 hover:text-slate-900 transition-colors text-xs font-medium"
                            aria-label="View source code on GitHub"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            <span>Code</span>
                        </a>
                    )}
                </div>
            </div>
        </div>
    );
}

export default function Hiro() {
    return (
        <div
            id="home"
            className="relative flex w-full min-h-[calc(100vh-5rem)] flex-col lg:flex-row gap-12 lg:gap-16 items-center justify-between text-slate-900 pt-28 pb-16 lg:pt-32 lg:pb-24"
        >
            {/* Subtle Ambient Decorative Glows */}
            <div className="absolute top-1/4 -left-12 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none -z-10" />
            <div className="absolute bottom-1/4 -right-12 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl pointer-events-none -z-10" />

            {/* Left Side - Name and Info */}
            <div className="lg:w-1/2 flex flex-col justify-center order-1 lg:order-1">
                <div className="flex flex-col w-full">
                    {/* Profile Picture */}
                    <img
                        src={pfpImage}
                        alt="Anis Houidi"
                        className="w-28 h-28 sm:w-36 sm:h-36 lg:w-44 lg:h-44 rounded-full object-cover object-[center_15%] mb-6"
                    />

                    {/* Large Name */}
                    <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold text-black tracking-tight leading-[1.08] mb-3">
                        Anis Houidi
                    </h1>

                    {/* Dynamic Typewriter Title */}
                    <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2.5">
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
                    <p className="text-base sm:text-lg font-normal text-slate-600 leading-relaxed max-w-xl mb-6">
                        Computer Science Engineer with a strong foundation in <span className="font-semibold text-slate-800">software development</span>, <span className="font-semibold text-slate-800">machine learning</span>, and <span className="font-semibold text-slate-800">data engineering</span>. Experienced in building and deploying end-to-end applications, optimizing models, and developing clean, scalable solutions.
                    </p>

                    {/* Specialization Badges */}
                    <div className="flex flex-wrap items-center gap-2 mb-8">
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
                <div className="flex flex-wrap items-center gap-4">
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

            {/* Right Side - Featured Interactive Demos */}
            <div className="lg:w-1/2 flex justify-center items-center order-2 lg:order-2 w-full">
                <div className="relative w-full max-w-lg flex flex-col gap-6">
                    {/* Header for Demos */}
                    <div className="text-center lg:text-left mb-1">
                        <h3 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                            Featured Highlights
                        </h3>
                    </div>

                    <FeaturedVideo
                        src={hyperraftVideo}
                        badge="Learned Video Codec"
                        badgeClass="bg-blue-600"
                        description="80% of the frames in this video are not real, generated via learned temporal interpolation."
                        to="/hyperraft"
                    />

                    <FeaturedVideo
                        src={walkVideo}
                        badge="Connectome Digital Twin"
                        badgeClass="bg-emerald-600"
                        description="No instructions or training, this biological walk emerges directly from a scanned fruit fly brain."
                        to="/fruitfly"
                        githubUrl="https://github.com/anis-hd/fruit-fly-connectome-digital-twin"
                    />
                </div>
            </div>
        </div>
    );
}

