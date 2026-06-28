import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleArrowRight, faPlay, faPause, faVolumeUp, faVolumeMute, faExpand } from "@fortawesome/free-solid-svg-icons";
import { faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import hyperraftVideo from '../assets/hyperraft.mp4';

export default function Hiro() {
    const [isPlaying, setIsPlaying] = useState(true);
    const [isMuted, setIsMuted] = useState(true);
    const [progress, setProgress] = useState(0);
    const [isHovered, setIsHovered] = useState(false);
    const videoRef = useRef(null);

    useEffect(() => {
        const video = videoRef.current;
        if (video) {
            const updateProgress = () => {
                const progress = (video.currentTime / video.duration) * 100;
                setProgress(progress);
            };
            video.addEventListener('timeupdate', updateProgress);
            return () => video.removeEventListener('timeupdate', updateProgress);
        }
    }, []);

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause();
            } else {
                videoRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const toggleMute = () => {
        if (videoRef.current) {
            videoRef.current.muted = !isMuted;
            setIsMuted(!isMuted);
        }
    };

    const handleFullscreen = () => {
        if (videoRef.current) {
            if (videoRef.current.requestFullscreen) {
                videoRef.current.requestFullscreen();
            }
        }
    };

    const handleProgressClick = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        if (videoRef.current) {
            videoRef.current.currentTime = pos * videoRef.current.duration;
        }
    };

    return (
        <>
            <div id="home" className="flex w-full min-h-screen flex-col lg:flex-row gap-10 lg:gap-16 items-center justify-center text-slate-900 relative pt-20">
                {/* Left Side - Name and Info */}
                <div className='lg:w-1/2 flex flex-col justify-center order-2 lg:order-1'>
                    <div className="flex flex-col w-full">
                        {/* Large Name */}
                        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-4 leading-tight text-slate-900">
                            Anis Houidi
                        </h1>
                        <p className="text-xl md:text-2xl font-semibold text-slate-700 mb-6">
                            <span className="text-blue-600">
                                <Typewriter
                                    texts={["Software Engineer", "AI Engineer", "Data Science Enthusiast"]}
                                    delay={80}
                                    infinite
                                />
                            </span>
                        </p>
                        <p className="text-lg font-light text-slate-600 leading-relaxed max-w-xl">
                            Computer Science Engineer with a foundation in software development, machine learning, and data engineering.
                            Experienced in building and deploying end-to-end applications, optimizing models, and developing clean, scalable
                            solutions.                        </p>
                    </div>

                    <div className="flex items-center gap-4 mt-8">
                        <a
                            href='https://www.linkedin.com/in/anis-ben-houidi/'
                            target="_blank"
                            rel="noreferrer"
                            className='inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-full font-medium shadow-sm hover:bg-blue-700 hover:shadow transition-all duration-300'
                        >
                            Let's connect!
                            <FontAwesomeIcon icon={faCircleArrowRight} className="group-hover:translate-x-1 transition-transform" />
                        </a>

                        <div className='flex gap-3 items-center'>
                            <a
                                href='https://github.com/anis-hd'
                                rel="noreferrer"
                                target="_blank"
                                className="w-12 h-12 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center text-slate-500 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-300 transition-all duration-300 group"
                            >
                                <FontAwesomeIcon icon={faGithub} className="text-xl group-hover:scale-110 transition-transform" />
                            </a>
                            <a
                                href='https://www.linkedin.com/in/anis-ben-houidi/'
                                rel="noreferrer"
                                target="_blank"
                                className="w-12 h-12 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center text-slate-500 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-300 transition-all duration-300 group"
                            >
                                <FontAwesomeIcon icon={faLinkedinIn} className="text-xl group-hover:scale-110 transition-transform" />
                            </a>
                        </div>
                    </div>
                </div>

                {/* Right Side - Video Player */}
                <div className='lg:w-1/2 flex justify-center items-center order-1 lg:order-2'>
                    <div
                        className="relative w-full max-w-lg group"
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                    >
                        {/* "Check This Out" Heading - Above Video */}
                        <h3 className="text-2xl md:text-3xl font-bold mb-4 text-center text-slate-900">
                            Check This Out!
                        </h3>

                        {/* Video Container */}
                        <div className="relative">
                            <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-white shadow-lg">
                                {/* Video Element */}
                                <video
                                    ref={videoRef}
                                    className="w-full h-auto aspect-video object-cover"
                                    autoPlay
                                    loop
                                    muted
                                    playsInline
                                >
                                    <source src={hyperraftVideo} type="video/mp4" />
                                </video>

                                {/* Play/Pause Overlay */}
                                <div
                                    className={`absolute inset-0 flex items-center justify-center bg-black/30 transition-opacity duration-300 cursor-pointer ${isHovered ? 'opacity-100' : 'opacity-0'}`}
                                    onClick={togglePlay}
                                >
                                    <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-md flex items-center justify-center hover:bg-white/30 transition-all duration-300 hover:scale-110">
                                        <FontAwesomeIcon
                                            icon={isPlaying ? faPause : faPlay}
                                            className="text-white text-xl ml-0.5"
                                        />
                                    </div>
                                </div>

                                {/* Custom Controls */}
                                <div className={`absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent transition-opacity duration-300 ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                                    {/* Progress Bar */}
                                    <div
                                        className="w-full h-1 bg-white/20 rounded-full mb-3 cursor-pointer group/progress"
                                        onClick={handleProgressClick}
                                    >
                                        <div
                                            className="h-full bg-blue-600 rounded-full relative"
                                            style={{ width: `${progress}%` }}
                                        >
                                            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover/progress:opacity-100 transition-opacity"></div>
                                        </div>
                                    </div>

                                    {/* Control Buttons */}
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-4">
                                            <button
                                                onClick={togglePlay}
                                                className="text-white hover:text-blue-400 transition-colors"
                                            >
                                                <FontAwesomeIcon icon={isPlaying ? faPause : faPlay} />
                                            </button>
                                            <button
                                                onClick={toggleMute}
                                                className="text-white hover:text-blue-400 transition-colors"
                                            >
                                                <FontAwesomeIcon icon={isMuted ? faVolumeMute : faVolumeUp} />
                                            </button>
                                        </div>
                                        <button
                                            onClick={handleFullscreen}
                                            className="text-white hover:text-blue-400 transition-colors"
                                        >
                                            <FontAwesomeIcon icon={faExpand} />
                                        </button>
                                    </div>
                                </div>

                                {/* "Learned Video Compression" Label - On Video */}
                                <div className="absolute top-4 left-4">
                                    <span className="px-3 py-1.5 bg-blue-600 text-white rounded-full text-xs font-semibold uppercase tracking-wider shadow">
                                        Learned Video Codec
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Description Text - Below Video */}
                        <p className="mt-4 text-center text-slate-500 text-sm md:text-base font-light">
                            80% of the frames in this video are reconstructed during decoding.
                        </p>

                        {/* Click for More Link */}
                        <Link
                            to="/hyperraft"
                            className="mt-3 flex items-center justify-center gap-2 text-slate-500 hover:text-blue-600 transition-all duration-300 group/link"
                        >
                            <span className="text-sm font-medium">Click for more</span>
                            <FontAwesomeIcon
                                icon={faCircleArrowRight}
                                className="group-hover/link:translate-x-1 transition-transform duration-300"
                            />
                        </Link>
                    </div>
                </div>
            </div>
        </>
    )
}