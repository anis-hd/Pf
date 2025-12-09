import React, { useState, useEffect } from "react";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleArrowRight, } from "@fortawesome/free-solid-svg-icons";
import { faFacebook, faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import WrapperComponent from "./WrapperComponent";

export default function Hiro() {
    const [mousePos, setMousePos] = useState({ x: 50, y: 50 });

    useEffect(() => {
        const handleMouseMove = (e) => {
            // Calculate percentage based on window size
            const x = (e.clientX / window.innerWidth) * 100;
            const y = (e.clientY / window.innerHeight) * 100;
            setMousePos({ x, y });
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    return (
        <>
            <div id="home" className="flex w-full min-h-screen flex-col md:flex-row gap-10 items-center justify-center text-white relative pt-20">
                <div className='md:w-1/2 flex justify-center items-center'>
                    <h2
                        className="text-[12rem] md:text-[15rem] font-bold leading-none bg-clip-text text-transparent select-none transition-all duration-75 ease-out"
                        style={{
                            backgroundImage: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, #6366f1, #d946ef, #0ea5e9)`
                        }}
                    >
                        HI.
                    </h2>
                </div>
                <div className='md:w-1/2 flex flex-col justify-center' data-aos="fade-left" data-aos-duration="1000">
                    <div className="flex flex-col w-full">
                        <h1 className="text-4xl md:text-6xl font-bold mb-4">Anis Ben Houidi</h1>
                        <p className="text-xl md:text-2xl font-semibold text-gray-300 mb-6">A <span className="text-primary"><Typewriter text="Computer Science Engineering Student " delay={120} infinite /></span></p>
                        <p className="text-lg font-light text-gray-400 leading-relaxed max-w-xl">
                            A computer science engineering graduate with a passion for solving complex challenges through out-of-the-box thinking and strong analytical skills. Interested in emerging technologies, especially Artificial Intelligence and Machine Learning.
                        </p>
                    </div>

                    <div className="flex items-center gap-6 mt-8">
                        <a href='https://www.linkedin.com/in/anis-ben-houidi/' className='inline-flex items-center gap-2 bg-primary hover:bg-secondary px-8 py-3 transition-all duration-300 font-medium shadow-lg hover:shadow-primary/50'>
                            Lets connect! <FontAwesomeIcon icon={faCircleArrowRight} />
                        </a>

                        <div className='flex gap-4 items-center'>
                            <a href='https://github.com/anis-hd' rel="noreferrer" target="_blank" className="text-gray-400 hover:text-white hover:scale-110 transition-all duration-300">
                                <FontAwesomeIcon size='2xl' icon={faGithub} />
                            </a>
                            <a href='https://www.linkedin.com/in/anis-ben-houidi/' rel="noreferrer" target="_blank" className="text-gray-400 hover:text-white hover:scale-110 transition-all duration-300">
                                <FontAwesomeIcon size='2xl' icon={faLinkedinIn} />
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}