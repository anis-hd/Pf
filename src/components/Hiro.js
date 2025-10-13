import React from "react";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {faCircleArrowRight, } from "@fortawesome/free-solid-svg-icons";
import {  faFacebook, faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import WrapperComponent from "./WrapperComponent";
export default function Hiro () {



    return (
        <>
        <div id="home" className="flex w-full h-screen flex-col md:flex-row gap-5 items-center justify-center text-white relative">
            <div className='md:w-3/6 md:p-4'>
                <div className="flex items-center justify-center h-screen">
                    <div className="relative text-white-800 text-[20rem] font-bold">

                            <h2>HI.</h2>

                    </div>
                </div>
            </div>
            <div className='md:w-3/6' data-aos="fade-right" data-aos-duration="1000" data-aos-offset="100">
                <div className="flex flex-col w-full mt-8">
                    <h1 className="text-2xl font-bold">Anis Ben Houidi</h1>
                    <p class="text-xl font-bold text-gray-300">A <Typewriter text="Computer Science Engineering Student " delay={120} infinite/></p>
                    <p className="text-md font-light text-gray-400 ">A final year computer science engineering student with a passion for solving complex challenges, with out-of-the-box thinking and strong analytical skills. Interested in emerging technologies, especially Artificial
Intelligence and Machine Learning, with the goal of leveraging my education and expertise to contribute to
innovative projects, gain valuable industry experience, and continue to grow professionally.</p>
                </div>
                <a href='https://www.linkedin.com/in/anis-ben-houidi/' className='mt-2 block'>Lets connect!<FontAwesomeIcon className='ml-2' icon={faCircleArrowRight}/> </a>

                <ul className='flex mt-2 gap-4 items-center'>
                   <li>
                        <a href='https://github.com/anis-hd' rel="noreferrer" target="_blank"><FontAwesomeIcon size='2xl' icon={faGithub} /></a>
                   </li>
                    <li>
                        <a href='https://www.linkedin.com/in/anis-ben-houidi/' rel="noreferrer" target="_blank"><FontAwesomeIcon size='2xl' icon={faLinkedinIn} /></a>
                    </li>
                </ul>
            </div>
        </div>
        </>
    )
}