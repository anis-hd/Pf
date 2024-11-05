import React from "react";
import Typewriter from './Typewriter';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {faCircleArrowRight, } from "@fortawesome/free-solid-svg-icons";
import {  faFacebook, faGithub, faLinkedinIn } from "@fortawesome/free-brands-svg-icons";
import hr from '../assets/curve-hr.svg';
import profile from "../assets/profile.png";
export default function Hiro () {



    return (
        <>
        <div id="home" className="flex w-full h-screen flex-col md:flex-row gap-5 items-center justify-center text-white relative">
            <div className='md:w-3/6 md:p-4'>
            <img src={profile} alt="profile"         style={{
                marginLeft: '20%',
          borderRadius: '30%', 
          width: '200px', 
          height: '200px', 
        }}
/>


            </div>
            <div className='md:w-3/6' data-aos="fade-right" data-aos-duration="1000" data-aos-offset="100" >
                <div className="flex flex-col w-full mt-8">
                    <h1 className="text-xl text-gray-400">Hi, I'm</h1>
                    <h1 className="text-2xl font-bold">Anis Ben Houidi</h1>
                    <p class="text-xl font-bold text-gray-300">I'm a <Typewriter text="Computer Science Engineering Student " delay={120} infinite/></p>
                    <p className="text-md font-light text-gray-400 ">A final year computer science engineering student with a passion for solving complex challenges, with out-of-the-box thinking and strong analytical skills. Interested in emerging technologies, especially Artificial
Intelligence and Machine Learning, with the goal of leveraging my education and expertise to contribute to
innovative projects, gain valuable industry experience, and continue to grow professionally.</p>
                </div>
                <a href='https://www.linkedin.com/in/anis-ben-houidi/' className='mt-2 block'>Lets connect!<FontAwesomeIcon className='ml-2' icon={faCircleArrowRight}/> </a>
                
                <ul className='flex mt-2 gap-4 items-center'>
                   <li>
                        <a href='https://github.com/4nisHd' rel="noreferrer" target="_blank"><FontAwesomeIcon size='2xl' icon={faGithub} /></a>
                   </li> 
                    <li>
                        <a href='https://www.linkedin.com/in/anis-ben-houidi/' rel="noreferrer" target="_blank"><FontAwesomeIcon size='2xl' icon={faLinkedinIn} /></a>
                    </li>
                </ul>
            </div>
            <img src={hr} className="w-full md:h-2 absolute bottom-0" alt="hr" />
        </div>
        </>
    )
}