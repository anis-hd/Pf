import ProjectCard from "./ProjectCards.js"

import hr from "../assets/curve-hr.svg"
import RadAi from "../assets/certs/RadAi.png"
import portfolio from "../assets/certs/portfolio.jpg"
import arabic from "../assets/certs/arabic.jpg"
import mcode from "../assets/certs/mcode.jpg"
import wordle from "../assets/certs/wordle.webp"
import Ptable from "../assets/certs/Ptable.png"
import crypt from "../assets/certs/crypt.jpg"
import bigdata from "../assets/certs/bigdata.jpg"
import bi from "../assets/certs/bi.webp"
// import 

export default function Projects(){
    return (
        <div id="certs" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Projects</h1>
            <p className="font-light text-gray-400">Here are some of my projects</p>

            {/* <div className="flex flex-col md:flex-row flex-wrap mt-4 gap-5"> */}
            <div className="grid grid-cols-1 md:grid-cols-3 justify-center mt-4 gap-5">
                <ProjectCard name="Big data user behavior analysis" img={bigdata} issued="Set up a distributed, multi-node HDFS environment in Docker. Used MapReduce to filter raw data, followed by data cleaning with Pig and Spark to create structured datasets. Leveraged Hive for complex queries, extracting insights on buying patterns." date="November 2024" />
                <ProjectCard name="Business Intelligence data pipeline and visualization" img={bi} issued="Used Apache NiFi with Power BI to automate the ingestion, processing, and visualization of data from a CSV file, enabling real-time insights through dashboards." date="October 2024" />

                <ProjectCard name="Cryptographic key rotation optimization with Deep Reinforcement Learning" img={crypt} issued="This was a proof of concept for DeepFlow AI Hackathon, winning first place from the technical jury. Keywords: Deep Q-Networks, Tensorflow, Gym" date="October 2024" />

                <ProjectCard name="MERN Stack Arabic NLP Platform" img={arabic} issued="Personal Project, Technologies used: React, Express, Node, MongoDB, Flask APIs, LLMs, Generative AI, NLP" date="Feb 2024 - Ongoing (Incompleted)" />
                <ProjectCard name="AI Radiology platform" img={RadAi} issued="ENSI PCD, Technologies used: Django, SQL, Image Processing, CNNs" date="Jan-May 2023" />
                <ProjectCard name="Personal Portfolio" img={portfolio} issued="React, TailwindCSS, Firebase" date="Sept - Dec 2023" />
                <ProjectCard name="Language Specific Compiler" img={mcode} issued="Compilation Theory, Lexical Analysis, Syntaxical Analysis, Semantic Analysis, C" date="Nov - Dec 2023" />
                <ProjectCard name="Wordle Clone in Android Studio" img={wordle} issued="Android Studio, Java, OOP" date="June - July 2022" />
                <ProjectCard name="Essence: A chemistry Companion App" img={Ptable} issued="Flutter, Dart, OOP" date="March - May 2022" />
            </div>
            <img src={hr} className="w-full mt-8 md:h-2" alt="hr" />
        </div>
    )
}