import EdCard from "./EdCards.js"

import hr from "../assets/curve-hr.svg"

export default function Education(){
    return (
        <div id="honors" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Honors & Awards</h1>
            <p className="font-light text-gray-400">Here are some of my honors and awards</p>

            <div className="flex flex-col md:flex-row mt-4 gap-5">
                <EdCard name="Nabeul Preparatory Institute for Engineering Studies" issued="Physics and Chemistry" desc="Two years of intensive studies in Mathematics, Physics and Industrial Sciences for the national engineering school entrance exams. Ranked 80/1200 on the PC exam." />
                <EdCard name="National School of Computer Sciences" issued="Computer Science Engineering Degree" desc="Three years of engineering mathematics, computer science, networks, linux and Unix systems, finance and banking systems" />
            </div>
            <img src={hr} className="w-full mt-8 md:h-2" alt="hr" />
        </div>
    )
}
