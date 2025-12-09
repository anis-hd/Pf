import { useState } from 'react';
import EdCard from "./EdCards.js"

import hr from "../assets/curve-hr.svg"

export default function Education() {
    const [isCollapsed, setIsCollapsed] = useState(false);

    return (
        <div id="honors" className="mt-12 text-white transition-all duration-300">
            <div
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                <h1 className="text-4xl font-bold mb-6 border-b-4 border-primary w-fit pb-2">Education</h1>
                <div className={`transform transition-transform duration-300 ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400 group-hover:text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>

            <div className={`overflow-hidden transition-all duration-500 ease-in-out ${isCollapsed ? 'max-h-0 opacity-0' : 'max-h-[1000px] opacity-100'}`}>
                <p className="font-light text-gray-400"></p>

                <div className="flex flex-col md:flex-row mt-4 gap-5">
                    <EdCard name="Preparatory Institute for Engineering Studies of Nabeul" issued="Physics and Chemistry" desc="Two years of intensive studies in Mathematics, Physics and Industrial Sciences for the national engineering contest." />
                    <EdCard name="National School of Computer Science" issued="M.Eng Computer Science, AI & Decision Systems" desc="Computer science, software engineering and applied mathematics." />
                </div>
            </div>
        </div>
    )
}