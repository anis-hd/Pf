import EdCard from "./EdCards.js"

import hr from "../assets/curve-hr.svg"

export default function Education(){
    return (
        <div id="honors" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Education</h1>
            <p className="font-light text-gray-400"></p>

            <div className="flex flex-col md:flex-row mt-4 gap-5">
                <EdCard name="Preparatory Institute for Engineering Studies of Nabeul" issued="Physics and Chemistry" desc="Two years of intensive studies in Mathematics, Physics and Industrial Sciences for the national engineering contest." />
                <EdCard name="National School of Computer Science" issued="M.Eng Computer Science, AI & Decision Systems" desc="Computer science, software engineering and applied mathematics." />
            </div>
        </div>
    )
}