import React from 'react';

export default function SkillCard({ img, name }) {
    return (
        <div className="group relative flex flex-col items-center justify-center p-2.5 sm:p-3 bg-white rounded-2xl border border-slate-200/90 shadow-xs hover:shadow-lg hover:shadow-blue-500/10 hover:border-blue-400 hover:-translate-y-1 transition-all duration-300 cursor-default">
            {/* Logo Container */}
            <div className="w-10 h-10 sm:w-12 sm:h-12 flex items-center justify-center rounded-xl bg-slate-50 group-hover:bg-blue-50/60 p-2 transition-colors duration-300">
                <img
                    src={img}
                    alt={name || "Skill logo"}
                    className="w-full h-full object-contain filter-none group-hover:scale-110 transition-transform duration-300"
                    loading="lazy"
                />
            </div>

            {/* Label */}
            <span className="mt-2 text-[11px] sm:text-xs font-semibold text-slate-700 text-center tracking-tight truncate w-full group-hover:text-blue-600 transition-colors">
                {name}
            </span>
        </div>
    );
}