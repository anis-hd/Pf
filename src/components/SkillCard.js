import React from 'react';

export default function SkillCard({ img, name }) {
    return (
        <div className="skill-card-container group">
            <img
                src={img}
                className="skill-logo bg-white rounded-full p-2"
                alt={name || "skill logo"}
            />

            {/* Skill name tooltip */}
            {name && (
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-slate-900 rounded text-xs font-medium whitespace-nowrap opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 z-50 pointer-events-none">
                    <span className="text-white">
                        {name}
                    </span>
                </div>
            )}
        </div>
    );
}