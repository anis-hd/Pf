import React, { useState, useRef, useEffect } from 'react';

export default function SkillCard({ img, name, mousePosition }) {
    const cardRef = useRef(null);
    const [style, setStyle] = useState({});
    const [isHovered, setIsHovered] = useState(false);

    useEffect(() => {
        if (!cardRef.current || mousePosition.x === null) return;

        const rect = cardRef.current.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        const distance = Math.sqrt(
            Math.pow(mousePosition.x - centerX, 2) +
            Math.pow(mousePosition.y - centerY, 2)
        );

        const maxDistance = 300;

        if (distance < maxDistance) {
            const scale = 1 + 0.15 * (1 - distance / maxDistance);
            const glowOpacity = 1 - (distance / maxDistance);
            const zIndex = 10;

            // Calculate Angle for conic gradient
            const x = mousePosition.x - centerX;
            const y = mousePosition.y - centerY;
            const angle = Math.atan2(y, x) * (180 / Math.PI);

            setStyle({
                '--glow-angle': `${angle}deg`,
                '--glow-opacity': glowOpacity,
                transform: `scale(${scale})`,
                zIndex: zIndex
            });
        } else {
            setStyle({
                '--glow-opacity': 0,
                transform: 'scale(1)',
                zIndex: 1
            });
        }
    }, [mousePosition]);

    return (
        <div
            className="skill-card-container group hover:!z-50"
            ref={cardRef}
            style={style}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <img
                src={img}
                className="skill-logo bg-white rounded-full p-2"
                alt={name || "skill logo"}
            />

            {/* Skill name tooltip */}
            {name && (
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-black/80 backdrop-blur-sm rounded text-xs font-medium whitespace-nowrap opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 z-50 pointer-events-none">
                    <span className="text-white">
                        {name}
                    </span>
                </div>
            )}
        </div>
    );
}