import React, { useState, useRef, useEffect } from 'react';

export default function SkillCard({ img, mousePosition }) {
    const cardRef = useRef(null);
    const [style, setStyle] = useState({});

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
            const scale = 1 + 0.2 * (1 - distance / maxDistance);
            const glowOpacity = 1 - (distance / maxDistance);
            const zIndex = 10;

            // --- NEW: Calculate Angle ---
            // Get mouse position relative to the center of the card
            const x = mousePosition.x - centerX;
            const y = mousePosition.y - centerY;
            // Calculate the angle using atan2 and convert to degrees
            const angle = Math.atan2(y, x) * (180 / Math.PI);
            // --- END NEW ---

            setStyle({
                // --- NEW: Pass angle to CSS ---
                '--glow-angle': `${angle}deg`,
                // --- END NEW ---
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
        <div className="skill-card-container" ref={cardRef} style={style}>
            <img src={img} className="skill-logo" alt="skill logo" />
        </div>
    );
}