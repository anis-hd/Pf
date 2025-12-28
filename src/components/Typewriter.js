import { useState, useEffect } from 'react';

const Typewriter = ({ texts, delay, infinite }) => {
  const [currentText, setCurrentText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [textIndex, setTextIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  // Support both single text (string) and multiple texts (array)
  const textArray = Array.isArray(texts) ? texts : [texts];
  const currentFullText = textArray[textIndex];

  useEffect(() => {
    let timeout;

    if (!isDeleting) {
      // Typing
      if (currentIndex < currentFullText.length) {
        timeout = setTimeout(() => {
          setCurrentText(currentFullText.substring(0, currentIndex + 1));
          setCurrentIndex(prevIndex => prevIndex + 1);
        }, delay);
      } else {
        // Finished typing, wait then start deleting
        timeout = setTimeout(() => {
          setIsDeleting(true);
        }, 2000); // Pause at end of word
      }
    } else {
      // Deleting
      if (currentIndex > 0) {
        timeout = setTimeout(() => {
          setCurrentText(currentFullText.substring(0, currentIndex - 1));
          setCurrentIndex(prevIndex => prevIndex - 1);
        }, delay / 2); // Delete faster than typing
      } else {
        // Finished deleting, move to next text
        setIsDeleting(false);
        if (infinite || textIndex < textArray.length - 1) {
          setTextIndex((prevIndex) => (prevIndex + 1) % textArray.length);
        }
      }
    }

    return () => clearTimeout(timeout);
  }, [currentIndex, delay, infinite, isDeleting, currentFullText, textArray.length, textIndex]);

  return (
    <span>
      {currentText}
      <span className="animate-pulse">|</span>
    </span>
  );
};

export default Typewriter;