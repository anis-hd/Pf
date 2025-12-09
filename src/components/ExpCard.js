export default function ExpCard(props) {
    return (
        // The main changes are in this line:
        <div data-aos="fade-up" data-aos-duration="500" data-aos-offset="0" className="w-full bg-dark-100 py-4 px-4 flex items-start">

            {/* Image on the left */}
            <img src={props.img} className="w-20 max-h-20 mr-4 flex-shrink-0" alt={props.name} />

            {/* Text content */}
            <div className="flex flex-col">
                <h1 className="font-bold md:text-xl">{props.name}</h1>
                <p className="font-light md:text-lg">{props.issued}</p>
                <p className="font-light text-gray-400 mt-2">{props.desc}</p>

                {/* Keywords (your styling is preserved) */}
                {props.keywords && (
                    <p className="font-light text-gray-400 mt-2">
                        {props.keywords.map((keyword, index) => (
                            <span key={index} className="font-bold text-gray-300">
                                {keyword}
                                {index < props.keywords.length - 1 ? ', ' : ''}
                            </span>
                        ))}
                    </p>
                )}
            </div>
        </div>
    );
}