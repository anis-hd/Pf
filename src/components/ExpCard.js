export default function ExpCard(props) {
    return (
        <div data-aos="fade-up" data-aos-duration="500" data-aos-offset="100" className="w-full md:w-2/6 bg-dark-100 rounded-md py-4 px-4 flex items-center">
            {/* Image on the left */}
            <img src={props.img} className="w-20 max-h-20 mr-4 rounded-md" alt={props.name} />
            
            {/* Text content */}
            <div className="flex flex-col mt-2">
                <h1 className="font-bold md:text-xl">{props.name}</h1>
                <p className="font-light md:text-lg">{props.issued}</p>
                <p className="font-light text-gray-400">{props.desc}</p>
                
                {/* Keywords */}
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
