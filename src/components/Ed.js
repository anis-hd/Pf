import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCalendar } from '@fortawesome/free-solid-svg-icons';

export default function Education() {
    const educationData = [
        {
            school: "National School of Computer Science",
            degree: "Engineer's Degree, Computer Science - AI & Decision Systems",
            description: "Computer science, software engineering and applied mathematics.",
            year: "2021 - 2025"
        },
        {
            school: "Preparatory Institute for Engineering Studies of Nabeul",
            degree: "Mathematics, Physics & Computer Science",
            description: "Two years of intensive studies in Mathematics, Physics and Industrial Sciences for the national engineering contest.",
            year: "2019 - 2021"
        }
    ];

    return (
        <section id="honors" className="py-20 text-slate-900 relative">
            <div className="relative z-10">
                {/* Section Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <h2 className="text-4xl md:text-5xl font-bold text-slate-900">
                            Education
                        </h2>
                        <div className="hidden md:block h-1 w-24 bg-blue-600 rounded-full" />
                    </div>
                </div>

                {/* Content */}
                <div className="opacity-100 transition-all duration-500 ease-in-out">
                    <p className="text-slate-500 text-lg mb-10">
                        My academic journey
                    </p>

                    {/* Timeline */}
                    <div className="relative">
                        {/* Timeline Line */}
                        <div className="absolute left-8 md:left-1/2 top-0 bottom-0 w-px bg-slate-200 hidden md:block" />

                        {/* Education Cards */}
                        <div className="space-y-8">
                            {educationData.map((edu, index) => (
                                <div
                                    key={index}
                                    className={`relative flex flex-col md:flex-row gap-8 ${index % 2 === 0 ? 'md:flex-row-reverse' : ''}`}
                                >
                                    {/* Timeline Dot */}
                                    <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-blue-600 z-10" />

                                    {/* Content */}
                                    <div className={`flex-1 ${index % 2 === 0 ? 'md:text-right md:pr-12' : 'md:pl-12'}`}>
                                        <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm hover:border-blue-500/30 hover:shadow-md transition-all duration-300 group">
                                            {/* Year Badge */}
                                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium mb-4">
                                                <FontAwesomeIcon icon={faCalendar} className="text-xs" />
                                                {edu.year}
                                            </div>

                                            {/* School Name */}
                                            <h3 className="text-xl md:text-2xl font-bold mb-2 text-slate-900 group-hover:text-blue-600 transition-colors">
                                                {edu.school}
                                            </h3>

                                            {/* Degree */}
                                            <p className="text-lg font-semibold text-blue-600 mb-3">
                                                {edu.degree}
                                            </p>

                                            {/* Description */}
                                            <p className="text-slate-600 leading-relaxed">
                                                {edu.description}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Spacer for timeline alignment */}
                                    <div className="hidden md:block flex-1" />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}