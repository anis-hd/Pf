import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faBrain, faVideo, faEye, faPersonWalking, faRotate, faChartLine, faMicroscope, faNetworkWired, faBolt, faComputer } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

import walkVideo from '../assets/walk.mp4';
import janeliaImg from '../assets/janelia.png';
import lifImg from '../assets/leakyintegrateandfire.png';

export default function FruitFlyPage() {
    useEffect(() => {
        document.title = 'Fruit-Fly Connectome Digital Twin - Anis Houidi';
        window.scrollTo(0, 0);
    }, []);

    const CodeBlock = ({ children }) => (
        <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 overflow-x-auto text-sm font-mono text-green-400">
            <code>{children}</code>
        </pre>
    );

    const FeatureCard = ({ icon, title, description, gradient }) => (
        <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md hover:border-slate-300 transition-all duration-300 group">
            <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                <FontAwesomeIcon icon={icon} className="text-2xl text-white" />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-slate-900">{title}</h3>
            <p className="text-slate-600 text-sm leading-relaxed">{description}</p>
        </div>
    );

    const CollapsibleSection = ({ title, children }) => {
        return (
            <div className="border border-slate-200 rounded-xl overflow-hidden mb-4 bg-white shadow-sm">
                <div className="w-full px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-left text-slate-900">
                    <span className="font-semibold text-lg">{title}</span>
                </div>
                <div className="p-6 bg-white text-slate-700">
                    {children}
                </div>
            </div>
        );
    };

    const SenseRow = ({ name, description }) => (
        <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 py-2 border-b border-slate-100 last:border-0">
            <span className="sm:w-44 flex-shrink-0 font-mono text-sm text-blue-600 font-medium">{name}</span>
            <span className="text-slate-600 text-sm leading-relaxed">{description}</span>
        </div>
    );

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900">
            {/* Navigation */}
            <nav className="fixed top-0 left-0 right-0 z-50 px-6 lg:px-20 xl:px-36 py-6 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm text-slate-900">
                <div className="flex items-center justify-between max-w-7xl mx-auto">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-3 text-slate-500 hover:text-slate-900 transition-all duration-300 group"
                    >
                        <FontAwesomeIcon
                            icon={faArrowLeft}
                            className="group-hover:-translate-x-1 transition-transform duration-300"
                        />
                        <span className="font-medium">Back to Portfolio</span>
                    </Link>
                    <a
                        href="https://github.com/anis-hd/fruit-fly-connectome-digital-twin"
                        target="_blank"
                        rel="noreferrer"
                        className="text-slate-500 hover:text-slate-900 transition-colors"
                    >
                        <FontAwesomeIcon icon={faGithub} size="xl" />
                    </a>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="px-6 lg:px-20 xl:px-36 pt-32 pb-16">
                <div className="max-w-7xl mx-auto">
                    {/* Project Badges */}
                    <div className="flex flex-wrap gap-3 mb-6">
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-full text-sm font-medium text-emerald-700">
                            <FontAwesomeIcon icon={faMicroscope} className="text-emerald-600" />
                            Spiking Digital Twin
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-purple-50 border border-purple-200 rounded-full text-sm font-medium text-purple-700">
                            <FontAwesomeIcon icon={faNetworkWired} className="text-purple-600" />
                            Male CNS v1.0 Connectome
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full text-sm font-medium text-blue-700">
                            <FontAwesomeIcon icon={faRotate} className="text-blue-600" />
                            Closed-Loop Brain + Body
                        </span>
                    </div>

                    {/* Title */}
                    <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4 text-slate-900">
                        Fruit-Fly Connectome
                    </h1>
                    <h2 className="text-2xl md:text-3xl font-semibold text-slate-700 mb-6">
                        Real-Time Spiking Digital Twin
                    </h2>

                    {/* Subtitle */}
                    <p className="text-lg md:text-xl text-slate-600 max-w-4xl mb-8 leading-relaxed">
                        A real-time spiking twin of the adult male fruit-fly central nervous system built directly on the{' '}
                        <span className="text-purple-600 font-semibold">Male CNS v1.0 wiring diagram (~211k neurons, ~152M synapses)</span>.
                        Every neuron is a spiking unit, <span className="text-emerald-600 font-semibold">no trained controller</span>; behavior
                        emerges from real synaptic weights plus <span className="text-blue-600 font-semibold">live sensory drive</span> into a
                        physically simulated walking body.
                    </p>

                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-4">
                        <a
                            href="#demo"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 shadow-sm transition-all duration-300 hover:scale-105"
                        >
                            <FontAwesomeIcon icon={faVideo} />
                            View Demo
                        </a>
                        <a
                            href="#how-it-works"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faBrain} />
                            How It Works
                        </a>
                        <a
                            href="https://github.com/anis-hd/fruit-fly-connectome-digital-twin"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            Source Code
                        </a>
                        <a
                            href="#simulation"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faBolt} />
                            Simulation Loop
                        </a>
                    </div>
                </div>
            </section>

            {/* Video Demo Section */}
            <section id="demo" className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-slate-900 text-center">Demo Video</h2>
                    <p className="text-center text-slate-500 mb-8">
                        Live follow-camera of the physics fly, where every step is steered by mean spike rates in the leg and descending motor pools
                    </p>
                    <div className="relative max-w-4xl mx-auto">
                        <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-white shadow-lg">
                            <video className="w-full h-auto" controls autoPlay loop muted playsInline>
                                <source src={walkVideo} type="video/mp4" />
                            </video>
                        </div>
                    </div>
                </div>
            </section>

            {/* Context Section */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-slate-900">Context</h2>
                    <div className="prose prose-slate max-w-none">
                        <p className="text-slate-600 leading-relaxed text-lg">
                            This project is a <strong className="text-slate-900">real-time spiking digital twin of the adult male fruit-fly central
                            nervous system</strong>. The anatomical wiring comes from the Male CNS v1.0 connectome, and every neuron is simulated
                            as a spiking unit. There is no trained controller, behavior emerges from real synaptic weights plus live sensory drive.
                        </p>
                        <p className="text-slate-600 leading-relaxed text-lg mt-4">
                            The <strong className="text-purple-600 font-semibold">brain</strong> is a custom spiking network built directly on the
                            connectome. The <strong className="text-emerald-600 font-semibold">body and environment</strong> are a physically
                            simulated fly that provides rich touch, smell, taste, visual, wind, temperature, humidity and proprioception senses,
                            and legs that walk.
                        </p>
                    </div>

                    {/* Dataset used */}
                    <div className="mt-8 flex flex-col sm:flex-row items-center gap-6 p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                        <div className="flex-shrink-0 w-full sm:w-80 lg:w-96 rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
                            <img src={janeliaImg} alt="Janelia Male CNS v1.0 dataset" className="w-full h-auto object-contain" loading="lazy" />
                        </div>
                        <div>
                            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Dataset used</p>
                            <h3 className="text-xl font-semibold text-slate-900 mb-2">Male CNS v1.0 : Janelia</h3>
                            <p className="text-slate-600 text-sm leading-relaxed">
                                Anatomical wiring (~211k annotated neurons, ~152M synapses) from the Janelia Male CNS v1.0
                                connectome. Every node and synaptic weight in the spiking network comes from this dataset.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Key Features */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center text-slate-900">Key Features</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <FeatureCard
                            icon={faNetworkWired}
                            title="Real Connectome Graph"
                            description="Tens of thousands of connected neurons, millions of strong edges after filtering weak, self and unannotated connections. Signed by predicted neurotransmitter and input-normalized per neuron."
                            gradient="from-purple-500 to-pink-500"
                        />
                        <FeatureCard
                            icon={faBolt}
                            title="Leaky Integrate-and-Fire Brain"
                            description="1 ms updates, ~10 ms membrane time constant, threshold + reset + 2 ms refractory period, background noise. One large sparse matrix multiply per step, on GPU if available."
                            gradient="from-yellow-500 to-orange-500"
                        />
                        <FeatureCard
                            icon={faEye}
                            title="12-Channel Sensory Drive"
                            description="Smell, stereo left/right smell, touch, taste, wind/speed, light, warm, cool, left/right proprioception and humidity, each Poisson-driving only its matching biological subset."
                            gradient="from-blue-500 to-cyan-500"
                        />
                        <FeatureCard
                            icon={faPersonWalking}
                            title="Tripod Walking Body"
                            description="42 active leg joint drives plus switchable foot adhesion. Six anti-phase tripod oscillators mapped through preprogrammed leg kinematics. Brain sets only left/right amplitude, turn and speed."
                            gradient="from-green-500 to-emerald-500"
                        />
                        <FeatureCard
                            icon={faRotate}
                            title="Closed-Loop Simulation"
                            description="Typically 1200 x 1 ms brain steps per run with scheduled stimulus windows, manual sensory / whole-brain bursts, gait gears, auto-settle, respawn, and runaway seizure detection."
                            gradient="from-indigo-500 to-purple-500"
                        />
                        <FeatureCard
                            icon={faChartLine}
                            title="Live Visualization"
                            description="Every flash is a real spike: 3D brain at real cell-body locations, scrolling raster in sensory | interneuron | motor order, rate plots, pool bars, body telemetry and arena position."
                            gradient="from-pink-500 to-orange-500"
                        />
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section id="how-it-works" className="px-6 lg:px-20 xl:px-36 py-16 bg-slate-100/60 border-y border-slate-200/80">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center text-slate-900">How It Works</h2>
                    <p className="text-slate-500 text-center mb-12">
                        Brain network, body senses, actuators and simulation loop, all driven by real spikes.
                    </p>

                    <div className="space-y-4">
                        <CollapsibleSection title="🧠 1. Brain : connectome graph">
                            <p className="text-slate-600 leading-relaxed mb-4">
                                Each node is a neuron, each edge is a synaptic connection weighted by synapse count. Weak, self and
                                unannotated connections are filtered out, leaving a large sparse directed graph of tens of thousands of
                                connected neurons.
                            </p>
                            <ul className="list-disc list-inside space-y-2 text-slate-600 text-sm leading-relaxed">
                                <li><strong className="text-slate-900">Signed by neurotransmitter:</strong> acetylcholine-like excites (+1); GABA / glutamate / histamine-like inhibits (−1); serotonin, dopamine, octopamine treated as weakly excitatory; missing predictions default to excitatory.</li>
                                <li><strong className="text-slate-900">Input-normalized:</strong> each neuron&apos;s total input strength is scaled to a common gain so highly connected neurons don&apos;t dominate, allowing stimulus responses to stand out and activity to stay stable.</li>
                                <li><strong className="text-slate-900">Functional zones:</strong> sensory and motor neurons identified from cell type, nerve, brain region and receptor labels (olfactory, visual, gustatory, mechanosensory, chordotonal / Johnston&apos;s organ, thermo, hygro, ascending as sensory; motor as motor), with graph in/out-degree as fallback.</li>
                            </ul>
                        </CollapsibleSection>

                        <CollapsibleSection title="⚡ 2. Brain : neuron dynamics">
                            <p className="text-slate-600 leading-relaxed mb-4">
                                Each neuron is a leaky integrate-and-fire unit updated every 1 ms: voltage decays slightly, adds input
                                current, spikes on threshold crossing (unless refractory), then resets to zero with a 2 ms refractory period.
                            </p>
                            <CodeBlock>{`input current = sum(spiking presynaptic * signed normalized weight)
             + external sensory / stimulus current
             + noise
voltage *= decay (tau ~10 ms); voltage += input current
if voltage >= threshold and not refractory:
    spike; voltage = 0; refractory = 2 ms`}</CodeBlock>
                            <div className="my-6 flex flex-col items-center">
                                <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-50 shadow-inner max-w-2xl w-full">
                                    <img src={lifImg} alt="Leaky integrate-and-fire neuron dynamics" className="w-full h-auto object-contain" loading="lazy" />
                                </div>
                                <p className="text-center text-slate-500 text-sm mt-2 italic">Leaky integrate-and-fire dynamics: decay, integrate, spike, reset</p>
                            </div>
                            <p className="text-slate-600 leading-relaxed mt-4 text-sm">
                                Spikes are binary events. The whole-brain state is just voltages plus a spike vector, computed as one large
                                sparse matrix multiply per step, on GPU if available.
                            </p>
                        </CollapsibleSection>

                        <CollapsibleSection title="👁️ 3. Body : sensors (12 channels)">
                            <p className="text-slate-500 mb-4 text-sm">
                                The physics fly walks on flat or blocky terrain in an arena with a food berry, a warm spot, a cool spot and a
                                humid spot. Each step reports 0..1 levels, converted back into Poisson drive into only the matching anatomical subset.
                            </p>
                            <div>
                                <SenseRow name="smell" description="Falls with distance to food, 1 when on top of it." />
                                <SenseRow name="left / right smell" description="Smell split by whether food is left or right of heading: stereo olfaction for turning toward food." />
                                <SenseRow name="touch" description="Ground contact force on feet, reported as total plus left and right maxima." />
                                <SenseRow name="taste" description="1 within eating distance of the berry, else 0. Eating fills satiety, then the berry respawns elsewhere." />
                                <SenseRow name="speed / wind" description="Thorax speed from frame-to-frame displacement: walking speed and wind / airflow sense." />
                                <SenseRow name="light" description="Baseline plus speed-dependent optic-flow proxy." />
                                <SenseRow name="warm / cool" description="Virtual temperature field: ambient 25°C with warm Gaussian bump (~33°C) and cool dip (~17°C), split into separate channels." />
                                <SenseRow name="humid" description="Virtual humidity field with moist peak decaying to ambient dry." />
                                <SenseRow name="proprio L / R" description="Leg joint angles and velocities vs neutral stance, averaged separately for left and right legs." />
                            </div>
                        </CollapsibleSection>

                        <CollapsibleSection title="🦵 4. Body : actuators (walking)">
                            <p className="text-slate-600 leading-relaxed mb-4">
                                The fly has 42 active leg joint drives plus switchable foot adhesion. Walking comes from a tripod central pattern
                                generator: six oscillators (LF, LM, LH, RF, RM, RH) in anti-phase tripod pattern, mapped through preprogrammed
                                leg kinematics with adhesion timed to stance phase.
                            </p>
                            <ul className="list-disc list-inside space-y-2 text-slate-600 text-sm leading-relaxed">
                                <li>The brain does not set joints directly, it sets <strong className="text-slate-900">left amplitude, right amplitude, turn and speed</strong>.</li>
                                <li>Tiny motor spike fractions (~0..0.03) are expanded by a recruitment curve to 0..1 amplitudes; left follows left descending / leg / motor activity, right mirrors it; speed follows overall descending / motor activity; turn follows left-right difference with a deadband.</li>
                                <li>All drives are smoothed so gait doesn&apos;t jitter. Speed selects a gait gear: slow amble ~5 Hz, walk ~8 Hz, fast stride ~12 Hz, with hysteresis.</li>
                                <li>Body auto-settles to stance at start, stays quiet briefly, and respawns if tipped over or invalid.</li>
                            </ul>
                        </CollapsibleSection>

                        <CollapsibleSection title="📐 5. Anatomical pools">
                            <p className="text-slate-600 leading-relaxed mb-4 text-sm">
                                Mean spike rate in each pool is what steers the body. A monitored subset of ~1200 neurons (all sensory + all
                                motor + sampled interneurons) is tracked live, grouped as sensory | interneurons | motor.
                            </p>
                            <div className="grid md:grid-cols-2 gap-4 text-sm">
                                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                                    <h4 className="font-semibold text-slate-900 mb-2">Sensory channels (12)</h4>
                                    <p className="text-slate-600 leading-relaxed">smell, left, right, touch, taste, wind, light, warm, cool, left proprioception, right proprioception, humidity.</p>
                                </div>
                                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                                    <h4 className="font-semibold text-slate-900 mb-2">Motor pools</h4>
                                    <p className="text-slate-600 leading-relaxed">leg motor by segment T1/T2/T3 and left/right, aggregated left-leg / right-leg, descending whole + left/right, abdominal, head, and whole-motor left/right halves.</p>
                                </div>
                            </div>
                        </CollapsibleSection>
                    </div>
                </div>
            </section>

            {/* Simulation Loop */}
            <section id="simulation" className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-center text-slate-900">Simulation Loop</h2>
                    <p className="text-slate-500 text-center mb-8">Time runs in 1 ms brain steps, typically 1200 steps per run. Each step, in order:</p>
                    <div className="space-y-4">
                        {[
                            { n: '1', title: 'Build external current', desc: 'Scheduled stimulus windows (Poisson bursts into a random fraction of sensory neurons), plus manual sensory-burst / whole-brain-burst presses, plus 12-channel body-sense drive.' },
                            { n: '2', title: 'Step all LIF neurons once', desc: 'Sparse network current + external current + noise → spikes.' },
                            { n: '3', title: 'Measure mean spike rates', desc: 'Whole population, whole motor pool, and each leg / descending / abdominal / head pool.' },
                            { n: '4', title: 'Drive the body', desc: 'Convert motor rates to left/right/speed/turn, step pattern generator + physics, read back new senses for the next step.' },
                            { n: '5', title: 'Sample + append histories', desc: 'Every N steps, sample monitored neurons into scrolling histories for raster and rate plots.' },
                        ].map((s) => (
                            <div key={s.n} className="flex gap-4 p-5 bg-white border border-slate-200 rounded-2xl shadow-sm">
                                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-blue-600 text-white font-bold flex items-center justify-center">{s.n}</div>
                                <div>
                                    <h3 className="font-semibold text-slate-900 mb-1">{s.title}</h3>
                                    <p className="text-slate-600 text-sm leading-relaxed">{s.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-8 p-6 bg-amber-50 border border-amber-200 rounded-2xl text-sm text-slate-700 leading-relaxed">
                        <strong className="text-slate-900">Runaway + effect quantification: </strong>
                        if mean activity exceeds ~50% the run is flagged as seizure-like. At run end, baseline motor rate before
                        stimulation is compared to post-stimulation rate to quantify whether the stimulus drove movement
                        (e.g. several-fold increase). An offline mode runs the same brain without the body on fixed stimulus windows
                        and saves a raster image + spike archive.
                    </div>
                </div>
            </section>

            {/* Visualization */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-slate-100/60 border-y border-slate-200/80">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center text-slate-900">Visualization</h2>
                    <p className="text-slate-500 text-center mb-12">Every flash is a real simulated spike from the monitored set.</p>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                            <FontAwesomeIcon icon={faBrain} className="text-2xl text-purple-600 mb-3" />
                            <h3 className="font-semibold text-lg mb-2 text-slate-900">3D Brain</h3>
                            <p className="text-slate-600 text-sm leading-relaxed">One point per neuron at its real cell-body location, colored by flash on spike and tinted by cell class. Background neurons give anatomical context. Rotatable, with top-down projection view.</p>
                        </div>
                        <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                            <FontAwesomeIcon icon={faChartLine} className="text-2xl text-pink-600 mb-3" />
                            <h3 className="font-semibold text-lg mb-2 text-slate-900">Spike Raster</h3>
                            <p className="text-slate-600 text-sm leading-relaxed">Rows are monitored neurons in sensory | interneuron | motor order with divider lines, columns are time. Stimulus windows highlighted so sensory-driven bands and following motor response are visible.</p>
                        </div>
                        <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                            <FontAwesomeIcon icon={faComputer} className="text-2xl text-blue-600 mb-3" />
                            <h3 className="font-semibold text-lg mb-2 text-slate-900">Rate Plots</h3>
                            <p className="text-slate-600 text-sm leading-relaxed">Scrolling whole-population and motor-pool rates, plus per-pool bars for left leg, right leg, descending command and leg segments T1/T2/T3. Live tuning of threshold, tau, noise, gain and stimulus strength.</p>
                        </div>
                        <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                            <FontAwesomeIcon icon={faVideo} className="text-2xl text-emerald-600 mb-3" />
                            <h3 className="font-semibold text-lg mb-2 text-slate-900">Body View</h3>
                            <p className="text-slate-600 text-sm leading-relaxed">Live follow-camera video of the physics fly plus telemetry: distance to food, amount eaten, touch, speed, temperature, humidity, proprioception, drive amplitudes and arena position. Controls: run, pause, reset, stimulate sensory / whole brain, drop food, reset body.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Technologies */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center text-slate-900">Technologies Used</h2>
                    <div className="flex flex-wrap justify-center gap-4">
                        {['Python', 'Spiking Neural Networks', 'Connectome Graph', 'LIF Neurons', 'Sparse Computation', 'GPU Acceleration', 'Physics Simulation', 'Poisson Stimulation', 'Real-time Visualization'].map((tech, index) => (
                            <span
                                key={index}
                                className="px-6 py-3 bg-slate-100 border border-slate-200 rounded-full font-medium hover:border-blue-500 hover:bg-slate-200 text-slate-700 transition-all duration-300 cursor-default"
                            >
                                {tech}
                            </span>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 mb-8">
                <div className="max-w-4xl mx-auto text-center p-12 bg-emerald-50/50 border border-emerald-100 rounded-3xl">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">
                        Emergent Behavior, No Controller
                    </h2>
                    <p className="text-slate-600 mb-8 max-w-2xl mx-auto">
                        Movement here is not learned or scripted; it is read out from real connectome dynamics driven by the world.
                        Stimulate the senses and watch the motor pools respond.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <a
                            href="#demo"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-emerald-600 text-white rounded-lg font-semibold hover:bg-emerald-700 shadow-sm transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faVideo} />
                            Watch It Walk
                        </a>
                        <a
                            href="https://github.com/anis-hd/fruit-fly-connectome-digital-twin"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            Source Code
                        </a>
                        <Link
                            to="/"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 shadow-sm transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faArrowLeft} />
                            Back to Portfolio
                        </Link>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="px-6 lg:px-20 xl:px-36 py-8 border-t border-slate-200">
                <div className="max-w-6xl mx-auto text-center text-slate-500 text-sm">
                    © {new Date().getFullYear()} Anis Ben Houidi. All rights reserved.
                </div>
            </footer>
        </div>
    );
}
