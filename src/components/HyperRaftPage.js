import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faRocket, faCube, faCode, faVideo, faBrain, faLayerGroup, faChartLine, faGears, faTerminal, faChevronDown, faChevronUp, faGraduationCap, faBuilding } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import AOS from 'aos';
import hyperraftVideo from '../assets/hyperraft.mp4';

// Import all images
import rdvcEncoding from '../assets/img/RDVCencoding.png';
import rdvcDecoding from '../assets/img/rdvcdecoding.png';
import encodingProcess from '../assets/img/sequenceuml encoding process.png';
import decodingProcess from '../assets/img/sequenceuml decoding process.png';
import raftArchitecture from '../assets/img/raftarchitecture.png';
import hyperpriorGraph from '../assets/img/hyperpriorcomponent graph.png';
import raftTraining from '../assets/img/rafttraining.png';
import raftPrediction from '../assets/img/raftprediction.png';
import rdCurvePSNR from '../assets/img/rd_curve_psnr.png';
import rdCurveMSSSIM from '../assets/img/rd_curve_msssim.png';
import psnrResolution from '../assets/img/psnr_by_resolution.png';
import msssimResolution from '../assets/img/msssim_by_resolution.png';
import temporalFiltering from '../assets/img/first order IIR temporal filtering.png';
import realVsReconstructed from '../assets/img/realvsreconstructed.png';

export default function HyperRaftPage() {
    const [expandedSections, setExpandedSections] = useState({});

    useEffect(() => {
        document.title = 'RDVC: Raft Deep Video Compression - Anis Houidi';
        AOS.init({ duration: 800 });
        window.scrollTo(0, 0);
    }, []);

    const toggleSection = (section) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }));
    };

    const CodeBlock = ({ children }) => (
        <pre className="bg-black/60 border border-white/10 rounded-lg p-4 overflow-x-auto text-sm font-mono text-green-400">
            <code>{children}</code>
        </pre>
    );

    const ImageWithCaption = ({ src, alt, caption }) => (
        <div className="my-6" data-aos="fade-up">
            <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/20">
                <img src={src} alt={alt} className="w-full h-auto" loading="lazy" />
            </div>
            {caption && <p className="text-center text-gray-500 text-sm mt-2 italic">{caption}</p>}
        </div>
    );

    const ImageGrid = ({ images }) => (
        <div className="grid md:grid-cols-2 gap-6 my-6" data-aos="fade-up">
            {images.map((img, idx) => (
                <div key={idx} className="text-center">
                    <div className="rounded-xl overflow-hidden border border-white/10 bg-black/20">
                        <img src={img.src} alt={img.alt} className="w-full h-auto" loading="lazy" />
                    </div>
                    <p className="text-gray-400 text-sm mt-2">{img.caption}</p>
                </div>
            ))}
        </div>
    );

    const FeatureCard = ({ icon, title, description, gradient }) => (
        <div
            className="p-6 bg-white/5 border border-white/10 rounded-2xl hover:bg-white/10 transition-all duration-300 group"
            data-aos="fade-up"
        >
            <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                <FontAwesomeIcon icon={icon} className="text-2xl" />
            </div>
            <h3 className="text-xl font-semibold mb-2">{title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
        </div>
    );

    const CollapsibleSection = ({ id, title, children, defaultOpen = false }) => {
        const isOpen = expandedSections[id] ?? defaultOpen;
        return (
            <div className="border border-white/10 rounded-xl overflow-hidden mb-4">
                <button
                    onClick={() => toggleSection(id)}
                    className="w-full px-6 py-4 bg-white/5 hover:bg-white/10 transition-colors flex items-center justify-between text-left"
                >
                    <span className="font-semibold text-lg">{title}</span>
                    <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} className="text-gray-400" />
                </button>
                {isOpen && (
                    <div className="p-6 bg-black/20">
                        {children}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-dark-500 to-dark-600 text-white">
            {/* Navigation */}
            <nav className="fixed top-0 left-0 right-0 z-50 px-6 lg:px-20 xl:px-36 py-6 bg-dark-500/80 backdrop-blur-md border-b border-white/10">
                <div className="flex items-center justify-between max-w-7xl mx-auto">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-3 text-gray-400 hover:text-white transition-all duration-300 group"
                    >
                        <FontAwesomeIcon
                            icon={faArrowLeft}
                            className="group-hover:-translate-x-1 transition-transform duration-300"
                        />
                        <span className="font-medium">Back to Portfolio</span>
                    </Link>
                    <a
                        href="https://github.com/anis-hd/PFE-RAFT-and-hyperprior-based-learned-video-compression"
                        target="_blank"
                        rel="noreferrer"
                        className="text-gray-400 hover:text-white transition-colors"
                    >
                        <FontAwesomeIcon icon={faGithub} size="xl" />
                    </a>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="px-6 lg:px-20 xl:px-36 pt-32 pb-16">
                <div className="max-w-7xl mx-auto">
                    {/* Project Badge */}
                    <div className="flex flex-wrap gap-3 mb-6" data-aos="fade-up">
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-full text-sm font-medium">
                            <FontAwesomeIcon icon={faRocket} className="text-purple-400" />
                            Final Year Project
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border border-blue-500/30 rounded-full text-sm font-medium">
                            <FontAwesomeIcon icon={faGraduationCap} className="text-blue-400" />
                            ENSI
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600/20 to-emerald-600/20 border border-green-500/30 rounded-full text-sm font-medium">
                            <FontAwesomeIcon icon={faBuilding} className="text-green-400" />
                            Talan Tunisie
                        </span>
                    </div>

                    {/* Title */}
                    <h1
                        className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-500 to-blue-400 bg-clip-text text-transparent"
                        data-aos="fade-up"
                        data-aos-delay="100"
                    >
                        RDVC
                    </h1>
                    <h2
                        className="text-2xl md:text-3xl font-semibold text-gray-300 mb-6"
                        data-aos="fade-up"
                        data-aos-delay="150"
                    >
                        Raft Deep Video Compression
                    </h2>

                    {/* Subtitle */}
                    <p
                        className="text-lg md:text-xl text-gray-400 max-w-4xl mb-8 leading-relaxed"
                        data-aos="fade-up"
                        data-aos-delay="200"
                    >
                        A Hybrid Video Compression Framework combining <span className="text-purple-400 font-medium">RAFT Optical Flow</span>,
                        <span className="text-pink-400 font-medium"> Hyperprior Entropy Coding</span>, and
                        <span className="text-blue-400 font-medium"> Quantum-Inspired I-Frame Encoding</span>.
                    </p>

                    {/* Action Buttons */}
                    <div
                        className="flex flex-wrap gap-4"
                        data-aos="fade-up"
                        data-aos-delay="300"
                    >
                        <a
                            href="#demo"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all duration-300 hover:scale-105"
                        >
                            <FontAwesomeIcon icon={faVideo} />
                            View Demo
                        </a>
                        <a
                            href="https://github.com/anis-hd/PFE-RAFT-and-hyperprior-based-learned-video-compression"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-white/10 border border-white/20 rounded-lg font-semibold hover:bg-white/20 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            Source Code
                        </a>
                        <a
                            href="#usage"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-white/10 border border-white/20 rounded-lg font-semibold hover:bg-white/20 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faTerminal} />
                            Quick Start
                        </a>
                    </div>
                </div>
            </section>

            {/* Video Demo Section */}
            <section id="demo" className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-center" data-aos="fade-up">Demo Video</h2>
                    <p className="text-center text-gray-400 mb-8" data-aos="fade-up">
                        80% of the frames in this video are reconstructed during decoding
                    </p>
                    <div className="relative" data-aos="zoom-in">
                        <div className="absolute -inset-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-3xl blur-2xl opacity-30"></div>
                        <div className="relative rounded-2xl overflow-hidden border border-white/20 shadow-2xl">
                            <video className="w-full h-auto" controls autoPlay loop muted playsInline>
                                <source src={hyperraftVideo} type="video/mp4" />
                            </video>
                        </div>
                    </div>
                </div>
            </section>

            {/* Context Section */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8" data-aos="fade-up">Context</h2>
                    <div className="prose prose-invert max-w-none" data-aos="fade-up">
                        <p className="text-gray-300 leading-relaxed text-lg">
                            This project implements <strong className="text-white">RDVC (Raft Deep Video Compression)</strong>, a Final Year Project developed at <strong className="text-green-400">Talan Tunisie</strong> in collaboration with the <strong className="text-blue-400">National School of Computer Science (ENSI)</strong>.
                        </p>
                        <p className="text-gray-300 leading-relaxed text-lg mt-4">
                            The system addresses the limitations of traditional codecs by proposing a deep learning-based inter-frame compression pipeline. It integrates <strong className="text-purple-400">RAFT</strong> (Recurrent All-Pairs Field Transforms) for optical flow estimation and <strong className="text-pink-400">Hyperprior Autoencoders</strong> for entropy coding. Additionally, the project explores <strong className="text-cyan-400">Quantum Computing</strong> simulations for I-frame compression.
                        </p>
                    </div>
                </div>
            </section>

            {/* Key Features */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center" data-aos="fade-up">Key Features</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <FeatureCard
                            icon={faBrain}
                            title="Learned Optical Flow (RAFT)"
                            description="Uses the RAFT architecture trained on MPI Sintel to estimate dense motion between frames, enabling effective motion compensation."
                            gradient="from-purple-500 to-pink-500"
                        />
                        <FeatureCard
                            icon={faCube}
                            title="Motion Compensation Network"
                            description="Refines warped frames to correct artifacts derived from occlusion or flow errors."
                            gradient="from-blue-500 to-cyan-500"
                        />
                        <FeatureCard
                            icon={faLayerGroup}
                            title="End-to-End Compression"
                            description="Motion Autoencoder compresses optical flow maps, while Residual Autoencoder handles the difference between predicted and actual frames."
                            gradient="from-pink-500 to-orange-500"
                        />
                        <FeatureCard
                            icon={faChartLine}
                            title="Multi-Phase Training"
                            description="A 3-phase strategy to stabilize convergence: GT flow training, end-to-end training, and perceptual fine-tuning with MS-SSIM."
                            gradient="from-green-500 to-emerald-500"
                        />
                        <FeatureCard
                            icon={faGears}
                            title="Post-Processing"
                            description="Includes Low Motion Region replacement, Histogram Matching, and Temporal IIR Filtering to compensate for compression artifacts."
                            gradient="from-yellow-500 to-orange-500"
                        />
                        <FeatureCard
                            icon={faCode}
                            title="Custom File Format"
                            description="Encodes videos into .rdvc (RAFT Deep Video Compression) files with efficient bitstream structure."
                            gradient="from-indigo-500 to-purple-500"
                        />
                    </div>
                </div>
            </section>

            {/* System Architecture */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-black/20">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center" data-aos="fade-up">System Architecture & Workflows</h2>
                    <p className="text-gray-400 text-center mb-12" data-aos="fade-up">
                        The RDVC codec operates on a P-frame architecture involving Motion and Residual branches.
                    </p>

                    <div className="space-y-8">
                        {/* Encoding/Decoding Block Diagrams */}
                        <div>
                            <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">High-Level Schematics</h3>
                            <ImageWithCaption
                                src={rdvcEncoding}
                                alt="RDVC Encoding Block Diagram"
                                caption="Detailed Encoding Block Diagram"
                            />
                            <ImageWithCaption
                                src={rdvcDecoding}
                                alt="RDVC Decoding Block Diagram"
                                caption="Detailed Decoding Block Diagram"
                            />
                        </div>

                        {/* Workflow Diagrams */}
                        <div>
                            <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Encoding & Decoding Processes</h3>
                            <p className="text-gray-400 mb-6" data-aos="fade-up">
                                The system uses a synchronized loop to ensure the encoder and decoder states remain identical.
                            </p>
                            <ImageGrid images={[
                                { src: encodingProcess, alt: "Video Encoding Workflow", caption: "Video Encoding Workflow" },
                                { src: decodingProcess, alt: "Video Decoding Workflow", caption: "Video Decoding Workflow" }
                            ]} />
                        </div>

                        {/* Core Components */}
                        <div>
                            <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Core Components</h3>
                            <ImageGrid images={[
                                { src: raftArchitecture, alt: "RAFT Architecture", caption: "RAFT Optical Flow Architecture" },
                                { src: hyperpriorGraph, alt: "Hyperprior Graph", caption: "Hyperprior Entropy Model" }
                            ]} />
                        </div>
                    </div>
                </div>
            </section>

            {/* Training & Optimization */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center" data-aos="fade-up">Training & Optimization</h2>

                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Training Dynamics</h3>
                        <ImageWithCaption
                            src={raftTraining}
                            alt="RAFT Training"
                            caption="RAFT Training Process"
                        />
                    </div>

                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Multi-Phase Training Strategy</h3>
                        <div className="grid md:grid-cols-3 gap-6" data-aos="fade-up">
                            <div className="p-6 bg-white/5 border border-purple-500/30 rounded-2xl">
                                <div className="text-4xl font-bold text-purple-400 mb-2">1</div>
                                <h4 className="font-semibold mb-2">Phase 1: Initialization</h4>
                                <p className="text-gray-400 text-sm">Train Residual AE using Ground Truth flow with MSE loss.</p>
                            </div>
                            <div className="p-6 bg-white/5 border border-pink-500/30 rounded-2xl">
                                <div className="text-4xl font-bold text-pink-400 mb-2">2</div>
                                <h4 className="font-semibold mb-2">Phase 2: End-to-End Loop</h4>
                                <p className="text-gray-400 text-sm">End-to-end training with reconstructed flow.</p>
                            </div>
                            <div className="p-6 bg-white/5 border border-blue-500/30 rounded-2xl">
                                <div className="text-4xl font-bold text-blue-400 mb-2">3</div>
                                <h4 className="font-semibold mb-2">Phase 3: Perceptual Tuning</h4>
                                <p className="text-gray-400 text-sm">Fine-tuning with MS-SSIM and Bitrate constraints.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Results & Evaluation */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-black/20">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center" data-aos="fade-up">Results & Evaluation</h2>
                    <p className="text-gray-400 text-center mb-12" data-aos="fade-up">
                        The system was evaluated on the <strong className="text-white">UVG Dataset</strong> (Beauty, Jockey, ReadySetGo).
                    </p>

                    {/* Qualitative Reconstruction */}
                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Qualitative Reconstruction</h3>
                        <p className="text-gray-400 mb-6" data-aos="fade-up">
                            Comparison between the original frame and the reconstructed frame after compression/decompression.
                        </p>
                        <ImageWithCaption
                            src={realVsReconstructed}
                            alt="Original vs Reconstructed"
                            caption="Original Frame vs Reconstructed Frame"
                        />
                    </div>

                    {/* RAFT Prediction */}
                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">RAFT Flow Prediction</h3>
                        <ImageWithCaption
                            src={raftPrediction}
                            alt="RAFT Prediction"
                            caption="RAFT Flow Prediction Sample"
                        />
                    </div>

                    {/* Benchmarks */}
                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Benchmarks</h3>
                        <h4 className="text-xl font-medium mb-4 text-gray-300" data-aos="fade-up">Rate-Distortion Curves</h4>
                        <ImageGrid images={[
                            { src: rdCurvePSNR, alt: "RD Curve PSNR", caption: "PSNR vs Bitrate" },
                            { src: rdCurveMSSSIM, alt: "RD Curve MS-SSIM", caption: "MS-SSIM vs Bitrate" }
                        ]} />

                        <h4 className="text-xl font-medium mb-4 mt-8 text-gray-300" data-aos="fade-up">Resolution Analysis</h4>
                        <ImageGrid images={[
                            { src: psnrResolution, alt: "PSNR by Resolution", caption: "PSNR by Resolution" },
                            { src: msssimResolution, alt: "MS-SSIM by Resolution", caption: "MS-SSIM by Resolution" }
                        ]} />
                    </div>

                    {/* Post-Processing */}
                    <div>
                        <h3 className="text-2xl font-semibold mb-6" data-aos="fade-up">Post-Processing</h3>
                        <ImageWithCaption
                            src={temporalFiltering}
                            alt="Temporal Filtering"
                            caption="First Order IIR Temporal Filtering"
                        />
                    </div>
                </div>
            </section>

            {/* Usage Section */}
            <section id="usage" className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-center" data-aos="fade-up">Usage</h2>

                    {/* Prerequisites */}
                    <div className="mb-8" data-aos="fade-up">
                        <h3 className="text-xl font-semibold mb-4">Prerequisites</h3>
                        <div className="flex flex-wrap gap-3">
                            {['Python 3.11', 'PyTorch (CUDA)', 'compressai', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'tqdm'].map((dep, idx) => (
                                <span key={idx} className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-sm">
                                    {dep}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Command Examples */}
                    <div className="space-y-4">
                        <CollapsibleSection id="encoding" title="📦 Encoding (Video → .rdvc)" defaultOpen={true}>
                            <p className="text-gray-400 mb-4">Encodes an input video into an .rdvc file.</p>
                            <CodeBlock>{`# Minimal Command
python codec_processing.py --mode encode

# With GPU selection
python codec_processing.py --mode encode --gpu 0

# Full Example
python codec_processing.py \\
    --mode encode \\
    --gpu 0 \\
    --raft_backend auto`}</CodeBlock>
                        </CollapsibleSection>

                        <CollapsibleSection id="decoding" title="📼 Decoding (.rdvc → Video)">
                            <p className="text-gray-400 mb-4">Decodes an .rdvc file into a reconstructed video.</p>
                            <CodeBlock>{`# Minimal Command
python codec_processing.py --mode decode

# With temporal filtering
python codec_processing.py --mode decode --temporal_filter_alpha 0.5

# Full Example
python codec_processing.py \\
    --mode decode \\
    --gpu 0 \\
    --temporal_filter_alpha 0.7`}</CodeBlock>
                        </CollapsibleSection>

                        <CollapsibleSection id="training" title="🧠 Model Training">
                            <p className="text-gray-400 mb-4">Train the RDVC neural video codec using the 3-phase training strategy.</p>
                            <CodeBlock>{`python new_train.py`}</CodeBlock>
                        </CollapsibleSection>

                        <CollapsibleSection id="cli" title="📋 CLI Arguments Summary">
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-white/20">
                                            <th className="text-left py-3 px-4 font-semibold">Argument</th>
                                            <th className="text-left py-3 px-4 font-semibold">Description</th>
                                        </tr>
                                    </thead>
                                    <tbody className="text-gray-400">
                                        <tr className="border-b border-white/10">
                                            <td className="py-3 px-4 font-mono text-purple-400">--mode</td>
                                            <td className="py-3 px-4">Operation mode: encode or decode (required)</td>
                                        </tr>
                                        <tr className="border-b border-white/10">
                                            <td className="py-3 px-4 font-mono text-purple-400">--gpu</td>
                                            <td className="py-3 px-4">GPU ID (0, 1, ...) or -1 for CPU</td>
                                        </tr>
                                        <tr className="border-b border-white/10">
                                            <td className="py-3 px-4 font-mono text-purple-400">--raft_backend</td>
                                            <td className="py-3 px-4">RAFT implementation (auto, torchvision, local)</td>
                                        </tr>
                                        <tr>
                                            <td className="py-3 px-4 font-mono text-purple-400">--temporal_filter_alpha</td>
                                            <td className="py-3 px-4">Temporal smoothing factor for decoder (0.0-1.0)</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </CollapsibleSection>
                    </div>
                </div>
            </section>

            {/* Project Structure */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-black/20">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-center" data-aos="fade-up">Project Structure</h2>
                    <div className="bg-black/40 border border-white/10 rounded-xl p-6 font-mono text-sm" data-aos="fade-up">
                        <div className="space-y-2 text-gray-300">
                            <p><span className="text-purple-400">📄 codec_processing.py</span> — Core VideoCodec class with Encoders, Decoders, Warping layers</p>
                            <p><span className="text-purple-400">📄 new_train.py</span> — Main training script with 3-phase training loop</p>
                            <p><span className="text-blue-400">📁 codec_checkpoints_*/</span> — Model checkpoints</p>
                            <p><span className="text-blue-400">📁 training_plots/</span> — Metric plots from training</p>
                            <p><span className="text-blue-400">📁 visualization_*/</span> — Reconstructed frames, flow maps, residuals</p>
                            <p><span className="text-blue-400">📁 benchmark/</span> — Performance graphs, RD curves, architecture diagrams</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Technologies */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center" data-aos="fade-up">Technologies Used</h2>
                    <div className="flex flex-wrap justify-center gap-4" data-aos="fade-up">
                        {['PyTorch', 'RAFT', 'CompressAI', 'Hyperprior AE', 'CUDA', 'Python', 'OpenCV', 'NumPy', 'TorchVision'].map((tech, index) => (
                            <span
                                key={index}
                                className="px-6 py-3 bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-full font-medium hover:border-purple-400 transition-all duration-300 hover:scale-105 cursor-default"
                            >
                                {tech}
                            </span>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 mb-8">
                <div
                    className="max-w-4xl mx-auto text-center p-12 bg-gradient-to-r from-purple-600/10 to-pink-600/10 border border-purple-500/20 rounded-3xl"
                    data-aos="fade-up"
                >
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">
                        Interested in Collaborating?
                    </h2>
                    <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
                        I'm always happy to collaborate on innovative projects. If you want to train the models from scratch, check out my RAFT repository for training on MPI Sintel.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <a
                            href="https://github.com/anis-hd/end2end"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-white/10 border border-white/20 rounded-lg font-semibold hover:bg-white/20 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            RAFT Repository
                        </a>
                        <Link
                            to="/"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all duration-300 hover:scale-105"
                        >
                            <FontAwesomeIcon icon={faArrowLeft} />
                            Back to Portfolio
                        </Link>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="px-6 lg:px-20 xl:px-36 py-8 border-t border-white/10">
                <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
                    © {new Date().getFullYear()} Anis Ben Houidi. All rights reserved.
                </div>
            </footer>
        </div>
    );
}
