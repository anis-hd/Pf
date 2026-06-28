import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faRocket, faCube, faCode, faVideo, faBrain, faLayerGroup, faChartLine, faGears, faTerminal, faGraduationCap, faBuilding } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

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
import phase1Metrics from '../assets/img/phase_1_metrics.png';
import phase2Metrics from '../assets/img/phase_2_metrics.png';
import phase3Vis from '../assets/img/epoch_0115_phase3_vis.png';

export default function HyperRaftPage() {
    useEffect(() => {
        document.title = 'RDVC: Raft Deep Video Compression - Anis Houidi';
        window.scrollTo(0, 0);
    }, []);

    const CodeBlock = ({ children }) => (
        <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 overflow-x-auto text-sm font-mono text-green-400">
            <code>{children}</code>
        </pre>
    );

    const ImageWithCaption = ({ src, alt, caption, maxWidth = "max-w-3xl", maxHeight = "max-h-[500px]" }) => (
        <div className="my-6 flex flex-col items-center">
            <div className={`relative rounded-xl overflow-hidden border border-slate-200 bg-slate-100 shadow-inner ${maxWidth} w-full`}>
                <img src={src} alt={alt} className={`w-full h-auto ${maxHeight} object-contain`} loading="lazy" />
            </div>
            {caption && <p className="text-center text-slate-500 text-sm mt-2 italic">{caption}</p>}
        </div>
    );

    const ImageGrid = ({ images, maxWidth = "max-w-4xl", maxHeight = "max-h-[300px]" }) => (
        <div className={`grid md:grid-cols-2 gap-6 my-6 ${maxWidth} mx-auto`}>
            {images.map((img, idx) => (
                <div key={idx} className="text-center">
                    <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-100 shadow-inner">
                        <img src={img.src} alt={img.alt} className={`w-full h-auto ${maxHeight} object-contain`} loading="lazy" />
                    </div>
                    <p className="text-slate-500 text-sm mt-2">{img.caption}</p>
                </div>
            ))}
        </div>
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
                        href="https://github.com/anis-hd/PFE-RAFT-and-hyperprior-based-learned-video-compression"
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
                    {/* Project Badge */}
                    <div className="flex flex-wrap gap-3 mb-6">
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-purple-50 border border-purple-200 rounded-full text-sm font-medium text-purple-700">
                            <FontAwesomeIcon icon={faRocket} className="text-purple-600" />
                            Final Year Project
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full text-sm font-medium text-blue-700">
                            <FontAwesomeIcon icon={faGraduationCap} className="text-blue-600" />
                            ENSI
                        </span>
                        <span className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full text-sm font-medium text-green-700">
                            <FontAwesomeIcon icon={faBuilding} className="text-green-600" />
                            Talan Tunisie
                        </span>
                    </div>

                    {/* Title */}
                    <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4 text-slate-900">
                        RDVC
                    </h1>
                    <h2 className="text-2xl md:text-3xl font-semibold text-slate-700 mb-6">
                        Raft Deep Video Compression
                    </h2>

                    {/* Subtitle */}
                    <p className="text-lg md:text-xl text-slate-600 max-w-4xl mb-8 leading-relaxed">
                        A Hybrid Video Compression Framework combining <span className="text-purple-600 font-semibold">RAFT Optical Flow</span>,
                        <span className="text-pink-600 font-semibold"> Hyperprior Entropy Coding</span>, and
                        <span className="text-blue-600 font-semibold"> Quantum-Inspired I-Frame Encoding</span>.
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
                            href="https://github.com/anis-hd/PFE-RAFT-and-hyperprior-based-learned-video-compression"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            Source Code
                        </a>
                        <a
                            href="#usage"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
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
                    <h2 className="text-3xl font-bold mb-8 text-slate-900 text-center">Demo Video</h2>
                    <p className="text-center text-slate-500 mb-8">
                        80% of the frames in this video are reconstructed during decoding
                    </p>
                    <div className="relative max-w-4xl mx-auto">
                        <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-white shadow-lg">
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
                    <h2 className="text-3xl font-bold mb-8 text-slate-900">Context</h2>
                    <div className="prose prose-slate max-w-none">
                        <p className="text-slate-600 leading-relaxed text-lg">
                            This project implements <strong className="text-slate-900">RDVC (Raft Deep Video Compression)</strong>, a Final Year Project developed at <strong className="text-green-700 font-semibold">Talan Tunisie</strong> in collaboration with the <strong className="text-blue-700 font-semibold">National School of Computer Science (ENSI)</strong>.
                        </p>
                        <p className="text-slate-600 leading-relaxed text-lg mt-4">
                            The system addresses the limitations of traditional codecs by proposing a deep learning-based inter-frame compression pipeline. It integrates <strong className="text-purple-600 font-semibold">RAFT</strong> (Recurrent All-Pairs Field Transforms) for optical flow estimation and <strong className="text-pink-600 font-semibold">Hyperprior Autoencoders</strong> for entropy coding. Additionally, the project explores <strong className="text-blue-600 font-semibold">Quantum Computing</strong> simulations for I-frame compression.
                        </p>
                    </div>
                </div>
            </section>

            {/* Key Features */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center text-slate-900">Key Features</h2>
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
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-slate-100/60 border-y border-slate-200/80">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center text-slate-900">System Architecture & Workflows</h2>
                    <p className="text-slate-500 text-center mb-12">
                        The RDVC codec operates on a P-frame architecture involving Motion and Residual branches.
                    </p>

                    <div className="space-y-8">
                        {/* Encoding/Decoding Block Diagrams */}
                        <div>
                            <h3 className="text-2xl font-semibold mb-6 text-slate-900">High-Level Schematics</h3>
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
                            <h3 className="text-2xl font-semibold mb-6 text-slate-900">Encoding & Decoding Processes</h3>
                            <p className="text-slate-500 mb-6">
                                The system uses a synchronized loop to ensure the encoder and decoder states remain identical.
                            </p>
                            <ImageGrid images={[
                                { src: encodingProcess, alt: "Video Encoding Workflow", caption: "Video Encoding Workflow" },
                                { src: decodingProcess, alt: "Video Decoding Workflow", caption: "Video Decoding Workflow" }
                            ]} />
                        </div>

                        {/* Core Components */}
                        <div>
                            <h3 className="text-2xl font-semibold mb-6 text-slate-900">Core Components</h3>
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
                    <h2 className="text-3xl font-bold mb-12 text-center text-slate-900">Training & Optimization</h2>

                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">Training Dynamics</h3>
                        <ImageWithCaption
                            src={raftTraining}
                            alt="RAFT Training"
                            caption="RAFT Training Process"
                        />
                    </div>

                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">Multi-Phase Training Strategy</h3>
                        <div className="grid md:grid-cols-3 gap-6">
                            <div className="p-6 bg-white border border-purple-200 shadow-sm rounded-2xl">
                                <div className="text-4xl font-bold text-purple-600 mb-2">1</div>
                                <h4 className="font-semibold mb-2 text-slate-900">Phase 1: Initialization</h4>
                                <p className="text-slate-600 text-sm">Train Residual AE using Ground Truth flow with MSE loss.</p>
                            </div>
                            <div className="p-6 bg-white border border-pink-200 shadow-sm rounded-2xl">
                                <div className="text-4xl font-bold text-pink-600 mb-2">2</div>
                                <h4 className="font-semibold mb-2 text-slate-900">Phase 2: End-to-End Loop</h4>
                                <p className="text-slate-600 text-sm">End-to-end training with reconstructed flow.</p>
                            </div>
                            <div className="p-6 bg-white border border-blue-200 shadow-sm rounded-2xl">
                                <div className="text-4xl font-bold text-blue-600 mb-2">3</div>
                                <h4 className="font-semibold mb-2 text-slate-900">Phase 3: Perceptual Tuning</h4>
                                <p className="text-slate-600 text-sm">Fine-tuning with MS-SSIM and Bitrate constraints.</p>
                            </div>
                        </div>

                        {/* Phase Metrics & Visualizations */}
                        <div className="mt-12 space-y-12">
                            <div>
                                <h4 className="text-xl font-medium mb-6 text-slate-700">Phase 1 & 2: Training Metrics</h4>
                                <ImageGrid
                                    images={[
                                        { src: phase1Metrics, alt: "Phase 1 Training Metrics", caption: "Phase 1: Loss & Accuracy" },
                                        { src: phase2Metrics, alt: "Phase 2 Training Metrics", caption: "Phase 2: Optimization Progress" }
                                    ]}
                                    maxWidth="max-w-6xl"
                                    maxHeight="max-h-[500px]"
                                />
                            </div>

                            <div>
                                <h4 className="text-xl font-medium mb-6 text-slate-700">Phase 3: Reconstruction Visualization</h4>
                                <p className="text-slate-500 mb-6 italic">
                                    Top: Original frame snippets. Bottom: Reconstructed counterparts after phase 3 fine-tuning.
                                </p>
                                <ImageWithCaption
                                    src={phase3Vis}
                                    alt="Phase 3 Visualization"
                                    caption="ResAE Visualization: Comparing original vs decoded frame blocks during Phase 3"
                                    maxWidth="max-w-6xl"
                                    maxHeight="max-h-[800px]"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Results & Evaluation */}
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-slate-100/60 border-y border-slate-200/80">
                <div className="max-w-7xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-center text-slate-900">Results & Evaluation</h2>
                    <p className="text-slate-500 text-center mb-12">
                        The system was evaluated on the <strong className="text-slate-900">UVG Dataset</strong> (Beauty, Jockey, ReadySetGo).
                    </p>

                    {/* Qualitative Reconstruction */}
                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">Qualitative Reconstruction</h3>
                        <p className="text-slate-500 mb-6">
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
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">RAFT Flow Prediction</h3>
                        <ImageWithCaption
                            src={raftPrediction}
                            alt="RAFT Prediction"
                            caption="RAFT Flow Prediction Sample"
                        />
                    </div>

                    {/* Benchmarks */}
                    <div className="mb-12">
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">Benchmarks</h3>
                        <h4 className="text-xl font-medium mb-4 text-slate-700">Rate-Distortion Curves</h4>
                        <ImageGrid images={[
                            { src: rdCurvePSNR, alt: "RD Curve PSNR", caption: "PSNR vs Bitrate" },
                            { src: rdCurveMSSSIM, alt: "RD Curve MS-SSIM", caption: "MS-SSIM vs Bitrate" }
                        ]} />

                        <h4 className="text-xl font-medium mb-4 mt-8 text-slate-700">Resolution Analysis</h4>
                        <ImageGrid images={[
                            { src: psnrResolution, alt: "PSNR by Resolution", caption: "PSNR by Resolution" },
                            { src: msssimResolution, alt: "MS-SSIM by Resolution", caption: "MS-SSIM by Resolution" }
                        ]} />
                    </div>

                    {/* Post-Processing */}
                    <div>
                        <h3 className="text-2xl font-semibold mb-6 text-slate-900">Post-Processing</h3>
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
                    <h2 className="text-3xl font-bold mb-8 text-center text-slate-900">Usage</h2>

                    {/* Prerequisites */}
                    <div className="mb-8">
                        <h3 className="text-xl font-semibold mb-4 text-slate-900">Prerequisites</h3>
                        <div className="flex flex-wrap gap-3">
                            {['Python 3.11', 'PyTorch (CUDA)', 'compressai', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'tqdm'].map((dep, idx) => (
                                <span key={idx} className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-700 shadow-sm">
                                    {dep}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Command Examples */}
                    <div className="space-y-4">
                        <CollapsibleSection id="encoding" title="📦 Encoding (Video → .rdvc)" defaultOpen={true}>
                            <p className="text-slate-500 mb-4">Encodes an input video into an .rdvc file.</p>
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
                            <p className="text-slate-500 mb-4">Decodes an .rdvc file into a reconstructed video.</p>
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
                            <p className="text-slate-500 mb-4">Train the RDVC neural video codec using the 3-phase training strategy.</p>
                            <CodeBlock>{`python new_train.py`}</CodeBlock>
                        </CollapsibleSection>

                        <CollapsibleSection id="cli" title="📋 CLI Arguments Summary">
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-slate-600">
                                    <thead>
                                        <tr className="border-b border-slate-200 text-slate-900">
                                            <th className="text-left py-3 px-4 font-semibold">Argument</th>
                                            <th className="text-left py-3 px-4 font-semibold">Description</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-b border-slate-100">
                                            <td className="py-3 px-4 font-mono text-blue-600 font-medium">--mode</td>
                                            <td className="py-3 px-4">Operation mode: encode or decode (required)</td>
                                        </tr>
                                        <tr className="border-b border-slate-100">
                                            <td className="py-3 px-4 font-mono text-blue-600 font-medium">--gpu</td>
                                            <td className="py-3 px-4">GPU ID (0, 1, ...) or -1 for CPU</td>
                                        </tr>
                                        <tr className="border-b border-slate-100">
                                            <td className="py-3 px-4 font-mono text-blue-600 font-medium">--raft_backend</td>
                                            <td className="py-3 px-4">RAFT implementation (auto, torchvision, local)</td>
                                        </tr>
                                        <tr>
                                            <td className="py-3 px-4 font-mono text-blue-600 font-medium">--temporal_filter_alpha</td>
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
            <section className="px-6 lg:px-20 xl:px-36 py-16 bg-slate-100/60 border-y border-slate-200/80">
                <div className="max-w-5xl mx-auto">
                    <h2 className="text-3xl font-bold mb-8 text-center text-slate-900">Project Structure</h2>
                    <div className="bg-white border border-slate-200 rounded-xl p-6 font-mono text-sm shadow-sm">
                        <div className="space-y-2 text-slate-600">
                            <p><span className="text-purple-600 font-semibold">📄 codec_processing.py</span> — Core VideoCodec class with Encoders, Decoders, Warping layers</p>
                            <p><span className="text-purple-600 font-semibold">📄 new_train.py</span> — Main training script with 3-phase training loop</p>
                            <p><span className="text-blue-600 font-semibold">📁 codec_checkpoints_*/</span> — Model checkpoints</p>
                            <p><span className="text-blue-600 font-semibold">📁 training_plots/</span> — Metric plots from training</p>
                            <p><span className="text-blue-600 font-semibold">📁 visualization_*/</span> — Reconstructed frames, flow maps, residuals</p>
                            <p><span className="text-blue-600 font-semibold">📁 benchmark/</span> — Performance graphs, RD curves, architecture diagrams</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Technologies */}
            <section className="px-6 lg:px-20 xl:px-36 py-16">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold mb-12 text-center text-slate-900">Technologies Used</h2>
                    <div className="flex flex-wrap justify-center gap-4">
                        {['PyTorch', 'RAFT', 'CompressAI', 'Hyperprior AE', 'CUDA', 'Python', 'OpenCV', 'NumPy', 'TorchVision'].map((tech, index) => (
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
                <div className="max-w-4xl mx-auto text-center p-12 bg-blue-50/50 border border-blue-100 rounded-3xl">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">
                        Interested in Collaborating?
                    </h2>
                    <p className="text-slate-600 mb-8 max-w-2xl mx-auto">
                        I'm always happy to collaborate on innovative projects. If you want to train the models from scratch, check out my RAFT repository for training on MPI Sintel.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <a
                            href="https://github.com/anis-hd/end2end"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 hover:border-slate-300 transition-all duration-300"
                        >
                            <FontAwesomeIcon icon={faGithub} />
                            RAFT Repository
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
