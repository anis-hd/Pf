import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import SkillCard from "./SkillCard.js"

import javascript from "../assets/skills/javascript.svg"
import bash from "../assets/skills/bash.svg"
import linux from "../assets/skills/linux.svg"
import python from "../assets/skills/python.svg"
import reactIcon from "../assets/skills/react.svg"
import tailwind from "../assets/skills/tailwind.svg"
import git from "../assets/skills/git.svg"
import tensorflow from "../assets/skills/tensorflow.png"
import hr from "../assets/curve-hr.svg"
import pytorch from "../assets/skills/pytorch.svg"
import scikitlearn from "../assets/skills/scikitlearn.png"
import django from "../assets/skills/dj.svg"
import flask from "../assets/skills/flask.svg"
import nodejsicon from "../assets/skills/nodejs-icon.svg"
import mongodb from "../assets/skills/mongodb.svg"
import java from "../assets/skills/java-4-logo.svg"
import cee from "../assets/skills/c.png"
import cpp from "../assets/skills/cpp.png"
import flutter from "../assets/skills/flutter.svg"
export default function Skills() {
    const settings = {
        dots: false,
        autoplay: true,
        infinite: true,
        slidesToShow: 2,
        slidesToScroll: 1
      };

    return (
        <div id="skills" className="mt-4 text-white">
            <h1 className="text-2xl font-bold">Skills</h1>
            <p className="font-light text-gray-400">Here are some of my skills</p>

            <div className="mt-4">
                <Slider {...settings}>
                <SkillCard name="Django" experience="2 years" img={django} />
                <SkillCard name="Flask" experience="2 years" img={flask} />
                <SkillCard name="Linux" experience="4 years" img={linux} />
                <SkillCard name="Bash" experience="4 years" img={bash} />
                <SkillCard name="Python" experience="4 years" img={python} />
                <SkillCard name="Javascript" experience="2 years" img={javascript} />
                <SkillCard name="React" experience="1 years" img={reactIcon} />
                <SkillCard name="Tailwind" experience="1 years" img={tailwind} />
                <SkillCard name="Git" experience="2 years" img={git} />
                <SkillCard name="Tensorflow" experience="2 years" img={tensorflow} />
                <SkillCard name="Pytorch" experience="1 years" img={pytorch} />
                <SkillCard name="Scikitlearn" experience="1 years" img={scikitlearn} />
                <SkillCard name="NodeJs" experience="1 years" img={nodejsicon} />
                <SkillCard name="MongoDB" experience="1 years" img={mongodb} />
                <SkillCard name="Java" experience="2 years" img={java} />
                <SkillCard name="C" experience="2 years" img={cee} />
                <SkillCard name="C++" experience="2 years" img={cpp} />
                <SkillCard name="Flutter/Dart" experience="2 years" img={flutter} />





                </Slider>
            </div>
        </div>
    )
}