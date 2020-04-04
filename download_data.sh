#! /bin/bash

mkdir data
mkdir data/audio
cd data/audio
wget http://www.montefiore.ulg.ac.be/services/acous/STSI/file/jim2012Chords.zip
unzip jim2012Chords.zip
rm jim2012Chords.zip
cd ..
mkdir metadata
