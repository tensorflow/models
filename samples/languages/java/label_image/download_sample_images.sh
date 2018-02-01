#!/bin/bash
DIR=$(dirname $0)
mkdir -p ${DIR}/images
cd ${DIR}/images

# Some random images
curl -o "porcupine.jpg" -L "https://cdn.pixabay.com/photo/2014/11/06/12/46/porcupines-519145_960_720.jpg"
curl -o "whale.jpg" -L "https://static.pexels.com/photos/417196/pexels-photo-417196.jpeg"
curl -o "terrier1u.jpg" -L "https://upload.wikimedia.org/wikipedia/commons/3/34/Australian_Terrier_Melly_%282%29.JPG"
curl -o "terrier2.jpg" -L "https://cdn.pixabay.com/photo/2014/05/13/07/44/yorkshire-terrier-343198_960_720.jpg"
