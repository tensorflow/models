FROM tensorflow/tensorflow:1.4.0
WORKDIR /
RUN apt-get update
RUN apt-get -y install maven openjdk-8-jdk
RUN mvn dependency:get -Dartifact=org.tensorflow:tensorflow:1.4.0
RUN mvn dependency:get -Dartifact=org.tensorflow:proto:1.4.0
CMD ["/bin/bash", "-l"]
