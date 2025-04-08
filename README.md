[![Machine Learning Client CI](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/ml-client-ci.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/ml-client-ci.yml)
![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# What is this app?
This is a project that links together a flask web app, an ML client, and a database to censor faces out of images. The ML model recognizes where faces are in the image, the database stores the images, and the flask app allows for streamlined I/O.

# How to run
Clone the project, install docker compose, and then run ```docker compuse up --build``` to make and run the three containers.

## Environent variables
before you build and run, create .env files in both ./web-app and ./machine-learning-client with the data specified in the existing .env.examples.

# Who made it?
[Apollo Wyndham](https://github.com/a-wyndham1)
[Benjamin DeWeese](https://github.com/bdeweesevans)
[Willow McKinnis](https://github.com/Willow-Zero)
[Noah Perelmuter](https://github.com/np2446)
