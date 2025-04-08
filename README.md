![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# What is this app?
This is a project that links together a flask web app, an ML client, and a database to censor faces out of images. The ML model recognizes where faces are in the image, the database stores the images, and the flask app allows for streamlined I/O.

# How to run
Clone the project, install docker compose, and then run ```docker compuse up --build``` to make and run the three containers

## Environent variables
before you build and run, create a .env file with the data specified in the existing file env.example.

# Who made it?
[Willow McKinnis](https://github.com/Willow-Zero), [a-wyndham1](https://github.com/a-wyndham1), [bdeweesevans](https://github.com/bdeweesevans), [np2446](https://github.com/np2446)


