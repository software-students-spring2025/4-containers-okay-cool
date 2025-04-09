[![ML-Client-CI](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/ml-client-ci.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/ml-client-ci.yml)
[![Web-App-CI](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/web-app-ci.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-okay-cool/actions/workflows/web-app-ci.yml)
![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Face Blocker
## Description
Face Blocker is a web app that allows a user to generate a version of any photo with the face blocked out. This system integrates a machine learning system that takes the image, detects how many faces are in the image, and produces an output image with all of the faces blocked out. Face Blocker can even use a custom photo the user provides to censor the faces. This system also integrates a MongoDB database using GridFS that facilitates sharing of images between the two other containers. 

## Developer Instructions

1. Clone the git repository 
    ``` git clone https://github.com/software-students-spring2025/4-containers-okay-cool.git ```
2. Create .env files for the web app and machine learning client
    * See [instructions](#environent-variables) below
3. Install docker and docker desktop
4. Open a terminal in the base repository of the project
5. Run the following command to start the web app, machine learning client, and MongoDB database
    * ```docker compose up --build```
    * If you make any changes to the docker-compose file or Dockerfiles run ```docker compose up --force-recreate --build```
6. Now you can view the web app at ```http://localhost:10000/```
    * You can change this port in ```docker-compose.yaml``` and in the two ```.env``` files
7. Run the following command to stop and remove the containers 
    * ```docker compose down```
    * Note that after any code changes you must compose down and then complete step 5 to apply them

## Environent variables
Both the web app and machine learning client require ```.env``` files to function. Follow the example files below to create your own versions. It is vital to include the same fields, but insert your personal uri and database name for MongoDB.
1. (Example ```.env``` for the web app)[web-app/.env.example]
2. (Example ```.env``` for the machine learning client)[machine-learning-client/.env.example]

## Authors
* [Apollo Wyndham](https://github.com/a-wyndham1)
* [Benjamin DeWeese](https://github.com/bdeweesevans)
* [Willow McKinnis](https://github.com/Willow-Zero)
* [Noah Perelmuter](https://github.com/np2446)
