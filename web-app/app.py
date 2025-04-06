#!/usr/bin/env python3

"""
Example flask-based web application.
See the README.md file for instructions how to set up and run the app in development mode.
"""

import os
import datetime
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv, dotenv_values

load_dotenv()  # load environment variables from .env file


def create_app():
    """
    Create and configure the Flask application.
    returns: app: the Flask application object
    """

    app = Flask(__name__)
    # load flask config from env variables
    config = dotenv_values()
    app.config.from_mapping(config)

    # cxn = pymongo.MongoClient(os.getenv("MONGO_URI"))
    # db = cxn[os.getenv("MONGO_DBNAME")]

    # try:
    #     cxn.admin.command("ping")
    #     print(" *", "Connected to MongoDB!")
    # except Exception as e:
    #     print(" * MongoDB connection error:", e)

    @app.route("/")
    def home():
        """
        Route for the home page.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("index.html")
    
    @app.route("/upload")#, methods = ["POST"])
    def upload():
        """
        Route for the upload page.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        # if request.method == "POST":
        #     #take in image here and get to client
        #     pass
        return render_template("upload.html")
    
    @app.route("/capture")#, methods = ["POST"])
    def capture():
        """
        Route for the capture page.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        # if request.method == "POST":
        #     #take in image here and get to client
        #     pass
        return render_template("capture.html")
    

    @app.errorhandler(Exception)
    def handle_error(e):
        """
        Output any errors - good for debugging.
        Args:
            e (Exception): The exception object.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("error.html", error=e)

    return app


app = create_app()

if __name__ == "__main__":
    FLASK_PORT = os.getenv("FLASK_PORT", "5000")
    FLASK_ENV = os.getenv("FLASK_ENV")
    print(f"FLASK_ENV: {FLASK_ENV}, FLASK_PORT: {FLASK_PORT}")

    app.run(port=FLASK_PORT)