"""This is a Flask Web App"""

import os
import io
import logging
from flask import Flask, render_template, request #, redirect, url_for
from dotenv import load_dotenv, dotenv_values
import pymongo
from PIL import Image

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

    cxn = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = cxn[os.getenv("MONGO_DBNAME")]

    try:
        cxn.admin.command("ping")
        print(" *", "Connected to MongoDB!")
    except Exception as e:
        print(" * MongoDB connection error:", e)

    # Set up logging in Docker container's output
    logging.basicConfig(level=logging.DEBUG)

    # Drop all collections to prevent duplicated data getting
    # inserted into the database whenever the app is restarted
    collections = db.list_collection_names()
    for collection in collections:
        db[collection].drop()

    @app.route("/")
    def home():
        """
        Route for the home page.
        Returns:
            rendered template (str): The rendered HTML template.
        """

        return render_template("index.html")
    
    @app.route("/final_image", methods = ["POST"])
    def final_image():
        """
        Route accepting a photo that was either uploaded or captured 
        and giving it to machine learning client through MongoDB.
        Renders final page with output from machine learning client
        Returns:
            rendered template (str): The rendered HTML template.
        """
        image = request.form.get("myFile", "")

        im = Image.open(image)

        image_bytes = io.BytesIO()
        im.save(image_bytes, format='JPEG')

        image = {
            'data': image_bytes.getvalue()
        }

        # add image to collection of new images
        image_id = db.user_image.insert_one(image).inserted_id
        app.logger.debug('adding image to collection: %s', image_id)

        # TODO: then fetch image and id and pass to ml client
        # TODO: add new image to separate collection with id of original 
        # TODO: pass new image to output page to display
        return render_template("index.html") #, new_image)
    

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