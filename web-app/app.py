"""This is a Flask Web App"""

import os
import logging
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv, dotenv_values
import pymongo
from werkzeug.utils import secure_filename

load_dotenv()  # load environment variables from .env file

INPUT_DIR = os.getenv("INPUT_DIR", "images/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "images/output")


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
    

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ["jpg", "png"]

    @app.route("/final_image", methods = ["POST"])
    def final_image():
        """
        Route accepting a photo that was either uploaded or captured 
        and giving it to machine learning client through MongoDB.
        Renders final page with output from machine learning client
        Returns:
            rendered template (str): The rendered HTML template.
        """

        image_file = request.files.get("faceImage")
        image_name = image_file.filename
        app.logger.debug("name of file %s", image_name)

        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(INPUT_DIR, filename))

            image_data = {
                "file_name": filename,
                "path": os.path.join(INPUT_DIR, filename),
                "is_cover": False
            }

            og_file_id = db.input_image.insert_one(image_data)
            app.logger.debug('adding image to collection: %s', og_file_id)

        # see if they have custom image
        cover_image = request.files.get("coverImage")
        if cover_image and allowed_file(cover_image.filename):
            filename = secure_filename(cover_image.filename)
            cover_path = os.path.join(INPUT_DIR, filename)
            cover_image.save(cover_path)

            add_cover = db.input_image.update_one({"_id": og_file_id}, {"$set": {"cover_path": cover_path}})
            app.logger.debug('adding cover_file path to image: %s', add_cover)

        # TODO: then fetch image and id and pass to ml client
        image = db.input_image.find_one({"_id": og_file_id})
        # TODO: add new image to separate collection with id of original 
        # TODO: pass new image to output page to display
        return render_template("index.html", {'output_path': image["path"], "success": True})
    

    @app.errorhandler(Exception)
    def handle_error(e):
        """
        Output any errors - good for debugging.
        Args:
            e (Exception): The exception object.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("error.html", error=e), 404

    return app


app = create_app()

if __name__ == "__main__":
    FLASK_PORT = os.getenv("FLASK_PORT", "5000")
    FLASK_ENV = os.getenv("FLASK_ENV")
    print(f"FLASK_ENV: {FLASK_ENV}, FLASK_PORT: {FLASK_PORT}")

    app.run(port=FLASK_PORT)