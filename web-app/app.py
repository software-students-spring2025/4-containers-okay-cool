import os
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv, dotenv_values
import pymongo
from PIL import Image
import io

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
    
    @app.route("/upload")
    def upload():
        """
        Route for the upload page.
        Returns:
            rendered template (str): The rendered HTML template.
        """

        return render_template("upload.html")
    
    @app.route("/capture")
    def capture():
        """
        Route for the capture page.
        Returns:
            rendered template (str): The rendered HTML template.
        """

        return render_template("capture.html")
    
    @app.route("/final_image", methods = ["POST"])
    def final_image():
        """
        Route accepting a photo that was either uploaded or captured and giving it to machine learning client through MongoDB.
        Renders final page with output from machine learning client
        Returns:
            rendered template (str): The rendered HTML template.
        """
        image = request.form.get("myFile")

        im = Image.open(image)

        image_bytes = io.BytesIO()
        im.save(image_bytes, format='JPEG')

        image = {
            'data': image_bytes.getvalue()
        }

        #image_id = images.insert_one(image).inserted_id

        return render_template("output.html")
    

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