"""This is a Flask Web App"""

import os
import logging
import time
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
)
from dotenv import load_dotenv, dotenv_values
import pymongo
from werkzeug.utils import secure_filename
import gridfs
from bson.objectid import ObjectId

load_dotenv()  # load environment variables from .env file


def setup_mongodb_connections():
    """Create MongoDB and GridFS connections and return them."""
    cxn = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = cxn[os.getenv("MONGO_DBNAME")]
    input_bucket = gridfs.GridFSBucket(db, bucket_name="input_images")
    output_bucket = gridfs.GridFSBucket(db, bucket_name="output_images")

    try:
        cxn.admin.command("ping")
        print(" *", "Connected to MongoDB!")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(" * MongoDB connection error:", e)

    return cxn, db, input_bucket, output_bucket


# pylint: disable=too-many-locals,too-many-statements
def create_app():
    """
    Create and configure the Flask application.
    returns: app: the Flask application object
    """

    flask_app = Flask(__name__)
    # load flask config from env variables
    config = dotenv_values()
    flask_app.config.from_mapping(config)

    # Create MongoDB connections
    cxn, db, input_bucket, output_bucket = setup_mongodb_connections()

    # Store MongoDB client and DB as Flask extensions
    if not hasattr(flask_app, "extensions"):
        flask_app.extensions = {}
    flask_app.extensions["pymongo"] = cxn
    flask_app.extensions["mongodb"] = db

    # Store GridFS buckets as Flask extensions
    flask_app.extensions["input_bucket"] = input_bucket
    flask_app.extensions["output_bucket"] = output_bucket

    # Set up logging in Docker container's output
    logging.basicConfig(level=logging.DEBUG)

    # Drop all collections to prevent duplicated data getting
    # inserted into the database whenever the app is restarted
    collections = db.list_collection_names()
    for collection in collections:
        if collection not in ["fs.files", "fs.chunks"]:  # Don't drop GridFS collections
            db[collection].drop()

    # Create collection for tracking image processing status
    processing_collection = db.image_processing

    @flask_app.route("/")
    def home():
        """
        Route for the home page.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("index.html")

    def allowed_file(filename):
        """Check if file has an allowed extension."""
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ["jpg", "png"]

    @flask_app.route("/final_image", methods=["POST"])
    def final_image():
        """
        Route accepting a photo that was either uploaded or captured
        and giving it to machine learning client through MongoDB.
        Renders final page with output from machine learning client
        Returns:
            JSON response with processing info
        """
        image_file = request.files.get("faceImage")
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        image_name = image_file.filename
        flask_app.logger.debug("name of file %s", image_name)

        # Track if we have a custom cover image
        has_custom_cover = False
        cover_image_id = None

        # See if they have custom image for redaction
        cover_image = request.files.get("coverImage")
        if cover_image and allowed_file(cover_image.filename):
            has_custom_cover = True
            cover_filename = secure_filename(cover_image.filename)

            # Save to GridFS
            cover_image.seek(0)
            cover_image_id = input_bucket.upload_from_stream(
                cover_filename,
                cover_image.stream,
                metadata={"type": "redaction_image"},
            )

        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)

            # Upload the image file to MongoDB GridFS
            image_file.seek(0)
            file_id = input_bucket.upload_from_stream(
                filename,
                image_file.stream,
                metadata={
                    "cover_image_id": cover_image_id if has_custom_cover else None,
                    "has_custom_cover": has_custom_cover,
                },
            )

            # Create a processing record
            processing_collection.insert_one(
                {
                    "input_file_id": file_id,
                    "filename": filename,
                    "status": "pending",
                    "created_at": time.time(),
                    "has_custom_cover": has_custom_cover,
                    "cover_image_id": cover_image_id if has_custom_cover else None,
                }
            )

            flask_app.logger.debug(
                "Image uploaded to MongoDB GridFS with file_id: %s", file_id
            )

            # Return success with the processing ID for status checks
            return jsonify(
                {
                    "success": True,
                    "file_id": str(file_id),
                    "filename": filename,
                    "status": "pending",
                    "message": "Your image has been uploaded and is being processed",
                }
            )

        return jsonify({"error": "Invalid image file"}), 400

    @flask_app.route("/check_status/<file_id>")
    def check_status(file_id):
        """
        Check the processing status of an uploaded image
        Args:
            file_id (str): The GridFS ID of the input image
        Returns:
            JSON with status information
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(file_id)

            flask_app.logger.debug("Checking status for file_id: %s", file_id)

            # Find the processing record by input_file_id instead of _id
            record = processing_collection.find_one({"input_file_id": object_id})

            if not record:
                flask_app.logger.warning(
                    "No processing record found with input_file_id: %s", object_id
                )
                return jsonify({"error": "Processing record not found"}), 404

            flask_app.logger.debug(
                "Found processing record: %s with status: %s",
                record.get("_id"),
                record.get("status", "unknown"),
            )

            # Return the status
            response = {"status": record.get("status", "unknown")}

            # If completed, also return the output file ID
            if record.get("status") == "completed" and "output_file_id" in record:
                # Return the processing record ID for the get_image endpoint
                response["file_id"] = str(record["_id"])
                flask_app.logger.debug(
                    "Processing completed, output_file_id: %s",
                    record.get("output_file_id"),
                )

            # If failed, include the error message
            if record.get("status") == "failed" and "error" in record:
                response["error"] = record["error"]
                flask_app.logger.warning(
                    "Processing failed with error: %s", record.get("error")
                )

            return jsonify(response)

        except Exception as e:  # pylint: disable=broad-exception-caught
            flask_app.logger.error("Error checking status: %s", str(e))
            return jsonify({"error": str(e)}), 400

    @flask_app.route("/get_image/<file_id>")
    def get_image(file_id):
        """
        Get the redacted image and display it

        Args:
            file_id: The ID of the processing record

        Returns:
            Rendered template with the processed image
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(file_id)

            # Find the processing record
            record = processing_collection.find_one({"_id": object_id})

            if not record:
                return render_template(
                    "error.html", error="Processing record not found"
                )

            if record.get("status") != "completed":
                return render_template(
                    "error.html", error=f"Image processing is {record.get('status')}"
                )

            # Get the output file ID
            output_file_id = record.get("output_file_id")
            if not output_file_id:
                return render_template("error.html", error="Output file ID not found")

            # Check if the output file exists
            try:
                # Try to open the file to verify it exists
                output_bucket.open_download_stream(output_file_id)
            except Exception as e:  # pylint: disable=broad-exception-caught
                return render_template(
                    "error.html", error=f"Output file not found: {str(e)}"
                )

            # Render the result template with the file ID and metadata
            return render_template(
                "result.html",
                image_id=str(output_file_id),
                filename=record.get("filename", "Unknown"),
                num_faces=record.get("num_faces", 0),
                processing_time=record.get("processing_time", 0),
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            flask_app.logger.error("Error getting image: %s", str(e))
            return render_template("error.html", error=str(e))

    @flask_app.route("/image_data/<file_id>")
    def image_data(file_id):
        """
        Stream image data from GridFS

        Args:
            file_id: The ID of the GridFS file

        Returns:
            Image file response
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(file_id)

            # Get the file data from GridFS
            grid_out = output_bucket.open_download_stream(object_id)

            # Return the image as a response
            mimetype = (
                "image/jpeg"
                if grid_out.filename.endswith((".jpg", ".jpeg"))
                else "image/png"
            )
            return Response(grid_out.read(), mimetype=mimetype)

        except Exception as e:  # pylint: disable=broad-exception-caught
            flask_app.logger.error("Error streaming image: %s", str(e))
            return jsonify({"error": str(e)}), 400

    @flask_app.errorhandler(Exception)
    def handle_error(e):
        """
        Output any errors - good for debugging.
        Args:
            e (Exception): The exception object.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("error.html", error=e)

    return flask_app


app = create_app()

if __name__ == "__main__":
    FLASK_PORT = os.getenv("FLASK_PORT", "5000")
    FLASK_ENV = os.getenv("FLASK_ENV")
    print(f"FLASK_ENV: {FLASK_ENV}, FLASK_PORT: {FLASK_PORT}")

    app.run(port=FLASK_PORT)
