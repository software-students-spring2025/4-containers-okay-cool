"""This is a Flask Web App"""

import os
import logging
import json
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv, dotenv_values
import pymongo
from werkzeug.utils import secure_filename
import gridfs
import bson
from bson.objectid import ObjectId

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

    # Create MongoDB connection
    cxn = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = cxn[os.getenv("MONGO_DBNAME")]
    
    # Store MongoDB client and DB as Flask extensions
    if not hasattr(app, 'extensions'):
        app.extensions = {}
    app.extensions['pymongo'] = cxn
    app.extensions['mongodb'] = db
    
    # Create GridFS buckets
    input_bucket = gridfs.GridFSBucket(db, bucket_name="input_images")
    output_bucket = gridfs.GridFSBucket(db, bucket_name="output_images")
    
    # Store GridFS buckets as Flask extensions
    app.extensions['input_bucket'] = input_bucket
    app.extensions['output_bucket'] = output_bucket

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
        if collection not in ['fs.files', 'fs.chunks']:  # Don't drop GridFS collections
            db[collection].drop()

    # Create collection for tracking image processing status
    processing_collection = db.image_processing

    @app.route("/")
    def home():
        """
        Route for the home page.
        Returns:
            rendered template (str): The rendered HTML template.
        """
        return render_template("index.html")

    def allowed_file(filename):
        """Check if file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ["jpg", "png"]

    @app.route("/final_image", methods=["POST"])
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
        app.logger.debug("name of file %s", image_name)

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
                metadata={"type": "redaction_image"}
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
                    "has_custom_cover": has_custom_cover
                }
            )

            # Create a processing record
            processing_id = processing_collection.insert_one({
                "input_file_id": file_id,
                "filename": filename,
                "status": "pending",
                "created_at": time.time(),
                "has_custom_cover": has_custom_cover,
                "cover_image_id": cover_image_id if has_custom_cover else None
            }).inserted_id

            app.logger.debug(f'Image uploaded to MongoDB GridFS with file_id: {file_id}')
            
            # Return success with the processing ID for status checks
            return jsonify({
                "success": True,
                "file_id": str(file_id),
                "filename": filename,
                "status": "pending",
                "message": "Your image has been uploaded and is being processed"
            })
        
        return jsonify({"error": "Invalid image file"}), 400

    @app.route("/check_status/<file_id>")
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
            
            app.logger.debug(f"Checking status for file_id: {file_id}")
            
            # Find the processing record by input_file_id instead of _id
            record = processing_collection.find_one({"input_file_id": object_id})
            
            if not record:
                app.logger.warning(f"No processing record found with input_file_id: {object_id}")
                return jsonify({"error": "Processing record not found"}), 404
            
            app.logger.debug(f"Found processing record: {record.get('_id')} with status: {record.get('status', 'unknown')}")
            
            # Return the status
            response = {
                "status": record.get("status", "unknown")
            }
            
            # If completed, also return the output file ID
            if record.get("status") == "completed" and "output_file_id" in record:
                response["file_id"] = str(record["_id"])  # Return the processing record ID for the get_image endpoint
                app.logger.debug(f"Processing completed, output_file_id: {record.get('output_file_id')}")
            
            # If failed, include the error message
            if record.get("status") == "failed" and "error" in record:
                response["error"] = record["error"]
                app.logger.warning(f"Processing failed with error: {record.get('error')}")
            
            return jsonify(response)
                
        except Exception as e:
            app.logger.error(f"Error checking status: {str(e)}")
            return jsonify({"error": str(e)}), 400

    @app.route("/get_image/<file_id>")
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
                return render_template("error.html", error="Processing record not found")
                
            if record.get("status") != "completed":
                return render_template("error.html", error=f"Image processing is {record.get('status')}")
                
            # Get the output file ID
            output_file_id = record.get("output_file_id")
            if not output_file_id:
                return render_template("error.html", error="Output file ID not found")
                
            # Check if the output file exists
            try:
                # Try to open the file to verify it exists
                output_bucket.open_download_stream(output_file_id)
            except Exception as e:
                return render_template("error.html", error=f"Output file not found: {str(e)}")
                
            # Render the result template with the file ID and metadata
            return render_template(
                "result.html",
                image_id=str(output_file_id),
                filename=record.get("filename", "Unknown"),
                num_faces=record.get("num_faces", 0),
                processing_time=record.get("processing_time", 0)
            )
                
        except Exception as e:
            app.logger.error(f"Error getting image: {str(e)}")
            return render_template("error.html", error=str(e))

    @app.route("/image_data/<file_id>")
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
            from flask import Response
            return Response(
                grid_out.read(), 
                mimetype="image/jpeg" if grid_out.filename.endswith((".jpg", ".jpeg")) else "image/png"
            )
                
        except Exception as e:
            app.logger.error(f"Error streaming image: {str(e)}")
            return jsonify({"error": str(e)}), 400

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