"""
Tests for the Flask web application.
"""

import io
import os
import sys
import time
from pathlib import Path

import base64
import pytest
from bson.objectid import ObjectId

# Add the parent directory to the path to allow importing 'app'
PARENT_DIR = str(Path(__file__).parent.parent.absolute())
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
# pylint: disable=wrong-import-position,import-error
from app import create_app


@pytest.fixture(scope="module")
def flask_app():
    """Create and configure a Flask app for testing."""
    # Create app with test config
    test_app = create_app()
    test_app.config.update({"TESTING": True, "WTF_CSRF_ENABLED": False})

    # Wait a moment for MongoDB connection to be established
    time.sleep(1)

    # Clear all test collections before each test
    mongo_client = test_app.extensions.get("pymongo")
    if mongo_client:
        db = mongo_client[os.getenv("MONGO_DBNAME", "okaycooldb")]
        # Don't drop GridFS collections fully, but clear them
        db.fs.files.delete_many({})
        db.fs.chunks.delete_many({})
        db.input_images.files.delete_many({})
        db.input_images.chunks.delete_many({})
        db.output_images.files.delete_many({})
        db.output_images.chunks.delete_many({})

        # Clear other collections
        db.image_processing.delete_many({})
        db.face_detection_results.delete_many({})

    return test_app


@pytest.fixture
def app(flask_app):  # pylint: disable=redefined-outer-name
    """Provide the app fixture."""
    return flask_app


@pytest.fixture
def client(app):  # pylint: disable=redefined-outer-name
    """Create a test client for the app."""
    return app.test_client()


@pytest.fixture
def test_image_jpg():
    """Create a minimal valid JPEG image.

    This is a 1x1 pixel JPEG file.
    """
    # Minimal valid JPEG file (1x1 pixel) - base64 encoded
    minimal_jpeg = base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP////////////////////////////"
        "/////////////////////////////////////////////////////"
        "wAALCAABAAEBAREA/8QAJgABAAAAAAAAAAAAAAAAAAAAAxABAAAAAAAA"
        "AAAAAAAAAAAAP/aAAgBAQAAPwBH/9k="
    )
    return io.BytesIO(minimal_jpeg)


@pytest.fixture
def test_image_png():
    """Create a minimal valid PNG image.

    This is a 1x1 pixel PNG file.
    """
    # Minimal valid PNG file (1x1 pixel) - base64 encoded
    minimal_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4"
        "//8/AAX+Av7czFnnAAAAAElFTkSuQmCC"
    )
    return io.BytesIO(minimal_png)


def test_home_page(client):  # pylint: disable=redefined-outer-name
    """Test that the home page returns 200 status code."""
    response = client.get("/")
    assert response.status_code == 200


def test_upload_image_valid(
    client, test_image_jpg
):  # pylint: disable=redefined-outer-name
    """Test uploading a valid image."""
    test_file = (test_image_jpg, "test_image.jpg")

    response = client.post(
        "/final_image",
        data={"faceImage": test_file},
        content_type="multipart/form-data",
    )

    # Check response
    assert response.status_code == 200
    assert b"Your image has been uploaded and is being processed" in response.data


def test_upload_image_with_cover(  # pylint: disable=redefined-outer-name
    client, test_image_jpg, test_image_png
):
    """Test uploading an image with a custom cover image."""
    # Create test files
    main_file = (test_image_jpg, "main_image.jpg")
    cover_file = (test_image_png, "cover_image.png")

    response = client.post(
        "/final_image",
        data={
            "faceImage": main_file,
            "coverImage": cover_file,
            "useCustomCover": "true",
        },
        content_type="multipart/form-data",
    )

    # Check response
    assert response.status_code == 200
    assert b"Your image has been uploaded and is being processed" in response.data


def test_upload_invalid_file_type(client):  # pylint: disable=redefined-outer-name
    """Test uploading an invalid file type."""
    # Create a test text file
    test_file = (io.BytesIO(b"not an image"), "test.txt")

    response = client.post(
        "/final_image",
        data={"faceImage": test_file},
        content_type="multipart/form-data",
    )

    # In a proper REST API, 400 is the correct status code for invalid inputs
    assert response.status_code == 400
    # Verify we get an error message in JSON format
    assert "error" in response.get_json()


def test_no_file_uploaded(client):  # pylint: disable=redefined-outer-name
    """Test submitting the form without a file."""
    response = client.post("/final_image", data={}, content_type="multipart/form-data")

    # In a proper REST API, 400 is the correct status code for missing required inputs
    assert response.status_code == 400
    # Verify we get an error message in JSON format
    assert "error" in response.get_json()


def test_check_status_endpoint(client, app):  # pylint: disable=redefined-outer-name
    """Test the check_status endpoint with real database."""
    # Get the MongoDB connections from the app
    db = app.extensions.get("mongodb")
    if not db:
        pytest.skip("MongoDB connection not available")

    processing_collection = db.image_processing

    # Create a test record
    output_file_id = ObjectId()
    record_id = processing_collection.insert_one(
        {
            "status": "completed",
            "output_file_id": output_file_id,
            "filename": "test.jpg",
            "created_at": time.time(),
        }
    ).inserted_id

    # Test the endpoint
    response = client.get(f"/check_status/{record_id}")
    assert response.status_code == 200

    # Parse JSON response
    data = response.get_json()
    assert data["status"] == "completed"
    assert "file_id" in data
    assert data["file_id"] == str(output_file_id)


def test_image_data_endpoint(  # pylint: disable=redefined-outer-name
    client, app, test_image_jpg
):
    """Test the image_data endpoint with real GridFS."""
    # Get the MongoDB connection and output bucket from app extensions
    db = app.extensions.get("mongodb")
    output_bucket = app.extensions.get("output_bucket")

    if not db or not output_bucket:
        pytest.skip("MongoDB or GridFS bucket not available")

    # Upload a test image to GridFS
    test_image_jpg.seek(0)  # Reset position to beginning
    file_id = output_bucket.upload_from_stream(
        "test.jpg", test_image_jpg.read(), metadata={"content_type": "image/jpeg"}
    )

    # Test the endpoint
    response = client.get(f"/image_data/{file_id}")
    assert response.status_code == 200

    # The response should be the image data
    test_image_jpg.seek(0)  # Reset position to beginning
    assert response.data == test_image_jpg.read()


def test_get_image_endpoint(  # pylint: disable=redefined-outer-name
    client, app, test_image_jpg
):
    """Test the get_image endpoint with real database and GridFS."""
    # Get the MongoDB connection and output bucket from app extensions
    db = app.extensions.get("mongodb")
    output_bucket = app.extensions.get("output_bucket")

    if not db or not output_bucket:
        pytest.skip("MongoDB or GridFS bucket not available")

    processing_collection = db.image_processing

    # Upload a test image to GridFS
    test_image_jpg.seek(0)  # Reset position to beginning
    output_file_id = output_bucket.upload_from_stream(
        "test.jpg", test_image_jpg.read(), metadata={"content_type": "image/jpeg"}
    )

    # Create a processing record
    record_id = processing_collection.insert_one(
        {
            "status": "completed",
            "output_file_id": output_file_id,
            "num_faces": 2,
            "filename": "test.jpg",
            "processing_time": 1.23,
            "created_at": time.time(),
            "completed_at": time.time(),
        }
    ).inserted_id

    # Test the endpoint
    response = client.get(f"/get_image/{record_id}")
    assert response.status_code == 200
    assert b"Face Redaction Results" in response.data
    assert b"test.jpg" in response.data


def test_error_handler(client):  # pylint: disable=redefined-outer-name
    """Test the error handler."""
    # Cause a deliberate exception by accessing a route that doesn't exist
    response = client.get("/nonexistent_route")

    # Should render the error template
    assert b"error" in response.data.lower()
