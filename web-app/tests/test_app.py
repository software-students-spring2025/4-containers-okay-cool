"""
Tests for the Flask web application.
"""

import os
import io
import time
import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Use the actual MongoDB connection as defined in docker-compose
    # The test will be run within Docker, so MongoDB will be available
    
    # Create directories if they don't exist
    os.makedirs(os.environ.get("INPUT_DIR", "images/input"), exist_ok=True)
    os.makedirs(os.environ.get("OUTPUT_DIR", "images/output"), exist_ok=True)
    
    # Create app with test config
    app = create_app()
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    })
    
    # Wait a moment for MongoDB connection to be established
    time.sleep(1)
    
    # Clear all test collections before each test
    if hasattr(app, 'db'):
        collections = app.db.list_collection_names()
        for collection in collections:
            app.db[collection].drop()
    
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()


def test_home_page(client):
    """Test that the home page returns 200 status code."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data


def test_upload_image_valid(client):
    """Test uploading a valid image file."""
    # Create a test image file
    test_image = (io.BytesIO(b"test image data"), "test.jpg")
    
    # Make the request
    response = client.post(
        "/final_image",
        data={
            "faceImage": test_image
        },
        content_type="multipart/form-data"
    )
    
    # Check the response
    assert response.status_code == 200


def test_upload_image_with_cover(client):
    """Test uploading both a face image and a cover image."""
    # Create test image files
    test_face_image = (io.BytesIO(b"test face image data"), "face.jpg")
    test_cover_image = (io.BytesIO(b"test cover image data"), "cover.png")
    
    # Make the request
    response = client.post(
        "/final_image",
        data={
            "faceImage": test_face_image,
            "coverImage": test_cover_image
        },
        content_type="multipart/form-data"
    )
    
    # Check response
    assert response.status_code == 200


def test_upload_invalid_file_type(client):
    """Test uploading an invalid file type."""
    # Create a test text file
    test_file = (io.BytesIO(b"not an image"), "test.txt")
    
    response = client.post(
        "/final_image",
        data={
            "faceImage": test_file
        },
        content_type="multipart/form-data"
    )
    
    # Should still return 200 but no processing should happen
    assert response.status_code == 200


def test_no_file_uploaded(client):
    """Test submitting the form without a file."""
    response = client.post(
        "/final_image",
        data={},
        content_type="multipart/form-data"
    )
    
    # Should still render the template
    assert response.status_code == 200


def test_error_handler(client):
    """Test the error handler."""
    # Cause a deliberate exception by accessing a route that doesn't exist
    response = client.get("/nonexistent_route")
    
    # Should return 404
    assert response.status_code == 404
    
    # Should render the error template
    assert b"error" in response.data.lower()
