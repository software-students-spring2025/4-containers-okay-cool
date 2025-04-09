"""
Tests for the Flask web application.
"""

import io
import time
import pytest
from bson.objectid import ObjectId
from app import create_app
from unittest.mock import patch, MagicMock


@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Create app with test config
    app = create_app()
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    })

    # Wait a moment for MongoDB connection to be established
    time.sleep(1)

    # Clear test collections before each test
    if hasattr(app, 'db'):
        collections = app.db.list_collection_names()
        for collection in collections:
            if collection not in ['fs.files', 'fs.chunks']:  # Don't drop GridFS collections
                app.db[collection].drop()

    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()


@pytest.fixture
def mock_gridfs():
    """Mock GridFS operations."""
    with patch('gridfs.GridFSBucket') as mock_grid:
        # Mock the upload_from_stream method to return an ObjectId
        mock_grid.return_value.upload_from_stream.return_value = ObjectId()
        
        # Mock the open_download_stream method
        mock_grid_out = MagicMock()
        mock_grid_out.read.return_value = b"mocked image data"
        mock_grid_out.filename = "test.jpg"
        mock_grid.return_value.open_download_stream.return_value = mock_grid_out
        
        yield mock_grid


def test_home_page(client):
    """Test that the home page returns 200 status code."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data


def test_upload_image_valid(client, mock_gridfs):
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
    assert mock_gridfs.return_value.upload_from_stream.called


def test_upload_image_with_cover(client, mock_gridfs):
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
    # Should be called twice, once for face image and once for cover image
    assert mock_gridfs.return_value.upload_from_stream.call_count == 2


def test_upload_invalid_file_type(client, mock_gridfs):
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

    # Should return error response, but still status 200
    assert response.status_code == 200
    # Upload should not be called for invalid file type
    assert not mock_gridfs.return_value.upload_from_stream.called


def test_no_file_uploaded(client, mock_gridfs):
    """Test submitting the form without a file."""
    response = client.post(
        "/final_image",
        data={},
        content_type="multipart/form-data"
    )

    # Should return an error response
    assert response.status_code == 400  # Bad request
    assert not mock_gridfs.return_value.upload_from_stream.called


def test_check_status(client):
    """Test the status check endpoint."""
    # Create a mock processing record in the database
    file_id = str(ObjectId())
    
    with patch('pymongo.collection.Collection.find_one') as mock_find:
        # Test with a "pending" status
        mock_find.return_value = {
            "_id": ObjectId(),
            "status": "pending",
            "input_file_id": ObjectId(file_id)
        }
        
        response = client.get(f"/check_status/{file_id}")
        assert response.status_code == 200
        assert b"pending" in response.data
        
        # Test with a "completed" status
        mock_find.return_value = {
            "_id": ObjectId(),
            "status": "completed",
            "input_file_id": ObjectId(file_id),
            "output_file_id": ObjectId()
        }
        
        response = client.get(f"/check_status/{file_id}")
        assert response.status_code == 200
        assert b"completed" in response.data


def test_get_image(client):
    """Test the get_image endpoint."""
    file_id = str(ObjectId())
    
    with patch('pymongo.collection.Collection.find_one') as mock_find:
        # Mock the database record
        mock_find.return_value = {
            "_id": ObjectId(),
            "status": "completed",
            "input_file_id": ObjectId(),
            "output_file_id": ObjectId(file_id),
            "filename": "test.jpg",
            "num_faces": 2,
            "processing_time": 1.5
        }
        
        response = client.get(f"/get_image/{file_id}")
        assert response.status_code == 200
        # Should render the result template
        assert b"Face Redaction Results" in response.data or b"result.html" in response.data


def test_image_data(client, mock_gridfs):
    """Test the image_data endpoint that streams GridFS data."""
    file_id = str(ObjectId())
    
    response = client.get(f"/image_data/{file_id}")
    
    # Should return the image data
    assert response.status_code == 200
    assert mock_gridfs.return_value.open_download_stream.called
    assert response.data == b"mocked image data"


def test_error_handler(client):
    """Test the error handler."""
    # Cause a deliberate exception by accessing a route that doesn't exist
    response = client.get("/nonexistent_route")

    # Should render the error template
    assert response.status_code == 404
    assert b"error" in response.data.lower()