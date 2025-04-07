"""
Unit tests for the face redaction client.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from client import FaceRedactionClient


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client for testing."""
    with patch("client.MongoClient") as mock_client:
        # Set up mock collection
        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_db.face_detection_results = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        yield mock_client


@pytest.fixture
def test_image():
    """Create a test image for face detection."""
    # Create a simple test image (100x100 with a grey square in the middle)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 128  # Grey square in the middle
    return img


@pytest.fixture
def test_redaction_image():
    """Create a test redaction image."""
    # Create a simple test image (50x50 red square with alpha channel)
    img = np.zeros((50, 50, 4), dtype=np.uint8)
    img[:, :, 0] = 0    # B
    img[:, :, 1] = 0    # G
    img[:, :, 2] = 255  # R
    img[:, :, 3] = 128  # Alpha (semi-transparent)
    return img


@pytest.fixture
def temp_directories():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    archive_dir = os.path.join(temp_dir, "archive")
    
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    os.makedirs(archive_dir)
    
    # Patch environment variables
    with patch.dict("os.environ", {
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": output_dir,
        "ARCHIVE_DIR": archive_dir
    }):
        yield input_dir, output_dir, archive_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_init(mock_mongo_client):
    """Test initialization of FaceRedactionClient."""
    # Test without redaction image
    client = FaceRedactionClient("mongodb://test", "test_db")
    
    assert client.detector is not None
    mock_mongo_client.assert_called_once_with("mongodb://test")
    assert client.db == mock_mongo_client.return_value.__getitem__.return_value
    assert client.collection == client.db.face_detection_results
    assert client.redaction_image is None
    assert client.redaction_method == "rectangle"
    
    # Reset mock
    mock_mongo_client.reset_mock()
    
    # Test with invalid redaction image
    client = FaceRedactionClient("mongodb://test", "test_db", "nonexistent.png")
    assert client.redaction_image is None
    assert client.redaction_method == "rectangle"


def test_detect_faces():
    """Test face detection."""
    with patch("client.MTCNN") as mock_mtcnn:
        # Set up mock detector
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [
            {"box": [10, 10, 50, 50], "confidence": 0.99}
        ]
        mock_mtcnn.return_value = mock_detector
        
        # Initialize client with mock detector
        client = FaceRedactionClient("mongodb://test", "test_db")
        
        # Test detect_faces method
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = client.detect_faces(img)
        
        assert len(faces) == 1
        assert faces[0]["box"] == [10, 10, 50, 50]
        assert faces[0]["confidence"] == 0.99
        mock_detector.detect_faces.assert_called_once()


def test_redact_faces_rectangle(test_image):
    """Test face redaction with rectangle method."""
    client = FaceRedactionClient("mongodb://test", "test_db")
    
    # Create test faces
    faces = [
        {"box": [10, 10, 30, 30], "confidence": 0.99},
        {"box": [50, 50, 30, 30], "confidence": 0.95}
    ]
    
    # Redact faces
    redacted_img, num_faces = client.redact_faces(test_image, faces)
    
    # Check that the number of faces is correct
    assert num_faces == 2
    
    # Check that the image was modified (not the same as the original)
    assert not np.array_equal(test_image, redacted_img)


def test_redact_faces_image(test_image, test_redaction_image):
    """Test face redaction with image method."""
    with patch("cv2.imread") as mock_imread:
        mock_imread.return_value = test_redaction_image
        
        client = FaceRedactionClient("mongodb://test", "test_db", "fake_path.png")
        client.redaction_image = test_redaction_image
        client.redaction_method = "image"
        
        # Create test faces
        faces = [
            {"box": [10, 10, 30, 30], "confidence": 0.99},
        ]
        
        # Redact faces
        redacted_img, num_faces = client.redact_faces(test_image, faces)
        
        # Check that the number of faces is correct
        assert num_faces == 1
        
        # Check that the image was modified (not the same as the original)
        assert not np.array_equal(test_image, redacted_img)


def test_store_result(mock_mongo_client):
    """Test storing results in MongoDB."""
    client = FaceRedactionClient("mongodb://test", "test_db")
    
    # Test store_result method
    client.store_result("test.jpg", 2, [0.99, 0.95], 0.5)
    
    # Check that insert_one was called with the right data
    mock_collection = client.collection
    mock_collection.insert_one.assert_called_once()
    call_args = mock_collection.insert_one.call_args[0][0]
    
    assert call_args["filename"] == "test.jpg"
    assert call_args["num_faces"] == 2
    assert call_args["confidence_scores"] == [0.99, 0.95]
    assert call_args["processing_time"] == 0.5
    assert call_args["redaction_method"] == "rectangle"


@patch("client.cv2.imread")
@patch("client.cv2.cvtColor")
@patch("client.cv2.imwrite")
@patch("client.os.rename")
def test_process_image(mock_rename, mock_imwrite, mock_cvtcolor, mock_imread, mock_mongo_client):
    """Test the process_image method."""
    # Setup
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Initialize client with a mock detector
    with patch.object(FaceRedactionClient, "detect_faces") as mock_detect:
        mock_detect.return_value = [
            {"box": [10, 10, 30, 30], "confidence": 0.99}
        ]
        
        client = FaceRedactionClient("mongodb://test", "test_db")
        
        # Mock store_result to avoid actual DB calls
        with patch.object(client, "store_result") as mock_store:
            # Test process_image
            result = client.process_image("test_path/test.jpg")
            
            # Verify method calls
            mock_imread.assert_called_once_with("test_path/test.jpg")
            mock_cvtcolor.assert_called_once()
            mock_detect.assert_called_once()
            mock_imwrite.assert_called_once()
            mock_rename.assert_called_once()
            mock_store.assert_called_once()
            
            # Check that the result is the output path
            assert result is not None


def test_run(temp_directories, mock_mongo_client):
    """Test the run method."""
    input_dir, _, _ = temp_directories
    
    # Create a test image in the input directory
    test_img_path = os.path.join(input_dir, "test.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, img)
    
    # Initialize client with patched INPUT_DIR
    with patch("client.INPUT_DIR", input_dir):
        client = FaceRedactionClient("mongodb://test", "test_db")
        
        # Mock process_image to avoid actual processing
        with patch.object(client, "process_image") as mock_process:
            mock_process.return_value = "output/test_redacted.jpg"
            
            # Mock time.sleep to exit after one iteration
            with patch("client.time.sleep", side_effect=KeyboardInterrupt):
                try:
                    client.run()
                except KeyboardInterrupt:
                    pass
                
                # Verify the process_image was called
                mock_process.assert_called_once()
                assert test_img_path in mock_process.call_args[0][0] 