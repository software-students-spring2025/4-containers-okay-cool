"""Tests for the Face Redaction Client."""

import unittest
import os
import time
import base64
import io
import numpy as np
import cv2
from bson.objectid import ObjectId
import gridfs
import pymongo
from pymongo import MongoClient

from client import FaceRedactionClient

# MongoDB connection info for tests
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:secret@mongodb:27017")
TEST_DB_NAME = "test_face_redaction_db"

def get_minimal_jpg():
    """Return a minimal valid JPEG image."""
    # Minimal valid JPEG file (1x1 pixel)
    minimal_jpeg = base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wAALCAABAAEBAREA/8QAJgABAAAAAAAAAAAAAAAAAAAAAxABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAAPwBH/9k="
    )
    return io.BytesIO(minimal_jpeg)


class TestFaceRedactionClient(unittest.TestCase):
    """Test cases for the Face Redaction Client."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Connect to MongoDB - will raise an exception if connection fails
        cls.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        # Test connection
        cls.mongo_client.admin.command('ping')
        cls.db = cls.mongo_client[TEST_DB_NAME]
        
        # Set up GridFS buckets
        cls.input_bucket = gridfs.GridFSBucket(cls.db, bucket_name="input_images")
        cls.output_bucket = gridfs.GridFSBucket(cls.db, bucket_name="output_images")
        
        print("Connected to MongoDB for tests")

    def setUp(self):
        """Set up test fixtures."""
        # Clear test collections
        self.db = self.__class__.db
        self.db.image_processing.delete_many({})
        self.db.face_detection_results.delete_many({})
        self.db.input_images.files.delete_many({})
        self.db.input_images.chunks.delete_many({})
        self.db.output_images.files.delete_many({})
        self.db.output_images.chunks.delete_many({})
        
        # Create real client
        self.client = FaceRedactionClient(MONGO_URI, TEST_DB_NAME)

    def test_detect_faces(self):
        """Test face detection."""
        # Create a mock image with a face
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a face-like feature
        cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
        
        # Test detection directly
        faces = self.client.detect_faces(image)
        
        # We can't guarantee faces will be detected in a test image,
        # but we can verify the method returns the expected format
        self.assertIsInstance(faces, list)
        
        # If any faces were detected, verify their structure
        if faces:
            self.assertIn('box', faces[0])
            self.assertIn('confidence', faces[0])

    def test_redact_faces_rectangle(self):
        """Test face redaction with rectangles."""
        # Create a mock image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Set face area to white
        image[30:70, 30:70] = [255, 255, 255]
        
        # Mock detected faces
        faces = [{'box': [30, 30, 40, 40], 'confidence': 0.99}]
        
        # Test redaction
        redacted, num_faces = self.client.redact_faces(image, faces)
        
        # Verify
        self.assertEqual(num_faces, 1)
        # Check that the face area is now black (or redacted)
        self.assertTrue(np.any(redacted[30:70, 30:70] != [255, 255, 255]))

    def test_process_gridfs_images(self):
        """Test processing images from GridFS."""
        # Load a test JPEG image
        test_image = get_minimal_jpg()
            
        # Upload to GridFS input bucket
        test_image.seek(0)
        input_file_id = self.__class__.input_bucket.upload_from_stream(
            "test.jpg", 
            test_image.read()
        )
        
        # Create a processing record
        self.db.image_processing.insert_one({
            'input_file_id': input_file_id,
            'filename': 'test.jpg',
            'status': 'pending',
            'created_at': time.time()
        })
        
        # Process the images
        self.client.process_gridfs_images()
        
        # Verify a record was updated
        processed_record = self.db.image_processing.find_one({'input_file_id': input_file_id})
        self.assertIsNotNone(processed_record)
        self.assertEqual(processed_record['status'], 'completed')
        self.assertIn('output_file_id', processed_record)
        
        # Verify GridFS output exists
        output_exists = self.db.output_images.files.find_one(
            {'_id': processed_record['output_file_id']}
        )
        self.assertIsNotNone(output_exists)
        
        # Verify results were stored
        result = self.db.face_detection_results.find_one({'filename': 'test.jpg'})
        self.assertIsNotNone(result)

    def test_store_result(self):
        """Test storing results in MongoDB."""
        # Test data
        filename = 'test.jpg'
        num_faces = 3
        confidence_scores = [0.99, 0.98, 0.97]
        processing_time = 1.23
        
        # Run the method
        self.client.store_result(filename, num_faces, confidence_scores, processing_time)
        
        # Verify
        result = self.db.face_detection_results.find_one({'filename': filename})
        self.assertIsNotNone(result)
        self.assertEqual(result['filename'], filename)
        self.assertEqual(result['num_faces'], num_faces)
        self.assertEqual(result['confidence_scores'], confidence_scores)
        self.assertEqual(result['processing_time'], processing_time)


if __name__ == '__main__':
    unittest.main() 