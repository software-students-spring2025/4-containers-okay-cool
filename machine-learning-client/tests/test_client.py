"""Tests for the Face Redaction Client."""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import cv2
import io
from bson.objectid import ObjectId

from client import FaceRedactionClient


class TestFaceRedactionClient(unittest.TestCase):
    """Test cases for the Face Redaction Client."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock MongoDB connection
        self.mongo_patch = patch('client.MongoClient')
        self.mock_mongo = self.mongo_patch.start()
        
        # Mock GridFS
        self.mock_db = MagicMock()
        self.mock_mongo.return_value.__getitem__.return_value = self.mock_db
        
        # Mock GridFS buckets
        self.mock_input_bucket = MagicMock()
        self.mock_output_bucket = MagicMock()
        self.gridfs_patch = patch('client.gridfs.GridFSBucket')
        self.mock_gridfs = self.gridfs_patch.start()
        self.mock_gridfs.side_effect = [self.mock_input_bucket, self.mock_output_bucket]
        
        # Create client with mocked MongoDB
        self.client = FaceRedactionClient('mongodb://localhost', 'test_db')
        
        # Set up collections
        self.client.collection = MagicMock()
        self.client.processing_collection = MagicMock()

    def tearDown(self):
        """Tear down test fixtures."""
        self.mongo_patch.stop()
        self.gridfs_patch.stop()

    def test_detect_faces(self):
        """Test face detection."""
        # Create a mock image with a face
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a face-like feature
        cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
        
        # Mock MTCNN detector
        self.client.detector.detect_faces = MagicMock(return_value=[
            {'box': [30, 30, 40, 40], 'confidence': 0.99}
        ])
        
        # Test detection
        faces = self.client.detect_faces(image)
        
        # Verify
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]['box'], [30, 30, 40, 40])
        self.assertEqual(faces[0]['confidence'], 0.99)
        self.client.detector.detect_faces.assert_called_once()

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
        # Check that the face area is now black
        self.assertTrue(np.all(redacted[30:70, 30:70] == [0, 0, 0]))

    def test_process_gridfs_images(self):
        """Test processing images from GridFS."""
        # Mock a processing record
        mock_record = {
            '_id': ObjectId(),
            'input_file_id': ObjectId(),
            'filename': 'test.jpg',
            'status': 'pending'
        }
        
        # Set up mocks
        self.client.processing_collection.find.return_value = [mock_record]
        
        # Mock the GridFS download stream
        mock_grid_out = MagicMock()
        # Create a small test image as bytes
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (30, 30), (70, 70), (255, 255, 255), -1)
        _, img_bytes = cv2.imencode('.jpg', test_img)
        mock_grid_out.read.return_value = img_bytes.tobytes()
        
        self.mock_input_bucket.open_download_stream.return_value = mock_grid_out
        
        # Mock the face detection
        self.client.detect_faces = MagicMock(return_value=[
            {'box': [30, 30, 40, 40], 'confidence': 0.99}
        ])
        
        # Mock the GridFS upload
        self.mock_output_bucket.upload_from_stream.return_value = ObjectId()
        
        # Run the method
        self.client.process_gridfs_images()
        
        # Verify
        self.mock_input_bucket.open_download_stream.assert_called_once_with(mock_record['input_file_id'])
        self.client.detect_faces.assert_called_once()
        self.mock_output_bucket.upload_from_stream.assert_called_once()
        self.client.processing_collection.update_one.assert_called_once()
        self.client.collection.insert_one.assert_called_once()

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
        self.client.collection.insert_one.assert_called_once()
        args = self.client.collection.insert_one.call_args[0][0]
        self.assertEqual(args['filename'], filename)
        self.assertEqual(args['num_faces'], num_faces)
        self.assertEqual(args['confidence_scores'], confidence_scores)
        self.assertEqual(args['processing_time'], processing_time)


if __name__ == '__main__':
    unittest.main() 