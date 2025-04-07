"""
Face Redaction Client

This script continuously processes images, detecting faces using MTCNN,
redacting them by filling in their boundary boxes, and storing the
analysis results in MongoDB.
"""

import os
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from mtcnn import MTCNN
from pymongo import MongoClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("face-redaction-client")

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:secret@mongodb:27017")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "okaycooldb")

# Image processing configuration
INPUT_DIR = os.getenv("INPUT_DIR", "images/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "images/output")
ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "images/archive")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))  # seconds

# Create directories if they don't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)


class FaceRedactionClient:
    """Client for detecting and redacting faces in images."""

    def __init__(self, mongo_uri: str, db_name: str) -> None:
        """
        Initialize the face redaction client.

        Args:
            mongo_uri: MongoDB connection URI
            db_name: MongoDB database name
        """
        self.detector = MTCNN()
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.collection = self.db.face_detection_results
        logger.info("Face Redaction Client initialized")

    def detect_faces(self, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an RGB image using MTCNN.

        Args:
            img_rgb: RGB image as numpy array

        Returns:
            List of detected faces with their bounding boxes and landmarks
        """
        faces = self.detector.detect_faces(img_rgb)
        return faces

    def redact_faces(
        self, img: np.ndarray, faces: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, int]:
        """
        Redact faces in the image by drawing filled rectangles over them.

        Args:
            img: Original image as numpy array
            faces: List of detected faces with their bounding boxes

        Returns:
            Tuple of (redacted image, number of faces)
        """
        img_with_patches = img.copy()
        num_faces = len(faces)

        for face in faces:
            x, y, w, h = face["box"]
            # Draw filled rectangle to redact the face
            img_with_patches = cv2.rectangle(
                img_with_patches, (x, y), (x + w, y + h), (0, 0, 0), -1  # Black fill
            )

        return img_with_patches, num_faces

    def store_result(
        self,
        filename: str,
        num_faces: int,
        confidence_scores: List[float],
        processing_time: float,
    ) -> None:
        """
        Store face detection result in MongoDB.

        Args:
            filename: Name of the processed image file
            num_faces: Number of faces detected
            confidence_scores: Confidence scores for each detected face
            processing_time: Time taken to process the image in seconds
        """
        result = {
            "filename": filename,
            "timestamp": datetime.datetime.now(),
            "num_faces": num_faces,
            "confidence_scores": confidence_scores,
            "processing_time": processing_time,
        }
        self.collection.insert_one(result)
        logger.info("Stored result for %s: %s faces detected", filename, num_faces)

    def process_image(self, image_path: str) -> Optional[str]:
        """
        Process an image by detecting and redacting faces.

        Args:
            image_path: Path to the input image

        Returns:
            Path to the redacted image or None if processing failed
        """
        try:
            start_time = time.time()

            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to read image: %s", image_path)
                return None

            # Convert to RGB for MTCNN
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = self.detect_faces(img_rgb)

            # Redact faces
            img_redacted, num_faces = self.redact_faces(img, faces)

            # Generate output filename
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(OUTPUT_DIR, f"{name}_redacted{ext}")
            archive_path = os.path.join(ARCHIVE_DIR, base_name)

            # Save redacted image
            cv2.imwrite(output_path, img_redacted)

            # Move original image to archive
            os.rename(image_path, archive_path)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract confidence scores
            confidence_scores = [face["confidence"] for face in faces] if faces else []

            # Store result in MongoDB
            self.store_result(base_name, num_faces, confidence_scores, processing_time)

            logger.info(
                "Processed %s: %s faces, %.2fs", base_name, num_faces, processing_time
            )
            return output_path

        except Exception as e:
            logger.error("Error processing %s: %s", image_path, str(e))
            return None

    def run(self) -> None:
        """Run the face redaction client in continuous mode."""
        logger.info("Starting face redaction client")

        while True:
            # Get all image files in input directory
            image_files = []
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.extend(list(Path(INPUT_DIR).glob(f"*{ext}")))
                image_files.extend(list(Path(INPUT_DIR).glob(f"*{ext.upper()}")))

            # Process each image
            for image_path in image_files:
                self.process_image(str(image_path))

            # Wait before checking for new images
            time.sleep(POLL_INTERVAL)


def main() -> None:
    """Main entry point for the face redaction client."""
    client = FaceRedactionClient(MONGO_URI, MONGO_DBNAME)
    client.run()


if __name__ == "__main__":
    main()
