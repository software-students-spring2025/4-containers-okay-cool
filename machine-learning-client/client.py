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
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
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

# Redaction configuration
REDACTION_IMAGE = os.getenv("REDACTION_IMAGE", None)  # Path to image used for redaction

# Create directories if they don't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)


class FaceRedactionClient:
    """Client for detecting and redacting faces in images."""

    def __init__(self, mongo_uri: str, db_name: str, redaction_image_path: Optional[str] = None) -> None:
        """
        Initialize the face redaction client.

        Args:
            mongo_uri: MongoDB connection URI
            db_name: MongoDB database name
            redaction_image_path: Path to the image to use for redacting faces. If None, black rectangles are used.
        """
        self.detector = MTCNN()
        
        # Attempt to connect to MongoDB but continue if it fails
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.mongo_available = False
        
        try:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[db_name]
            self.collection = self.db.face_detection_results
            self.mongo_available = True
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning("MongoDB connection failed: %s. Continuing without database functionality.", str(e))
        
        # Load redaction image if provided
        self.redaction_image = None
        self.redaction_method = "rectangle"
        
        if redaction_image_path:
            try:
                self.redaction_image = cv2.imread(redaction_image_path, cv2.IMREAD_UNCHANGED)
                if self.redaction_image is None:
                    logger.error("Failed to load redaction image: %s. Using rectangle redaction instead.", redaction_image_path)
                else:
                    logger.info("Loaded redaction image: %s", redaction_image_path)
                    self.redaction_method = "image"
            except Exception as e:
                logger.error("Error loading redaction image: %s. Using rectangle redaction instead.", str(e))
        
        logger.info("Face Redaction Client initialized with method: %s", self.redaction_method)

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

    def apply_rectangle_redaction(
        self, image: np.ndarray, face: Dict[str, Any], color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Redact a face by drawing a filled rectangle over it.
        
        Args:
            image: Original image
            face: Face detection result with bounding box
            color: BGR color tuple for the redaction rectangle
            
        Returns:
            Image with face redacted by rectangle
        """
        x, y, w, h = face["box"]
        return cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)  # -1 for fill

    def apply_image_redaction(
        self, image: np.ndarray, face: Dict[str, Any], redaction_img: np.ndarray
    ) -> np.ndarray:
        """
        Redact a face by placing another image over it.
        
        Args:
            image: Original image
            face: Face detection result with bounding box
            redaction_img: Image to place over the face
            
        Returns:
            Image with face redacted by overlaying another image
        """
        x, y, w, h = face["box"]
        result = image.copy()
        
        # Resize redaction image to match face size
        overlay = cv2.resize(redaction_img, (w, h))
        
        # Handle both 3-channel (BGR) and 4-channel (BGRA) redaction images
        if overlay.shape[2] == 4:  # With alpha channel
            # Get the alpha channel
            alpha = overlay[:, :, 3] / 255.0
            
            # Extract RGB channels from overlay
            overlay_rgb = overlay[:, :, :3]
            
            # Apply alpha blending
            for c in range(3):  # Apply for each color channel
                result[y:y+h, x:x+w, c] = (
                    overlay_rgb[:, :, c] * alpha + 
                    result[y:y+h, x:x+w, c] * (1 - alpha)
                ).astype(np.uint8)
        else:  # Without alpha channel (BGR)
            result[y:y+h, x:x+w] = overlay
            
        return result

    def redact_faces(
        self, img: np.ndarray, faces: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, int]:
        """
        Redact faces in the image using the configured method.

        Args:
            img: Original image as numpy array
            faces: List of detected faces with their bounding boxes

        Returns:
            Tuple of (redacted image, number of faces)
        """
        if not faces:
            return img.copy(), 0
            
        img_with_redactions = img.copy()
        num_faces = len(faces)

        for face in faces:
            # Apply the appropriate redaction method
            if self.redaction_method == "image" and self.redaction_image is not None:
                img_with_redactions = self.apply_image_redaction(
                    img_with_redactions, face, self.redaction_image
                )
            else:  # Default to rectangle redaction
                img_with_redactions = self.apply_rectangle_redaction(img_with_redactions, face)

        return img_with_redactions, num_faces

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
        if not self.mongo_available:
            logger.info("MongoDB not available. Skipping data storage for %s", filename)
            return
            
        try:
            result = {
                "filename": filename,
                "timestamp": datetime.datetime.now(),
                "num_faces": num_faces,
                "confidence_scores": confidence_scores,
                "processing_time": processing_time,
                "redaction_method": self.redaction_method,
            }
            self.collection.insert_one(result)
            logger.info("Stored result for %s: %s faces detected", filename, num_faces)
        except Exception as e:
            logger.error("Failed to store result in MongoDB: %s", str(e))

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

            # Store result in MongoDB (if available)
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
                if os.path.basename(image_path) != os.path.basename(REDACTION_IMAGE or ""):
                    self.process_image(str(image_path))

            # Wait before checking for new images
            time.sleep(POLL_INTERVAL)


def main() -> None:
    """Main entry point for the face redaction client."""
    client = FaceRedactionClient(MONGO_URI, MONGO_DBNAME, REDACTION_IMAGE)
    client.run()


if __name__ == "__main__":
    main()
