"""
Face Redaction Client

This script continuously processes images, detecting faces using MTCNN,
redacting them by filling in their boundary boxes, and storing the
analysis results in MongoDB.
"""

import io
import os
import sys
import time
import datetime
import logging
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from mtcnn import MTCNN
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import gridfs

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
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))  # seconds

# Redaction configuration
REDACTION_IMAGE = os.getenv("REDACTION_IMAGE", None)  # Path to image used for redaction


class FaceRedactionClient:
    """Client for detecting and redacting faces in images."""

    def __init__(
        self, mongo_uri: str, db_name: str, redaction_image_path: Optional[str] = None
    ) -> None:
        """
        Initialize the face redaction client.

        Args:
            mongo_uri: MongoDB connection URI
            db_name: MongoDB database name
            redaction_image_path: Path to the image to use for redacting faces.
                If None, black rectangles are used.
        """
        self.detector = MTCNN()

        # Attempt to connect to MongoDB but continue if it fails
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.processing_collection = None
        self.input_bucket = None
        self.output_bucket = None
        self.mongo_available = False

        try:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.mongo_client.admin.command("ping")
            self.db = self.mongo_client[db_name]
            self.collection = self.db.face_detection_results
            self.processing_collection = self.db.image_processing

            # Set up GridFS buckets
            self.input_bucket = gridfs.GridFSBucket(self.db, bucket_name="input_images")
            self.output_bucket = gridfs.GridFSBucket(
                self.db, bucket_name="output_images"
            )

            self.mongo_available = True
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(
                "MongoDB connection failed: %s. Exiting as GridFS is required.", str(e)
            )
            raise

        # Load redaction image if provided
        self.redaction_image = None
        self.redaction_method = "rectangle"

        if redaction_image_path:
            try:
                self.redaction_image = cv2.imread(
                    redaction_image_path, cv2.IMREAD_UNCHANGED
                )
                if self.redaction_image is not None:
                    logger.info(
                        "Using custom redaction image: %s", redaction_image_path
                    )
                    self.redaction_method = "image"
                else:
                    logger.warning(
                        "Could not load redaction image: %s. Using black rectangles instead.",
                        redaction_image_path,
                    )
            except Exception as e:
                logger.warning(
                    "Error loading redaction image: %s. Using black rectangles instead.",
                    str(e),
                )

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using MTCNN.

        Args:
            image: The image to detect faces in (RGB format)

        Returns:
            A list of dictionaries containing face detection results
        """
        faces = self.detector.detect_faces(image)
        return faces

    def redact_faces(
        self, image: np.ndarray, faces: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, int]:
        """
        Redact faces in an image.

        Args:
            image: The image to redact faces in (BGR format)
            faces: A list of dictionaries containing face detection results

        Returns:
            The redacted image and the number of faces redacted
        """
        redacted_image = image.copy()
        num_faces = 0

        for face in faces:
            # Get the bounding box
            x, y, width, height = face["box"]

            # Setting bounding box size to += 10%
            width = int(width * 1.1)
            height = int(height * 1.1)

            # Fix negative coordinates (sometimes MTCNN returns negative values)
            x = max(0, x)
            y = max(0, y)

            # Get the confidence score
            confidence = face["confidence"]

            # Skip low confidence detections
            if confidence < 0.9:
                continue

            num_faces += 1

            if self.redaction_method == "image" and self.redaction_image is not None:
                # Resize the redaction image to fit the face
                redaction_resized = cv2.resize(self.redaction_image, (width, height))

                # If the redaction image has an alpha channel, use it for blending
                if redaction_resized.shape[-1] == 4:
                    # Split the redaction image into color and alpha channels
                    redaction_rgb = redaction_resized[:, :, 0:3]
                    redaction_alpha = redaction_resized[:, :, 3] / 255.0

                    # Extract the region of interest from the original image
                    roi = redacted_image[y : y + height, x : x + width]

                    # Blend based on alpha
                    for c in range(0, 3):
                        roi[:, :, c] = (
                            roi[:, :, c] * (1 - redaction_alpha)
                            + redaction_rgb[:, :, c] * redaction_alpha
                        )

                    # Put the blended ROI back into the image
                    redacted_image[y : y + height, x : x + width] = roi
                else:
                    # If no alpha channel, just overlay the redaction image
                    redacted_image[y : y + height, x : x + width] = redaction_resized
            else:
                # Draw a filled black rectangle over the face
                cv2.rectangle(
                    redacted_image, (x, y), (x + width, y + height), (0, 0, 0), -1
                )

        return redacted_image, num_faces

    def store_result(
        self,
        filename: str,
        num_faces: int,
        confidence_scores: List[float],
        processing_time: float,
    ) -> None:
        """
        Store the face detection result in MongoDB.

        Args:
            filename: The name of the processed file
            num_faces: The number of faces detected
            confidence_scores: The confidence scores for each face
            processing_time: The time taken to process the image in seconds
        """
        if not self.mongo_available:
            return

        try:
            result = {
                "filename": filename,
                "timestamp": datetime.datetime.now(),
                "num_faces": num_faces,
                "confidence_scores": confidence_scores,
                "processing_time": processing_time,
            }

            self.collection.insert_one(result)
            logger.debug(
                "Stored detection result in MongoDB: %s faces in %s",
                num_faces,
                filename,
            )
        except Exception as e:
            logger.error("Failed to store detection result in MongoDB: %s", str(e))

    # pylint: disable=too-many-statements,too-many-nested-blocks
    def process_gridfs_images(self) -> None:
        """Process any pending images in GridFS."""
        try:
            # Find pending processing records
            pending_records = self.processing_collection.find({"status": "pending"})

            for record in pending_records:
                try:
                    start_time = time.time()

                    # Get the input file ID
                    input_file_id = record.get("input_file_id")
                    if not input_file_id:
                        continue

                    # Download the image from GridFS
                    logger.info("Processing GridFS image with ID: %s", input_file_id)
                    grid_out = self.input_bucket.open_download_stream(input_file_id)
                    image_data = grid_out.read()

                    # Convert to OpenCV format
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        raise ValueError("Failed to decode image data")

                    # Convert to RGB for MTCNN
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Check if we should use a custom redaction image for this request
                    custom_redaction_image = None
                    orig_redaction_image = self.redaction_image
                    orig_redaction_method = self.redaction_method

                    if record.get("has_custom_cover") and record.get("cover_image_id"):
                        try:
                            # Download the custom cover image
                            cover_grid_out = self.input_bucket.open_download_stream(
                                record["cover_image_id"]
                            )
                            cover_data = cover_grid_out.read()

                            # Convert to OpenCV format
                            cover_nparr = np.frombuffer(cover_data, np.uint8)
                            custom_redaction_image = cv2.imdecode(
                                cover_nparr, cv2.IMREAD_UNCHANGED
                            )

                            if custom_redaction_image is not None:
                                # Temporarily change the redaction settings
                                self.redaction_image = custom_redaction_image
                                self.redaction_method = "image"
                                logger.info(
                                    "Using custom redaction image for this request"
                                )
                        except Exception as e:
                            logger.error(
                                "Error loading custom redaction image: %s", str(e)
                            )

                    # Detect faces
                    faces = self.detect_faces(img_rgb)

                    # Redact faces
                    img_redacted, num_faces = self.redact_faces(img, faces)

                    # Encode the image to bytes
                    is_jpg = (
                        record.get("filename", "").lower().endswith((".jpg", ".jpeg"))
                    )
                    ext = ".jpg" if is_jpg else ".png"
                    success, redacted_bytes = cv2.imencode(ext, img_redacted)

                    if not success:
                        raise ValueError("Failed to encode redacted image")

                    # Upload to GridFS
                    filename = record.get("filename", "unknown")
                    name, ext = os.path.splitext(filename)
                    output_filename = f"{name}_redacted{ext}"

                    output_file_id = self.output_bucket.upload_from_stream(
                        output_filename,
                        io.BytesIO(redacted_bytes),
                        metadata={
                            "input_file_id": input_file_id,
                            "num_faces": num_faces,
                            "processing_time": time.time() - start_time,
                        },
                    )

                    # Calculate processing time
                    processing_time = time.time() - start_time

                    # Extract confidence scores
                    confidence_scores = (
                        [face["confidence"] for face in faces] if faces else []
                    )

                    # Update the processing record
                    self.processing_collection.update_one(
                        {"_id": record["_id"]},
                        {
                            "$set": {
                                "status": "completed",
                                "output_file_id": output_file_id,
                                "num_faces": num_faces,
                                "confidence_scores": confidence_scores,
                                "processing_time": processing_time,
                                "completed_at": time.time(),
                            }
                        },
                    )

                    # Store result in face_detection_results collection too
                    self.store_result(
                        filename, num_faces, confidence_scores, processing_time
                    )

                    logger.info(
                        "Processed GridFS image %s: %s faces, %.2fs",
                        filename,
                        num_faces,
                        processing_time,
                    )

                    # Restore original redaction settings if changed for this image
                    if custom_redaction_image is not None:
                        self.redaction_image = orig_redaction_image
                        self.redaction_method = orig_redaction_method

                except Exception as e:
                    logger.error("Error processing GridFS image: %s", str(e))

                    # Update processing record to show failure
                    self.processing_collection.update_one(
                        {"_id": record["_id"]},
                        {
                            "$set": {
                                "status": "failed",
                                "error": str(e),
                                "completed_at": time.time(),
                            }
                        },
                    )

        except Exception as e:
            logger.error("Error in GridFS processing: %s", str(e))

    def run(self) -> None:
        """Run the face redaction client in continuous mode."""
        logger.info("Starting face redaction client")

        while True:
            # Process images from GridFS only
            if self.mongo_available:
                self.process_gridfs_images()
            else:
                logger.error("MongoDB not available, cannot process images")
                time.sleep(60)  # Wait longer when there's a connection issue

            # Wait before checking for new images
            time.sleep(POLL_INTERVAL)


def main() -> None:
    """Main entry point for the face redaction client."""
    try:
        client = FaceRedactionClient(MONGO_URI, MONGO_DBNAME, REDACTION_IMAGE)
        client.run()
    except Exception as e:
        logger.critical("Fatal error initializing face redaction client: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
