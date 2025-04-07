# Face Redaction Client

This service automatically detects and redacts faces in images, storing the detection results in MongoDB.

## Features

- Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Custom redaction options: black rectangles (default) or custom image overlays
- Automatic monitoring of input directory for new images
- Processing and redaction of faces in images
- Storage of detection results in MongoDB (optional for local testing)
- Containerized for easy deployment

## Directory Structure

```
./
├── images/
│   ├── input/     # Where new images are placed for processing
│   ├── output/    # Where redacted images are saved
│   └── archive/   # Where original images are moved after processing
├── client.py      # Main application code
├── Dockerfile     # Container definition
├── .env           # Environment configuration
├── Pipfile        # Python dependencies
├── Pipfile.lock   # Locked dependencies
└── test_client.py # Unit tests
```

## Custom Face Redaction

By default, the client redacts faces by drawing black rectangles over them. However, you can use any image (like a mask, emoji, or other overlay) to redact faces by setting the `REDACTION_IMAGE` environment variable in the `.env` file:

```
REDACTION_IMAGE=images/redaction/fawkes.png
```

The redaction image should ideally have an alpha channel (transparency) for best results. The image will be automatically resized to fit each detected face.

## Configuration

The client can be configured using environment variables:

- `MONGO_URI`: MongoDB connection URI (default: `mongodb://admin:secret@mongodb:27017`)
- `MONGO_DBNAME`: MongoDB database name (default: `okaycooldb`)
- `INPUT_DIR`: Directory to watch for new images (default: `images/input`)
- `OUTPUT_DIR`: Directory to save redacted images (default: `images/output`)
- `ARCHIVE_DIR`: Directory to move processed images (default: `images/archive`)
- `POLL_INTERVAL`: Seconds between checking for new images (default: `5`)
- `REDACTION_IMAGE`: Path to image used for redaction (if not set, black rectangles will be used)

## Local Development and Testing

For local development and testing, you can run the client without MongoDB. The client will attempt to connect to MongoDB but will still process and redact images even if the connection fails.

To run the client locally:

```bash
# Install dependencies
pipenv install

# Activate the virtual environment
pipenv shell

# Run the client
python client.py
```

When running locally without MongoDB, you may see connection errors in the logs, but face detection and redaction will still work properly.

## Example Usage

1. Place an image with faces in the `images/input` directory (e.g., `protest.jpg`)
2. If using a custom redaction image, place it in an accessible location and update the `.env` file
3. Run the client
4. Redacted images will appear in the `images/output` directory (e.g., `protest_redacted.jpg`)
5. Original images will be moved to `images/archive` directory

## Docker Deployment

The client is designed to be run as a Docker container. When deployed with Docker Compose, it will connect to MongoDB automatically:

```bash
# Build and run with docker-compose
docker-compose up -d
```

## Testing

Run the unit tests using pytest:

```bash
pipenv run pytest test_client.py -v
```

Check test coverage:

```bash
pipenv run pytest --cov=client test_client.py
```

## Data Storage

When connected to MongoDB, the client stores the following data for each processed image:

```json
{
  "filename": "protest.jpg",
  "timestamp": "2023-04-07T12:34:56.789Z",
  "num_faces": 2,
  "confidence_scores": [0.998, 0.975],
  "processing_time": 1.25,
  "redaction_method": "image"  // or "rectangle"
}
``` 