# Face Redaction Client

This service automatically detects and redacts faces in images, storing the detection results in MongoDB.

## Features

- Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Automatic monitoring of input directory for new images
- Processing and redaction of faces in images
- Storage of detection results in MongoDB
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

## Usage

### Configuration

The client can be configured using environment variables:

- `MONGO_URI`: MongoDB connection URI (default: `mongodb://admin:secret@mongodb:27017`)
- `MONGO_DBNAME`: MongoDB database name (default: `okaycooldb`)
- `INPUT_DIR`: Directory to watch for new images (default: `images/input`)
- `OUTPUT_DIR`: Directory to save redacted images (default: `images/output`)
- `ARCHIVE_DIR`: Directory to move processed images (default: `images/archive`)
- `POLL_INTERVAL`: Seconds between checking for new images (default: `5`)

### Running with Docker

The client is designed to be run as a Docker container. You can use the following commands:

```bash
# Build the container
docker build -t face-redaction-client .

# Run the container
docker run -d \
  --name face_redaction_client \
  -e MONGO_URI=mongodb://admin:secret@mongodb:27017 \
  -e MONGO_DBNAME=okaycooldb \
  -v ./images:/app/images \
  face-redaction-client
```

### Running with Docker Compose

The preferred way to run the client is with Docker Compose, which will also start the MongoDB database:

```bash
docker-compose up -d
```

### Testing

Run the unit tests using pytest:

```bash
pytest test_client.py -v
```

Check test coverage:

```bash
pytest --cov=client test_client.py
```

## Data Storage

The client stores the following data in MongoDB:

```json
{
  "filename": "example.jpg",
  "timestamp": "2023-04-07T12:34:56.789Z",
  "num_faces": 2,
  "confidence_scores": [0.998, 0.975],
  "processing_time": 1.25
}
```

## Development

For development, you can install the dependencies using pipenv:

```bash
pipenv install
pipenv shell
```

And run the client directly:

```bash
python client.py
``` 