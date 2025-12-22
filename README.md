# Face Mask Detection API

This repository serves a pre-trained face mask detection model via a FastAPI web service. The model classifies input images into one of three categories: with_mask, without_mask, or mask_weared_incorrect.

## Requirements

- Python 3.6+
- Virtual environment (recommended)

## Setup

1. Clone the repository.
2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI server:
```bash
python -m src.main
```

## API Usage

### POST /predict

Accepts an image file and returns the prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "prediction": "with_mask"
}
```
## Project Structure

* `src/` – FastAPI application code
   * `main.py` – Application factory and server entry point
   * `app/` – API routes
   * `pred/` – Model loading and inference logic
* `model/` – Trained model artifacts
   * `detector.pth` – PyTorch model (saved using `torch.save`)
   * `le.pickle` – Label encoder (saved using `pickle`)
* `pyimagesearch/` – Required model definition (`ObjectDetector` class) needed to deserialize the trained model
* `requirements.txt` – Python dependencies
