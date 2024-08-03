# myapp/views.py

import os
from django.shortcuts import render, redirect
from django.conf import settings
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import imutils

def index(request):
    # Render the main page with model selection
    return render(request, 'myapp/index.html')

def detect_text_google_vision(request):
    # Set the environment variable for authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(settings.BASE_DIR, 'vision.json')

    # Define directories for image storage
    upload_dir = os.path.join(settings.BASE_DIR, 'static/images/uploaded')
    processed_dir = os.path.join(settings.BASE_DIR, 'processed_images/google_vision')

    # Ensure directories exist
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Handle image upload
    if request.method == 'POST' and 'image_upload' in request.FILES:
        image_file = request.FILES['image_upload']
        img_loc = os.path.join(upload_dir, image_file.name)
        with open(img_loc, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        return redirect('detect_text_google_vision')

    # Get the selected image from query parameters
    selected_image = request.GET.get('selected_image', None)
    detected_text = ''
    encoded_img = ''

    if selected_image:
        img_loc = os.path.join(upload_dir, selected_image)

        if os.path.exists(img_loc):
            # Initialize the Vision API client
            client = vision.ImageAnnotatorClient()

            with open(img_loc, 'rb') as image_file:
                content = image_file.read()

            # Perform text detection on the image
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations

            if texts:
                # Get the full detected text
                detected_text = texts[0].description  # The first element contains the full text

                # Load the image for drawing
                img = Image.open(img_loc)
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()

                # Draw bounding boxes for each detected word
                for text in texts[1:]:  # Skip the first element, which contains full text
                    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                    draw.line(vertices + [vertices[0]], width=2, fill='red')

                # Save and encode the processed image
                processed_img_path = os.path.join(processed_dir, 'processed_' + selected_image)
                img.save(processed_img_path)

                with open(processed_img_path, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

    # Get available images for selection
    available_images = os.listdir(upload_dir)

    # Context for rendering the template
    context = {
        'detected_text': detected_text,
        'processed_image_base64': encoded_img,
        'available_images': available_images,
        'selected_image': selected_image
    }

    return render(request, 'myapp/google_vision.html', context)



def detect_text_custom_model(request):
    # Directories for image storage
    upload_dir = os.path.join(settings.BASE_DIR, 'static/images/uploaded')
    processed_dir = os.path.join(settings.BASE_DIR, 'processed_images/custom_model')

    # Ensure directories exist
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Handle image upload
    if request.method == 'POST' and 'image_upload' in request.FILES:
        image_file = request.FILES['image_upload']
        img_loc = os.path.join(upload_dir, image_file.name)
        with open(img_loc, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        return redirect('detect_text_custom_model')

    selected_image = request.GET.get('selected_image', None)
    detected_text = ''
    encoded_img = ''

    if selected_image:
        img_loc = os.path.join(upload_dir, selected_image)

        if os.path.exists(img_loc):
            # Path to the custom handwriting model
            model_path = os.path.join(settings.BASE_DIR, 'myapp', 'handwriting.model') # Ensure this path is correct

            # Load the handwriting OCR model
            try:
                model = load_model(model_path)
                print("[INFO] Handwriting model loaded successfully")
            except Exception as e:
                error_message = f"Error loading model: {str(e)}"
                print(error_message)
                return render(request, 'myapp/error.html', {'error_message': error_message})

            # Load and preprocess the input image
            image = cv2.imread(img_loc)

            # Convert it to grayscale and blur it to reduce noise
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Perform adaptive thresholding to highlight the text
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Perform edge detection
            edged = cv2.Canny(binary, 30, 150)

            # Find contours and sort from left-to-right
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="left-to-right")[0]

            # Initialize the list of contour bounding boxes and associated characters
            chars = []

            # Loop over the contours
            for c in cnts:
                # Compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)

                # Use more precise filtering conditions for bounding boxes
                if (w >= 15 and h >= 30) and (w / h < 1.5):
                    # Extract the character and threshold it
                    roi = gray[y:y + h, x:x + w]
                    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                    # Resize the character to 32x32 pixels
                    thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)

                    # Normalize the character image
                    padded = thresh.astype("float32") / 255.0
                    padded = np.expand_dims(padded, axis=-1)

                    # Update our list of characters that will be OCR'd
                    chars.append((padded, (x, y, w, h)))

            # Batch process the characters for predictions
            if chars:
                chars_np = np.array([c[0] for c in chars], dtype="float32")
                preds = model.predict(chars_np)

                # Define the list of label names
                labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                labelNames = [l for l in labelNames]

                # Extract text and draw bounding boxes for each character
                for (pred, (x, y, w, h)) in zip(preds, [b[1] for b in chars]):
                    # Find the index of the label with the largest probability
                    i = np.argmax(pred)
                    label = labelNames[i]

                    # Add the label to the detected text
                    detected_text += label

                    # Draw the bounding box and label on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save and encode the processed image for display
            processed_img_path = os.path.join(processed_dir, 'processed_' + selected_image)
            cv2.imwrite(processed_img_path, image)

            # Convert image to base64 to render on HTML
            with open(processed_img_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

    available_images = os.listdir(upload_dir)

    context = {
        'detected_text': detected_text,
        'processed_image_base64': encoded_img,
        'available_images': available_images,
        'selected_image': selected_image
    }

    return render(request, 'myapp/custom_model.html', context)