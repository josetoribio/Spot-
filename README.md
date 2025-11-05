# 🧠 Image Fetch & Viewer

This project combines **YOLOv8 object detection** with **OpenAI’s GPT-4 Vision API** to enable visual perception and manipulation for a robotic system.

---

## 🎥 Demo Video

▶️ [Watch the demo on YouTube](https://www.youtube.com/watch?v=jwCCXasD81w)

[![YouTube Video](https://img.youtube.com/vi/jwCCXasD81w/0.jpg)](https://www.youtube.com/watch?v=jwCCXasD81w)

---

## 🚀 Overview

### 🟢 Image Fetch
- Uses **YOLOv8** for real-time **object detection**  
- Draws **bounding boxes** around detected objects  
- Computes the **center coordinates** of each object — ideal for robotic grasping and targeting tasks  

### 🖼️ Image Viewer
- Displays live YOLOv8 detections from **all robot cameras**  
- Useful for debugging, visualization, and multi-camera monitoring  

---

## 🧩 OpenAI Vision Integration

A sample Python script demonstrates how to send an image to the **GPT-4 Vision model** for analysis.  
It encodes an image in Base64, sends it to OpenAI’s API, and extracts the text output.

> ⚠️ **Important:**  
> Replace `api_key` with your **own OpenAI API key**.  
> **Never commit your API key** to GitHub — use environment variables or `.env` files instead.

---

## 💻 Example Code

```python
import base64
import requests

# 🔑 Set your OpenAI API key (do not hardcode this when uploading to GitHub)
api_key = "YOUR_API_KEY_HERE"

# Encode an image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "/home/jose/Downloads/hand_color_image2.jpg"
base64_image = encode_image(image_path)

# Prepare API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s the title of the book?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ],
    "max_tokens": 300,
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Extract output text from the response
result_text = response.json()["choices"][0]["message"]["content"]

# Save and print result
with open("Output.txt", "w") as f:
    f.write(result_text)

print(result_text)
