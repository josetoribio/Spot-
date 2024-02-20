import base64
import requests

# OpenAI API Key
#change this when uploading to github !!!!


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/home/jose/Downloads/hand_color_image2.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "make the outputt only the answer. What’s the tittle of the book?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

y = response.text

#used to get content seperated 
#index into class instance *response*
y = y.split(',')
y = y[8].split('"')
y = y[3]

text_file = open("Output.txt", "w")
text_file.write(y)
text_file.close()

file = open("Output.txt", "r")
content = file.read()
file.close()
print(content)


