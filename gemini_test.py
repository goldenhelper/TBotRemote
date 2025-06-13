from google import genai

client = genai.Client(api_key='AIzaSyB6asNMtCnN1ZnoBesuUcuPB4R52cTfaPQ', http_options={'api_version':'v1alpha'})

response = client.models.generate_content(
    model='gemini-2.5-flash-preview-05-20',
    contents='Explain how RLHF works in simple terms.',
)

print(response.text)