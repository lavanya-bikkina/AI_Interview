import google.generativeai as genai
genai.configure(api_key="AIzaSyD7tc1ImpzbyODtAK5lHcrOn2EXVYEU-aU")
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Explain Machine Learning in simple terms.")
print(response.text)
