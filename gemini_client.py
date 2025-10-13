import google.generativeai as genai

genai.configure(api_key="GEMINI_API_KEY")  # Replace with your actual Gemini API key

model = genai.GenerativeModel("gemini-pro")

def generate_summary(prompt, document_text):
    full_prompt = f"{prompt}\n\nDocument:\n{document_text}"
    response = model.generate_content(full_prompt)
    return response.text
