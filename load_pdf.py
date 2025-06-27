import requests
import json
import os

def load_pdf():
    url = "http://localhost:8000/load-pdf"
    
    # Get absolute path to data.pdf
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data.pdf")
    
    try:
        # Open and send the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            files = {'file': ('data.pdf', pdf_file, 'application/pdf')}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    load_pdf() 