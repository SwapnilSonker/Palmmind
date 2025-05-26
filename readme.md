# Callback Request App

A simple Streamlit web app that collects callback requests with validation, leveraging PyTorch and other ML components.

---

## Features

- Input form for collecting name, phone, email, and preferred callback time.
- Validation of phone numbers, emails, and time format using Pydantic validators.
- Integration with PyTorch-based models (e.g., for NER or other tasks).
- Streamlit frontend for fast prototyping and deployment.

---

## Prerequisites

- Python 3.8+
- Streamlit
- PyTorch

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SwapnilSonker/Palmmind.git
   
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt


streamlit run app.py --server.runOnSave=false
Note: The --server.runOnSave=false flag disables Streamlit’s auto-reloader to avoid errors related to PyTorch internal modules during development.