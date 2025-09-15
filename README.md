
1. Create and activate venv:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
4. Set OpenAI key in .env  file :
   ```
    OPENAI_API_KEY="sk-..."
   ```
5. Run API:
   ```
   uvicorn app:app --reload --port 8000
   ```
6. Test:
   ```
   curl -F "file=@sample_inputs/sample1.txt" http://localhost:8000/process-text
   ```
