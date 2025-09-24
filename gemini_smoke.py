import os
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

PROJECT  = os.getenv("VERTEX_PROJECT_ID", "")
LOCATION = "us-central1"
KEY_PATH = "service-account.json"

creds = service_account.Credentials.from_service_account_file(KEY_PATH)
vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)

# 1) Chat test
chat = GenerativeModel("gemini-2.5-flash-lite")
resp = chat.generate_content("Reply with OK if you can read this.")
print("Chat:", resp.text.strip())

# 2) Embedding test
emb = TextEmbeddingModel.from_pretrained("text-embedding-004")
vec = emb.get_embeddings(["malaria cases in 2025"])[0].values
print("Embedding dims:", len(vec))
