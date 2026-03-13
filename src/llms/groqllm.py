from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        try:
            load_dotenv()
        except PermissionError:
            # In some sandboxed runners, reading `.env` can be disallowed.
            # The model key can still be provided via process env vars.
            pass


    def get_llm(self):
        try:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise ValueError("Missing GROQ_API_KEY. Set it in your environment (or a .env file) before calling this endpoint.")

            # Ensure downstream libraries can also read it from the environment.
            os.environ["GROQ_API_KEY"] = self.groq_api_key
            llm=ChatGroq(api_key=self.groq_api_key,model="llama-3.1-8b-instant")
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception: {e}")