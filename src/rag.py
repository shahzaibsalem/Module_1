import os
import re
import spacy
from typing import List
from dotenv import load_dotenv

from prompt_loader import load_prompt_from_yaml
from paths import folder_path
from vectordb import VectorDB

from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env variables
load_dotenv()


# ---------------------------------------------------------
# DOCUMENT LOADER
# ---------------------------------------------------------

def load_documents() -> List[dict]:
    results = []
    absolute_folder = os.path.abspath(folder_path)

    for filename in os.listdir(absolute_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(absolute_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = f.read()
                results.append({
                    "content": data,
                    "metadata": {"source": filename}
                })

    return results


# ---------------------------------------------------------
# RAG ASSISTANT
# ---------------------------------------------------------

class RAGAssistant:

    def __init__(self):
        print("Initializing RAG Assistant...")

        self.llm = self._initialize_llm()
        self.vector_db = VectorDB()
        self.prompt_template = load_prompt_from_yaml("prompt_config.yaml")

        # Build LLM chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # Memory
        self.buffer_memory = ConversationBufferMemory(return_messages=True)
        self.summary = ""

        # spaCy for query sanitization
        self.nlp = spacy.load("en_core_web_sm")

        print("RAG Assistant initialized successfully.")

    # ---------------------------------------------------------
    # LLM LOADER
    # ---------------------------------------------------------

    def _initialize_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.0
            )

        if os.getenv("GROQ_API_KEY"):
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.0
            )

        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
                temperature=0.0
            )

        raise ValueError("Missing API keys. Set OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY.")

    # ---------------------------------------------------------
    # QUERY SANITIZATION
    # ---------------------------------------------------------

    def preprocess_query(self, query: str) -> str:
        """Clean + normalize query before embedding."""

        query = query.lower()
        query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
        query = " ".join(query.split())

        doc = self.nlp(query)
        cleaned = " ".join([token.text for token in doc if not token.is_stop])

        return cleaned

    # ---------------------------------------------------------
    # ADD DOCUMENTS
    # ---------------------------------------------------------

    def add_documents(self, docs: List):
        self.vector_db.add_documents(docs)

    # ---------------------------------------------------------
    # OPTIONAL SUMMARY
    # ---------------------------------------------------------

    def _summarize_old_messages(self):
        messages = self.buffer_memory.chat_memory.messages

        if len(messages) > 16:  # 8 turns
            old_part = messages[:-16]
            text_to_summarize = "\n".join([m.content for m in old_part])

            prompt = PromptTemplate.from_template(
                "Summarize the following conversation in 200 words only:\n{chat}"
            )

            summary_chain = LLMChain(llm=self.llm, prompt=prompt)
            summary_result = summary_chain.invoke({"chat": text_to_summarize})

            self.summary += "\n" + summary_result["text"]
            self.buffer_memory.chat_memory.messages = messages[-16:]

    # ---------------------------------------------------------
    # MAIN INFERENCE FUNCTION
    # ---------------------------------------------------------

    def invoke(self, user_input: str, n_results: int = 3):
        if user_input.lower() == "summary":
            recent = "\n".join(
                [m.content for m in self.buffer_memory.chat_memory.messages[-8:]]
            )
            return f"Summary:\n{self.summary}\n\nRecent messages:\n{recent}"

        # 1. Clean/sanitize user query
        cleaned_query = self.preprocess_query(user_input)

        # 2. Enrich query for better retrieval (inject into template)
        # This improves semantic search significantly
        enriched_query = self.prompt_template.format(
            context="",
            question=cleaned_query
        )

        # 3. Vector search using enriched query
        search_result = self.vector_db.search(enriched_query, n_results=n_results)

        if not search_result or not search_result.get("documents"):
            return "No relevant documents found."

        retrieved_context = "\n\n".join(search_result["documents"])

        inputs = {
            "context": retrieved_context,
            "question": user_input  # original user question for generation
        }

        try:
            answer = self.chain.invoke(inputs)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        # Store in memory
        self.buffer_memory.chat_memory.add_user_message(user_input)
        self.buffer_memory.chat_memory.add_ai_message(answer)

        self._summarize_old_messages()

        return answer


# ---------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------

def main():
    try:
        assistant = RAGAssistant()

        print("\nLoading documents...")
        documents = load_documents()
        print(f"Loaded {len(documents)} documents.")
        assistant.add_documents(documents)

        while True:
            query = input("\nAsk something (or 'summary' / 'quit'): ").strip()
            if query.lower() == "quit":
                print("Goodbye!")
                break

            response = assistant.invoke(query)
            print("\n" + response)

    except Exception as e:
        print(f"Error: {e}")
        print("Ensure your .env file contains OPENAI / GROQ / GOOGLE API key.")


if __name__ == "__main__":
    main()
