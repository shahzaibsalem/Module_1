import os
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

# Load environment variables
load_dotenv()


def load_documents() -> List[dict]:
    results = []
    absolute_folder = os.path.abspath(folder_path)

    for filename in os.listdir(absolute_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(absolute_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = file.read()
                results.append({
                    "content": data,
                    "metadata": {"source": filename}
                })
    return results


class RAGAssistant:
    def __init__(self):
        """Initialize the RAG assistant with memory and LLM."""
        self.llm = self._initialize_llm()
        self.vector_db = VectorDB()
        self.prompt_template = load_prompt_from_yaml('prompt_config.yaml')

        # Build RAG chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # Keep last 8 turns (each turn = 2 messages)
        self.buffer_memory = ConversationBufferMemory(return_messages=True)
        self.summary = ""  # Holds summarized chat history

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """Initialize available LLM in order of preference."""
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                              model=model_name, temperature=0.0)

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
                            model=model_name, temperature=0.0)

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"),
                                          model=model_name, temperature=0.0)

        else:
            raise ValueError("No valid API key found. Please set one of: "
                             "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file.")

    def add_documents(self, documents: List) -> None:
        """Add documents to vector database."""
        self.vector_db.add_documents(documents)

    def _summarize_old_messages(self):
        """Summarize old messages when chat exceeds 8 turns."""
        messages = self.buffer_memory.chat_memory.messages

        if len(messages) > 16: 
            old_part = messages[:-16]
            text_to_summarize = "\n".join([m.content for m in old_part])

            prompt = PromptTemplate.from_template(
                "Summarize the following conversation in 200 words only:\n{chat}"
            )
            summary_chain = LLMChain(llm=self.llm, prompt=prompt)
            summary_result = summary_chain.invoke({"chat": text_to_summarize})

            # Merge with previous summary
            self.summary += "\n" + summary_result["text"]

            # Keep only last 8 turns
            self.buffer_memory.chat_memory.messages = messages[-16:]

    def invoke(self, input: str, n_results: int = 3) -> str:
        """Handle user input, summarize long chats, and reuse memory."""
        if input.lower() == "summary":
            recent_context = "\n".join(
                [m.content for m in self.buffer_memory.chat_memory.messages[-8:]]
            )
            return f"🧾 Chat Summary:\n{self.summary.strip()}\n\nRecent messages:\n{recent_context}"

        # Retrieve related context
        search_result = self.vector_db.search(query=input, n_results=n_results)
        if not search_result or not search_result.get("documents"):
            return "No relevant documents found."

        context = "\n\n".join(search_result.get("documents"))
        inputs = {"context": context, "question": input}

        # Generate LLM answer
        try:
            llm_answer = self.chain.invoke(inputs)
        except Exception as e:
            llm_answer = f"Error generating answer: {e}"

        # Store conversation
        self.buffer_memory.chat_memory.add_user_message(input)
        self.buffer_memory.chat_memory.add_ai_message(llm_answer)

        # Summarize if needed
        self._summarize_old_messages()

        return llm_answer


def main():
    """Main entry to interact with the RAG assistant."""
    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} documents.")
        assistant.add_documents(sample_docs)

        while True:
            question = input("\nAsk something (or type 'summary' / 'quit'): ").strip()
            if question.lower() == "quit":
                print("Exiting RAG Assistant.")
                break
            result = assistant.invoke(question)
            print("\n" + result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Ensure your .env file includes one of the following API keys:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
