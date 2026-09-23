"""Chat and embedding clients. One Ollama base URL serves both."""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import EMBED_CTX, KEEP_ALIVE, MAX_ANSWER_TOKENS, env


def embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=env("EMBED_MODEL"),
        base_url=env("OLLAMA_BASE_URL"),
        num_ctx=EMBED_CTX,
        keep_alive=KEEP_ALIVE,
    )


def llm() -> ChatOllama:
    return ChatOllama(
        model=env("CHAT_MODEL"),
        base_url=env("OLLAMA_BASE_URL"),
        temperature=0,
        keep_alive=KEEP_ALIVE,
        num_ctx=int(env("NUM_CTX")),
        num_predict=MAX_ANSWER_TOKENS,
        validate_model_on_init=True,  # fail with a clear message if it isn't pulled
    )
