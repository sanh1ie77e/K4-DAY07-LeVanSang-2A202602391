from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)

        # 2. Build prompt with context chunks
        context_parts = []
        for i, r in enumerate(results, start=1):
            context_parts.append(f"[{i}] {r['content']}")
        context = "\n".join(context_parts)

        prompt = (
            f"Dựa trên các đoạn thông tin sau đây, hãy trả lời câu hỏi.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Câu trả lời:"
        )

        # 3. Call LLM
        return self.llm_fn(prompt)
