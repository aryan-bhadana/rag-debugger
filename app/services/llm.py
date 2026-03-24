from __future__ import annotations

from typing import List

from groq import Groq

from app.core.config import settings


class LLMService:
    def __init__(self) -> None:
        self.api_key = settings.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"

    def _build_prompt(self, query: str, context_text: str, detailed: bool = False) -> str:
        instruction = (
            "Provide a complete and detailed answer using the context.\n\n"
            if detailed
            else ""
        )

        return (
            "You are a helpful assistant.\n\n"
            "Answer the question ONLY using the provided context.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"{instruction}"
            "Context:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{query}\n"
        )

    def generate(self, query: str, context: List[str]) -> str:
        if not context:
            return "I don't know"

        context_text = "\n\n".join(context)
        prompt = self._build_prompt(query, context_text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )

        answer = response.choices[0].message.content or "I don't know"

        if len(answer.split()) < 5:
            retry_prompt = self._build_prompt(query, context_text, detailed=True)
            retry_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0,
                top_p=1,
            )
            answer = retry_response.choices[0].message.content or answer

        return answer
