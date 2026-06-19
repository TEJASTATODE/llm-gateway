from google import genai
from google.genai import types
from typing import AsyncGenerator
import asyncio

from gateway.providers.base import BaseProvider, Message, CompletionResponse, Usage
from gateway.config import settings


MODEL_MAP = {
    "gemini-flash": "gemini-2.5-flash-lite",
    "gemini-pro": "gemini-2.5-pro",
    "auto": "gemini-2.5-flash-lite",
}

class GeminiProvider(BaseProvider):

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def is_available(self) -> bool:
        return (
            bool(settings.gemini_api_key)
            and settings.gemini_api_key != "placeholder"
        )

    def _resolve_model(self, model: str) -> str:
        return MODEL_MAP.get(model, "gemini-2.5-flash")

    def _build_contents(self, messages: list[Message]) -> tuple[str, list]:
        """Extract system prompt and build contents list for new SDK"""
        system_prompt = ""
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.content)]
                    )
                )
            elif msg.role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=msg.content)]
                    )
                )

        return system_prompt, contents

    async def complete(
        self,
        model: str,
        messages: list[Message],
    ) -> CompletionResponse:
        resolved_model = self._resolve_model(model)
        system_prompt, contents = self._build_contents(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        )

        text = response.text or ""
        estimated_prompt_tokens = sum(len(m.content.split()) for m in messages) * 4 // 3
        estimated_completion_tokens = len(text.split()) * 4 // 3

        return CompletionResponse(
            content=text,
            model=resolved_model,
            provider="gemini",
            usage=Usage(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
            ),
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        resolved_model = self._resolve_model(model)
        system_prompt, contents = self._build_contents(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content_stream(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text