"""
API Client Wrapper for Hermes Pipeline

Supports both OpenAI-compatible APIs (Qwen via Nous) and Anthropic.
"""
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

class BaseAPIClient:
    """Base class for API clients"""
    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Tuple[str, Dict]:
        raise NotImplementedError

class OpenAICompatClient(BaseAPIClient):
    """OpenAI-compatible client for Qwen and other models"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.model = model or "qwen/qwen3.5-plus-02-15"
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://inference-api.nousresearch.com/v1"
        )
    
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        max_tokens: int = 2500,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text using OpenAI-compatible API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            content = response.choices[0].message.content
            end_time = datetime.now(timezone.utc)
            
            api_meta = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "model_id": self.model,
                "timestamp_utc": end_time.isoformat(),
                "latency_ms": int((end_time - start_time).total_seconds() * 1000),
                "retries": 0,
            }
            
            return content, api_meta
            
        except Exception as e:
            return None, {
                "error": str(e),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

class AnthropicClient(BaseAPIClient):
    """Anthropic API client (original implementation)"""
    
    def __init__(self, api_key: str, model: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.model = model or "claude-opus-4-7"
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2500,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text using Anthropic API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            
            content = response.content[0].text
            end_time = datetime.now(timezone.utc)
            
            api_meta = {
                "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                "model_id": self.model,
                "timestamp_utc": end_time.isoformat(),
                "latency_ms": int((end_time - start_time).total_seconds() * 1000),
                "retries": 0,
            }
            
            return content, api_meta
            
        except Exception as e:
            return None, {
                "error": str(e),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

def create_client(provider: str = "nous", api_key: str = None, base_url: str = None, model: str = None):
    """Factory function to create appropriate client (supports: nous, openrouter, anthropic)"""
    
    if provider == "anthropic":
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicClient(api_key=api_key, model=model)
    else:
        # Default to OpenAI-compatible (Nous/Qwen)
        if not api_key:
            api_key = os.getenv("NOUS_API_KEY") or os.getenv("OPENAI_API_KEY")
        return OpenAICompatClient(
            api_key=api_key,
            base_url=base_url or os.getenv("API_BASE_URL"),
            model=model
        )
