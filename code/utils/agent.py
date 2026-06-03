import asyncio
import backoff
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from .openai_utils import num_tokens_from_string, model2max_context, DEFAULT_MAX_CONTEXT


class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0,
                 base_url: str = None, api_key: str = "dummy") -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=20)
    async def query(self, messages: "list[dict]", max_tokens: int, temperature: float) -> str:
        if self.sleep_time > 0:
            await asyncio.sleep(self.sleep_time)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": memory})
        print(f"----- {self.name} -----\n{memory}\n")

    async def ask(self, temperature: float = None):
        num_context_token = sum(num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst)
        max_token = model2max_context.get(self.model_name, DEFAULT_MAX_CONTEXT) - num_context_token
        return await self.query(self.memory_lst, max_token, temperature=temperature if temperature else self.temperature)
