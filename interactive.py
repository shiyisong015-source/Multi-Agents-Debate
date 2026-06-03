"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import json
import asyncio
from code.utils.agent import Agent


VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "dummy"

NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]


class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float,
                 base_url: str, api_key: str, sleep_time: float) -> None:
        super().__init__(model_name, name, temperature, sleep_time, base_url, api_key)


class Debate:
    def __init__(self,
                 model_name: str,
                 temperature: float = 0,
                 base_url: str = VLLM_BASE_URL,
                 api_key: str = VLLM_API_KEY,
                 config: dict = None,
                 max_round: int = 3,
                 sleep_time: float = 0,
                 debate_id: int = 0) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.debate_id = debate_id

        self.init_prompt()
        self.creat_agents()

    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("affirmative_meta_prompt")
        prompt_replace("negative_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature,
                         base_url=self.base_url, api_key=self.api_key, sleep_time=self.sleep_time)
            for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    async def init_agents(self):
        self.affirmative.set_meta_prompt(self.config['affirmative_meta_prompt'])
        self.negative.set_meta_prompt(self.config['negative_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])

        print(f"[Debate-{self.debate_id}] ===== Round-1 =====\n")
        self.affirmative.add_event(self.config['affirmative_prompt'])
        self.aff_ans = await self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)
        self.config['base_answer'] = self.aff_ans

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = await self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(
            self.config['moderator_prompt']
            .replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        self.mod_ans = await self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = json.loads(self.mod_ans)

    def round_dct(self, num: int):
        dct = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
               6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'}
        return dct[num]

    def print_answer(self):
        print(f"\n[Debate-{self.debate_id}] ===== Done! =====")
        print(f"Topic: {self.config['debate_topic']}")
        print(f"Base Answer: {self.config['base_answer']}")
        print(f"Debate Answer: {self.config['debate_answer']}")
        print(f"Reason: {self.config.get('Reason', '')}")

    async def run(self):
        await self.init_agents()

        for round in range(self.max_round - 1):
            if self.mod_ans["debate_answer"] != '':
                break

            print(f"[Debate-{self.debate_id}] ===== Round-{round+2} =====\n")
            self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
            self.aff_ans = await self.affirmative.ask()
            self.affirmative.add_memory(self.aff_ans)

            self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
            self.neg_ans = await self.negative.ask()
            self.negative.add_memory(self.neg_ans)

            self.moderator.add_event(
                self.config['moderator_prompt']
                .replace('##aff_ans##', self.aff_ans)
                .replace('##neg_ans##', self.neg_ans)
                .replace('##round##', self.round_dct(round + 2))
            )
            self.mod_ans = await self.moderator.ask()
            self.moderator.add_memory(self.mod_ans)
            self.mod_ans = json.loads(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True
        else:
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature,
                                        base_url=self.base_url, api_key=self.api_key, sleep_time=self.sleep_time)
            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            judge_player.add_event(self.config['judge_prompt_last1']
                                   .replace('##aff_ans##', self.aff_ans)
                                   .replace('##neg_ans##', self.neg_ans))
            ans = await judge_player.ask()
            judge_player.add_memory(ans)

            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = await judge_player.ask()
            judge_player.add_memory(ans)

            ans = json.loads(ans)
            if ans["debate_answer"] != '':
                self.config['success'] = True
            self.config.update(ans)
            self.players.append(judge_player)

        self.print_answer()


async def run_debates(topics: list[str], model_name: str, max_round: int = 3,
                      base_url: str = VLLM_BASE_URL, api_key: str = VLLM_API_KEY,
                      concurrency: int = None):
    """Run multiple debates concurrently, one per topic."""
    current_script_path = os.path.abspath(__file__)
    mad_path = os.path.dirname(current_script_path)

    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def run_one(topic: str, debate_id: int):
        config = json.load(open(os.path.join(mad_path, "code", "utils", "config4all.json"), "r"))
        config['debate_topic'] = topic
        debate = Debate(model_name=model_name, base_url=base_url, api_key=api_key,
                        config=config, max_round=max_round, temperature=0, debate_id=debate_id)
        if sem:
            async with sem:
                await debate.run()
        else:
            await debate.run()

    await asyncio.gather(*[run_one(topic, i) for i, topic in enumerate(topics)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="vLLM model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--topics-file", default=None,
                        help="Path to JSON file from prepare_mmlu.py (list of {topic, ...} objects)")
    parser.add_argument("--n", type=int, default=None, help="Max number of topics to run")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent debates")
    parser.add_argument("--max-round", type=int, default=3)
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    args = parser.parse_args()

    if args.topics_file:
        with open(args.topics_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        topics = [item["topic"] if isinstance(item, dict) else item for item in raw]
        if args.n:
            topics = topics[: args.n]
        print(f"Loaded {len(topics)} topics from {args.topics_file}")
    else:
        topics = []
        n = args.n or int(input("Number of debates: ").strip())
        for i in range(n):
            topic = ""
            while not topic:
                topic = input(f"Topic [{i+1}/{n}]: ").strip()
            topics.append(topic)

    asyncio.run(run_debates(topics, model_name=args.model, base_url=args.base_url,
                            api_key=VLLM_API_KEY, max_round=args.max_round,
                            concurrency=args.concurrency))
