import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, TResponseInputItem
from agents.run import RunConfig
from pydantic import BaseModel
import asyncio


load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError('Gemini API Key is not valid.')


external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


agent = Agent(
    name='Assistant',
    instructions='Help user in solving thier queries. Be concise and precise.',
    model=model
)


convo: list[TResponseInputItem] = []


async def main():
    while True:
        user_input = input('User: ')
        convo.append({ 'content': user_input, 'role': 'user' })
        result = await Runner.run(agent, convo, run_config=config)

        # for message in result.to_input_list():
        #     print(message)

        print(f'Ai: {result.final_output}')

        convo = result.to_input_list()

asyncio.run(main())