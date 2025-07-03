import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, model_settings, ModelSettings, function_tool
from agents.run import RunConfig
from pydantic import BaseModel
import asyncio
import requests


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
    tracing_disabled=True,
    model_settings=ModelSettings(temperature=0.7, top_p=0.7)
)


@function_tool
def get_weather(city: str) -> str:
    '''Get weather for the given city
    '''
    print('tool called')
    result = requests.get(f"http://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}")
    data = result.json()
    return f"{data['current']['temp_c']}°C with {data['current']['condition']['text']}"



weather_agent = Agent(
    name='Weather Agent',
    instructions='You are weather agent. If API fails to fetch weather, try again 3 more times calling the same tool.',
    model=model,
    tools=[get_weather]
)

async def main():
    while True:
        user_input = input('User: ')
        result = await Runner.run(
            weather_agent,
            user_input, 
            run_config=config, 
        )
        print(f'Ai: {result.final_output}')

asyncio.run(main())