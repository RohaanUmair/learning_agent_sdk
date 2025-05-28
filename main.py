import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

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
    name='Calculator Agent',
    instructions='You are a calculator Ai agent you can only do calculations. If anything else is told just tell the user you can\'t do anything but calculations.',
    model=model
)


while True:
    user_input = input('User: ')
    result = Runner.run_sync(agent, user_input, run_config=config)

    print(result.final_output)