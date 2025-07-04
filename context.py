import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, model_settings, ModelSettings, function_tool, RunContextWrapper
from agents.run import RunConfig
from pydantic import BaseModel
from dataclasses import dataclass


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


@dataclass
class UserInfo:
    name: str
    uid: int
    
user_info = UserInfo('Rohaan', 123)


@function_tool
async def fetch_user_age(ctx: RunContextWrapper[UserInfo]) -> str:  
    """Fetch the age of the user. Call this function to get user's age information."""
    return f"The user {ctx.context.name} is 47 years old"


agent = Agent[UserInfo](
    name='Agent',
    instructions='You are supportive agent',
    model=model,
    tools=[fetch_user_age]
)



while True:
    user_input = input('User: ')
    result = Runner.run_sync(
        agent,
        user_input, 
        run_config=config,
        context=user_info
    )
    print(f'Ai: {result.final_output}')