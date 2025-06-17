import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, handoff, RunContextWrapper
from agents.run import RunConfig
from pydantic import BaseModel
from agents.extensions import handoff_filters



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

class Problems(BaseModel):
    user_question: str
    topic: str


def on_handoff(ctx: RunContextWrapper, input_data: Problems):
    print(f'\nUser Question: {input_data.user_question}\n')
    print(f'\nTopic: {input_data.topic}\n')


maths_agent = Agent(
    name='maths assistant',
    instructions='You are a maths assistant you only solve mathematic problems.',
    model=model,
    handoff_description='This agent only answers maths related queries'
)

history_agent = Agent(
    name='history assistant',
    instructions='You are a history assistant you only tell about history.',
    model=model,
    handoff_description='This agent only answers historical queries'
)

triage_agent = Agent(
    name='triage agent',
    instructions='You are a teaching assistant. You handoff user request to corresponding agent if present. If such agent is not present to handoff, you simply tell user what queries you can answer.',
    model=model,
    handoffs=[
        handoff(agent=maths_agent, input_type=Problems, on_handoff=on_handoff, input_filter=handoff_filters.remove_all_tools),
        handoff(agent=history_agent, input_type=Problems, on_handoff=on_handoff)
    ]
)


user_input = input('Input: ')
result = Runner.run_sync(triage_agent, user_input, run_config=config)

print('Agent: ', result.final_output)