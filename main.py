import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from pydantic import BaseModel


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

class AgentOutput(BaseModel):
    response: str
    agent_name: str

calculator_agent = Agent(
    name='Calculator Agent',
    instructions='You are a calculator AI agent you can only do calculations. If anything else is told just tell the user you can\'t do anything but calculations.',
    model=model,
    output_type=AgentOutput
)

translator_agent = Agent(
    name='Translator Agent',
    instructions='You are a translator AI agent you can only do translations. If anything else is told just tell the user you can\'t do anything but translations.',
    model=model,
    output_type=AgentOutput
)

triage_agent = Agent(
    name='Triage Agent',
    instructions='You are a triage Agent. You can only transfer request to other agents. And you cannot answer directlty. If no agent is present to handle a request tell the user that you can "only perform calculations and translations". Do not tell user that you are a triage agent and you can transfer, just handoff to agent or tell "I can only perform calculations and translations" as per required sitiation.',
    model=model,
    handoffs=[calculator_agent, translator_agent]
)



while True:
    user_input = input('User: ')
    result = Runner.run_sync(
        triage_agent,
        user_input, 
        run_config=config, 
        # max_turns=2
    )

    if isinstance(result.final_output, AgentOutput):
        print(f'{result.final_output.agent_name}: {result.final_output.response}\n')
    else:
        # Raw response from triage agent itself
        print(f'Main Agent: {result.final_output}')