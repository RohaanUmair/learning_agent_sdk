import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
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


@function_tool
def add(a: float, b: float) -> float:
    '''
    Adds given two numbers

    Args:
        a: first number to add
        b: second number to add
    '''

    print('add tool called')
    return a + b


@function_tool
def subtract(a: float, b: float) -> float:
    '''
    Subtracts given two numbers

    Args:
        a: first number to subtract from
        b: second number to subtract
    '''

    print('subtract tool called')
    return a - b


@function_tool
def multiply(a: float, b: float) -> float:
    '''
    Muliplies given two numbers

    Args:
        a: first number to multiply
        b: second number to multiply
    '''

    print('multiply tool called')
    return a * b


@function_tool
def divide(a: float, b: float) -> float:
    '''
    Performs division between two numbers

    Args:
        a: dividend
        b: diviser
    '''

    print('divide tool called')
    return a / b


calculator_agent = Agent(
    name='Calculator Agent',
    instructions='You are a calculator Agent. You can only perform calculations and if user asks anything else tell user that you cannot do anything else.',
    model=model,
    tools=[add, subtract, multiply, divide]
)



while True:
    user_input = input('User: ')
    result = Runner.run_sync(calculator_agent, user_input, run_config=config)
    print(f'Agent: {result.final_output}')