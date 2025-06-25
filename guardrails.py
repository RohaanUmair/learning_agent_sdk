import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, handoff, RunContextWrapper, input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from agents.run import RunConfig
from pydantic import BaseModel
from agents.extensions import handoff_filters
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


class MathsQueryDetectorOutput(BaseModel):
    is_not_maths_query: bool
    explanation: str


maths_detector_guardrail_agent = Agent(
    name='Math Detector',
    instructions='Check if the user query is related to maths or not.',
    output_type=MathsQueryDetectorOutput
)


@output_guardrail
async def validate_numeric_output(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    # Trip if output is not a pure number
    try:
        float(output)
        is_not_number = False
    except ValueError:
        is_not_number = True

    return GuardrailFunctionOutput(
        tripwire_triggered=is_not_number,
        output_info=output
    )


@input_guardrail
async def not_maths_homework_detection_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list) -> GuardrailFunctionOutput:
    detection_result = await Runner.run(maths_detector_guardrail_agent, input, run_config=config)

    return GuardrailFunctionOutput(
        tripwire_triggered=detection_result.final_output.is_not_maths_query,
        output_info=detection_result.final_output
    )


math_homework_agent = Agent(
    name='Maths Helper Agent',
    instructions='Answer maths query. Only give answer in numbers form no other character',
    model=model,
    input_guardrails=[not_maths_homework_detection_guardrail],
    output_guardrails=[validate_numeric_output]
)


async def main():
    try:
        user_input = input('User: ')
        result = await Runner.run(math_homework_agent, user_input, run_config=config)
        # print('Guardrail not triggered')
        print(f'Response: {result.final_output}')
    except InputGuardrailTripwireTriggered as e:
        # print('Maths Guardrail triggered')
        # print(f'Exception details: {str(e)}')
        print('Response: You can only ask Maths related Queries!')
    except OutputGuardrailTripwireTriggered as e:
        print('Unexpected Output')



if __name__ == '__main__':
    asyncio.run(main())