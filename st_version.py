import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from pydantic import BaseModel
import streamlit as st
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


st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
)

st.markdown('<h1 style="position: fixed; font-size: 40px; margin-bottom: 40px; background-color: #0E1117; z-index: 999; width: 90%; top: 40px;">Unit Conversion using AI</h1>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

role_icon_map = {
    "user": "🧑",
    "assistant": "🤖",
    "Calculator Agent": "🧮",
    "Translator Agent": "🌐",
    "Triage Agent": "🛣️"
}


async def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        intro_message = "Hello! I am an AI Agent. I can translate, and perfrom Calculations. How can I help you?"
        st.session_state['history'].append({'role': 'assistant', 'parts': [intro_message]})


    chat_input = st.chat_input('Enter your prompt')
    
    if chat_input:
        st.session_state['history'].append({'role': 'user', 'parts': [chat_input]})

        result = await Runner.run(triage_agent, chat_input, run_config=config)

        if isinstance(result.final_output, AgentOutput):
            st.session_state['history'].append({'role': result.final_output.agent_name, 'parts': [result.final_output.response]})
        else:
            # Raw response from triage agent itself
            st.session_state['history'].append({'role': 'assistant', 'parts': [result.final_output]})


    for message in st.session_state['history']:
        role = message['role']
        content = message['parts'][0]
        icon = role_icon_map.get(role, "❓")
        st.chat_message(role, avatar=icon).write(content)


asyncio.run(main())