import os
import logging
import time

import pandas as pd
import gradio as gr
import instructor

import execution_agent.single_task_models as cg
from execution_agent.call_openai_api import api_function_call


CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_answer(query: str, chatbot):
    start = time.time()

    query_validation, error = api_function_call(
        system_message=cg.system_message_validation,
        query=query,
        model="gpt-3.5-turbo-0125",
        response_model=cg.QueryValidation,
    )
    end_validation = time.time()

    completion_string = "Validating query:\n" + query_validation.model_dump_json(
        indent=2
    )
    completion_string += f"\ntime taken for validation call: {end_validation - start:0.2f} sec\n\n Generating a plan ..."
    yield completion_string

    if query_validation.is_valid is False:
        completion_string += "\nThe query is not valid"
        yield completion_string
        return

    start_plan = time.time()
    plan, error = api_function_call(
        system_message=cg.system_message_plan,
        query=query,
        # response_model=instructor.Partial[cg.TaskPlan],
        response_model=cg.TaskPlan,
        stream=False,
        max_retries=3,
    )
    if isinstance(plan, str):
        completion_string += f"\n{plan}"
        yield completion_string
        return
    completion_string += "\nPlan for the task:\n" + plan.model_dump_json(indent=2)
    end_plan = time.time()
    completion_string += f"\ntime taken for plan call and python execution: {end_plan - start_plan:0.2f} sec\n\n"

    input = (
        f"user_question: {query} \n"
        + f"python_code: {plan.code_to_execute}"
        + f"repl_output: {plan.result} \n"
    )

    start_second_call = time.time()
    response, error = api_function_call(
        system_message=cg.system_message_synthesiser,
        query=input,
        model="gpt-4-0125-preview",
        stream=True,
    )

    completion_string += "\n\nAnswering the user query:\n"
    for token in response:
        completion_string += token
        yield completion_string

    end = time.time()
    completion_string += (
        f"\n\ntime taken for answer call: {end - start_second_call:0.2f} sec"
    )
    completion_string += f"\ntime taken for whole process: {end - start:0.2f} sec"
    yield completion_string


example_questions = [
    "Compare the skills needed for an LLM engineer to a machine learning engineer",
    "What are the skills required for a senior computer vision engineer, and how does that compare to a junior engineer?",
    "What is the meaning of life?",
    "What is the average salary of junior data scientists",
    "Which hybrid jobs require to know how to fine-tune LLMs",
    "Give me the top 5 data engineering jobs with the highest salary and identify their main required skills",
    "Give me the top 5 data engineering jobs with the highest salary and identify what the company is trying to build",
    "I know how to develop APIs, train LLMs, and do a bit of front-end development. What jobs can I apply for?",
    "I know a bit of Python, and I am looking for a remote job.",
    "I am looking for a prompt engineering job",
    "Find me jobs that require prompt engineering and are in Europe.",
    "what are the skills for an entry-level data scientist role",
    "Which remote job has the highest salary",
    "What are the top 5 remote jobs with the highest salary",
    "Where are most onsite jobs located?",
    "Can you show me jobs that require data analytics?",
    "What jobs are available in Canada or remotely for North Americans?",
    "Are there any part-time positions available for data scientists?",
    "Show me the latest jobs submitted to the database.",
    "How do I apply for a ml engineer at Google?",
    "What are the visa requirements to work in Europe?",
    "Tell me more about Amazon.",
    "What benefits does Netflix offer?",
    "How can I make my application stand out for an AI Research Scientist role?",
    "What are common interview questions for NLP Engineer positions?",
    "What is the salary range for a Robotics Engineer in India?",
]


chatbot = gr.Chatbot(show_copy_button=True, scale=2)
with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        get_answer,
        chatbot=chatbot,
        examples=example_questions,
        fill_height=True,
        title="Jobs Database Chatbot- BETA",
    )

demo.queue()
demo.launch(debug=False, share=False, max_threads=CONCURRENCY_COUNT)
