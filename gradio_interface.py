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


def user(query, chatbot):
    """Adds user's query immediately to the chat."""
    return "", chatbot + [[query, None]]


def get_answer(history):
    query = history[-1][0]
    history[-1][1] = ""
    history[-1][1] = "Plan for the task:"

    start = time.time()
    plan, error = api_function_call(
        system_message=cg.system_message,
        query=query,
        response_model=instructor.Partial[cg.TaskPlan],
        stream=True,
    )

    for task in plan:
        obj = task.model_dump_json(indent=2)
        history[-1][1] = "Plan for the task:\n" + str(obj)
        yield history

    end_first_call = time.time()
    history[-1][
        1
    ] += f"\ntime taken for first API call: {end_first_call - start:0.2f} sec\n"

    if task.query_validation.is_valid is False:
        history[-1][1] += "\nThe query is not valid"
        yield history
        return

    start_execution = time.time()
    result = task.execute_code()
    end_execution = time.time()
    history[-1][
        1
    ] += (
        f"time taken for python execution: {end_execution - start_execution:0.2f} sec\n"
    )

    if "Error" in result or "Exception" in result:
        logger.error(f"An error occurred: {result}")
    else:
        input = (
            f"user_question: {query} \n"
            + f"python_code: {task.code_to_execute}"
            + f"repl_output: {result} \n"
        )

        start_second_call = time.time()
        response, error = api_function_call(
            system_message=cg.system_message_synthesiser,
            query=input,
            model="gpt-3.5-turbo-0125",
            stream=True,
        )

        history[-1][1] += "\n"
        for token in response:
            history[-1][1] += token
            yield history

        end = time.time()
        history[-1][
            1
        ] += f"\ntime taken for second API call: {end - start_second_call:0.2f} sec"
        history[-1][1] += f"\ntime taken for whole process: {end - start:0.2f} sec"
        yield history


example_questions = [
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


with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML(
            "<h3><center>BETA - Towards AI 🤖: A Question-Answering Bot for anything AI-jobs related</center></h3>"
        )
    latest_completion = gr.State()
    chatbot = gr.Chatbot(elem_id="chatbot", show_copy_button=True)
    with gr.Row():
        query = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary")
    with gr.Row():
        examples = gr.Examples(
            examples=example_questions,
            inputs=query,
        )
    completion = gr.State()

    submit.click(
        user,
        [query, chatbot],
        [query, chatbot],
    ).then(get_answer, inputs=[chatbot], outputs=[chatbot])
    query.submit(user, [query, chatbot], [query, chatbot]).then(
        get_answer, inputs=[chatbot], outputs=[chatbot]
    )

demo.queue()
demo.launch(debug=False, share=False, max_threads=CONCURRENCY_COUNT)
