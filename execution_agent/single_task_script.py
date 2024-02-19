import argparse
import time
import logging

import tiktoken
import instructor
from rich.console import Console
from rich.markdown import Markdown

import single_task_models as cg
from call_openai_api import api_function_call

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenAI API interaction script")
    parser.add_argument("--query", type=str, required=False, help="Question to ask")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # query = args.query
    # query = "What are the skills required for a senior computer vision engineer and how does that compare to a junior engineer?"
    query = "What are the key differences between a data scientist and a data engineer?"
    # query = "What is the meaning of life?"
    # query = "whats the average salary of junior data scientists"
    # query = "Which hybrid jobs require know how to fine-tuning LLMs"
    # query = "Give me the top 5 data engineering jobs with the highest salary and identify their main required skills"
    # query = "Give me the top 5 data engineering jobs with the highest salary and identify what the company is trying to build and solve"

    # query = "I know how to develop APIs, train LLMs and a bit of front-end development. What jobs can I apply for?"
    # query = "I a bit of python and I am looking for a remote job."

    # This query is not working
    # query = "I am looking for a prompt engineering job"
    # query = "Find me jobs that require prompt engineering and are in europe."
    # query = "what are the skills for an entry level data scientist role" # would be better to get the top skills
    print("query:", query)

    start_first_call = time.time()
    plan, error = api_function_call(
        system_message=cg.system_message_plan,
        query=query,
        response_model=cg.TaskPlan,
        # response_model=instructor.Partial[cg.TaskPlan],
        max_retries=2,
        # stream=True,
        stream=False,
    )

    if isinstance(plan, str):
        print(plan)
        return

    print(plan.model_dump_json(indent=2))
    logger.info(plan.model_dump_json(indent=2))
    end_first_call = time.time()
    print("time taken for first API call:", end_first_call - start_first_call)
    print("\n")

    input = (
        "REMEMBER: That you are a job counselor. Give a complete answer to the user question and do not cut down you answer. If you are given 20 twenty urls, you must also output 20 twenty urls to the user\n"
        + "Avoid short answers, avoid statements like '...and more.'. Provide a complete and helpful answer. User need to know about all the jobs available to them. Do not summarize your answer.\n"
        + f"user_question: {query} \n"
        + f"python_code: {plan.code_to_execute} \n"
        + f"repl_tool_output: {plan.result} \n\n"
        + "REMEMBER: Make sure to give a complete and useful answers to the user question and not cut down you answer. If the repl_tool has 20 urls, you must also output 20 urls to the user\n"
        + "Avoid concise answers, avoid unhelpful statements such as '...and more.'. Provide a complete answers. User need to know about all the jobs available to them. Do not summarize or cut down you answer.\n"
    )
    logger.info(f"input to openai: {input}")

    response, error = api_function_call(
        system_message=cg.system_message_synthesiser,
        query=input,
        # model="gpt-4-0125-preview",
        model="gpt-3.5-turbo-0125",
        # response_model=cg.SynthesiserResponse,
        response_model=instructor.Partial[cg.SynthesiserResponse],
        stream=True,
        # stream=False,
    )

    # logger.info(response.model_dump_json(indent=2))

    console = Console()
    for partial in response:
        obj = partial.model_dump()
        console.clear()
        console.print(obj["answer"])

    # console.print(Markdown(obj["answer"]))

    #     second_call_start = time.time()
    #     response, error = api_function_call(
    #         system_message=cg.system_message_synthesiser,
    #         query=input,
    #         model="gpt-3.5-turbo-0125",
    #         max_retries=args.max_retries,
    #         stream=False,
    #     )
    #     end = time.time()

    #     console = Console()
    #     md = Markdown(response.choices[0].message.content)
    #     console.print(md)

    #     print("time taken for second API call:", end - second_call_start)
    #     print("time taken for whole process:", end - start_first_call)


if __name__ == "__main__":
    main()

# TODO: The variation in the keywords is too large, 'prompt engineering' is not the same as NLP engineer
# TODO: The result of code can be over 16k tokens, so we need to handle that.

# Example of code error:
"""
df = pd.read_pickle('data/extracted_cleaned_df.pkl')
print(df.loc[df['experience_level'] == 'entry-level' & df['job_title'].str.contains('data scientist'), ['job_skills', 'jobs_towardsai_url']])
WARNING:langchain_community.utilities.python:Python REPL can execute arbitrary code. Use with caution.
time taken for python execution: 0.04783201217651367
An error occurred: TypeError("Cannot perform 'rand_' with a dtyped [bool] array and scalar of type [bool]")
"""
