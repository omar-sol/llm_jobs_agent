import argparse
import time

import tiktoken
import instructor
from rich.console import Console
from rich.markdown import Markdown

import single_task_models as cg
from call_openai_api import api_function_call


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenAI API interaction script")
    parser.add_argument("--query", type=str, required=False, help="Question to ask")
    parser.add_argument(
        "--model", default="gpt4", type=str, help="Model to use for the OpenAI API"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Number of retries for the OpenAI API",
    )
    return parser.parse_args()


def get_model(model_name):
    model_mapping = {
        "gpt3.5": "gpt-3.5-turbo-0125",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpt4": "gpt-4-0125-preview",
    }
    return model_mapping[model_name]


def main():
    args = parse_arguments()
    model = get_model(args.model)
    print("model:", model)

    # query = args.query
    query = "What are the skills required for a senior computer vision engineer and how does that compare to a junior engineer?"
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
        system_message=cg.system_message,
        query=query,
        model=model,
        # response_model=cg.TaskPlan,
        response_model=instructor.Partial[cg.TaskPlan],
        max_retries=args.max_retries,
        stream=True,
    )

    console = Console()
    for task in plan:
        obj = task.model_dump()
        console.clear()
        console.print(obj)

    # print(task.model_dump_json(indent=2))

    end_first_call = time.time()
    print("time taken for first API call:", end_first_call - start_first_call)
    print("\n")

    if task.query_validation.is_valid is False:
        print("The query is not valid")
        return

    start = time.time()
    result = task.execute_code()
    end = time.time()
    print("time taken for code execution:", end - start)
    print("\n")

    if "Error" in result or "Exception" in result:
        print(f"An error occurred: {result}")
    else:
        input = (
            f"user_question: {query} \n"
            + f"python_code: {task.code_to_execute}"
            + f"repl_output: {result} \n"
        )

        second_call_start = time.time()
        response, error = api_function_call(
            system_message=cg.system_message_synthesiser,
            query=input,
            model="gpt-3.5-turbo-0125",
            max_retries=args.max_retries,
            stream=False,
        )
        end = time.time()

        console = Console()
        md = Markdown(response.choices[0].message.content)
        console.print(md)

        print("time taken for second API call:", end - second_call_start)
        print("time taken for whole process:", end - start_first_call)


if __name__ == "__main__":
    main()

# TODO: Add validation for the generated code, retry when the code does not work
# TODO: The variation in the keywords is too large, 'prompt engineering' is not the same as NLP engineer
# TODO: Write a Python REPL tool that uses asyncio to run the code. + Use the async OpenAI client.

# Example of code error:
"""
df = pd.read_pickle('data/extracted_cleaned_df.pkl')
print(df.loc[df['experience_level'] == 'entry-level' & df['job_title'].str.contains('data scientist'), ['job_skills', 'jobs_towardsai_url']])
WARNING:langchain_community.utilities.python:Python REPL can execute arbitrary code. Use with caution.
time taken for python execution: 0.04783201217651367
An error occurred: TypeError("Cannot perform 'rand_' with a dtyped [bool] array and scalar of type [bool]")
"""


# How can this be improved?
# - The code generated can be improved by using a better model
