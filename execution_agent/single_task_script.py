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
    # query = "What are the key differences between a data scientist and a data engineer?"
    # query = "What is the meaning of life?"
    # query = "whats the average salary of junior data scientists"
    # query = "Which hybrid jobs require know how to fine-tuning LLMs"
    # query = "Give me the top 5 data engineering jobs with the highest salary and identify their main required skills"
    # query = "Give me the top 5 data engineering jobs with the highest salary and identify what the company is trying to build and solve"

    # query = "I know how to develop APIs, train LLMs and a bit of front-end development. What jobs can I apply for?"
    query = "I a bit of python and I am looking for a remote job."

    # This query is not working
    # query = "I am looking for a prompt engineering job"
    # query = "Find me jobs that require prompt engineering and are in europe."
    # query = "what are the skills for an entry level data scientist role" # would be better to get the top skills

    logger.info(f"query: {query}")
    logger.info(f"Generating plan for the query")
    start_first_call = time.time()
    plan, error = api_function_call(
        system_message=cg.system_message_plan,
        query=query,
        model="gpt-4-turbo",
        response_model=cg.TaskPlan,
        max_retries=2,
        stream=False,
    )

    if isinstance(plan, str):
        print(plan)
        return

    # logger.info("Plan:\n", plan.model_dump_json(indent=2))
    end_first_call = time.time()
    print("time taken for first API call:", end_first_call - start_first_call)
    print("\n")

    synthesiser_prompt = cg.synthesiser_prompt.format(
        query=query,
        code_to_execute=plan.code_to_execute,
        result=plan.result,
        url_slugs=plan.url_slugs,
    )
    logger.info(f"synthesiser prompt: {synthesiser_prompt}")

    logger.info(f"Generating final answer for query")
    response, error = api_function_call(
        system_message=cg.system_message_synthesiser,
        query=synthesiser_prompt,
        model="gpt-4-turbo",
        # response_model=cg.SynthesiserResponse,
        # stream=True,
        stream=False,
    )

    # logger.info(response.model_dump_json(indent=2))

    console = Console()
    # for partial in response:
    # console.clear()
    # console.print(partial)

    # obj = partial.model_dump()
    # console.print(Markdown(str(obj["answer"])))

    # console.print(Markdown(str(response.answer)))
    console.print(Markdown(str(response.choices[0].message.content)))

    end = time.time()
    print("time taken for second API call:", end - end_first_call)
    print("time taken for whole process:", end - start_first_call)


if __name__ == "__main__":
    main()


# for the DB info on SQL, I need to check all the possible values for each column
# for requirement remote, the LLM does not know which column to filter on e.g. it looks into 'city' and 'country' columns

# Fix the salaries for the SQL DB.

# Looking into LLM Engineers should be done by looking with 'LLM' and 'Engineer' separately in the job title + cleaned description

# How to properly evaluate the system?
