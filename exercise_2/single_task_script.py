import argparse
import logging
import pdb
import time

import instructor
import structured_output_models as models
from call_openai_api import api_function_call
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.markdown import Markdown

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Excel RAG Chatbot")
    parser.add_argument("--query", type=str, required=False, help="Question to ask")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # query = args.query
    # query = "Quelle est la vente la plus conséquente?"
    # query = "Combien a rapporté le client le plus profitable??"
    query = "Quelle est la valeur vie client moyenne?"

    logger.info(f"query: {query}")
    logger.info(f"Generating plan for the query")
    start_first_call = time.time()
    plan, error = api_function_call(
        system_message=models.system_message_plan,
        query=query,
        model="gpt-4o",
        response_model=models.TaskPlan,
        max_retries=2,
        stream=False,
    )

    if not isinstance(plan, models.TaskPlan):
        return

    logger.info(f"Plan:\n{plan.model_dump_json(indent=2)}")

    end_first_call = time.time()
    print(
        "time taken for first API call and code execution:",
        end_first_call - start_first_call,
    )
    print("\n")

    synthesiser_prompt = models.synthesiser_prompt.format(
        query=query,
        code_to_execute=plan.code_to_execute,
        result=plan.result,
    )
    logger.info(f"synthesiser prompt: {synthesiser_prompt}")

    logger.info(f"Generating final answer for query")
    response, error = api_function_call(
        system_message=models.system_message_synthesiser,
        query=synthesiser_prompt,
        model="gpt-4o",
        stream=True,
    )

    console = Console()

    if isinstance(response, ChatCompletion):
        console.print(Markdown(str(response.choices[0].message.content)))
    else:  # Handle streaming response (generator)
        for chunk in response:
            console.print(chunk, end="")
        console.print()

    end = time.time()
    print("time taken for second API call:", end - end_first_call)
    print("time taken for whole process:", end - start_first_call)


if __name__ == "__main__":
    main()
