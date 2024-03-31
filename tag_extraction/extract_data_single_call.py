import argparse
import os
import time

import tiktoken
import pandas as pd

from nested_structure import JobDetails
from call_openai_api import api_function_call


def main():
    parser = argparse.ArgumentParser(
        description="Extract tags from a specified row in a CSV file."
    )
    parser.add_argument(
        "--index", type=int, help="Row index to extract data from", required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for the OpenAI API",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="Number of retries for the OpenAI API",
    )
    args = parser.parse_args()

    row_index = args.index

    model_mapping = {
        "gpt3.5": "gpt-3.5-turbo-0125",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpt4": "gpt-4-0125-preview",
    }
    model = model_mapping[args.model]
    print("model:", model)

    df1 = pd.read_json("data/formatted_jobs_feb5_24.json")
    # df1 = pd.read_csv(
    #     "data/jobs_cleaned_data.csv",
    #     low_memory=True,
    # )
    print("number of rows:", len(df1))

    # convert to string, else its not json serializable
    df1["created_at"] = df1["created_at"].astype(str)
    df1 = df1.rename(columns={"title": "job_title"})

    df1 = df1.dropna(subset="cleaned_description")
    print("number of rows after dropping NA:", len(df1))
    # df1 = df1.rename(columns={"cleaned_description": "content"})

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    df1["num_tokens"] = df1["cleaned_description"].apply(
        lambda x: num_tokens_from_string(x, "cl100k_base")
    )

    # Now, you can filter the DataFrame to only keep rows with num_tokens less than 8000
    df1 = df1[df1["num_tokens"] < 8000]
    print("number of rows below <8000:", len(df1))
    df1.drop("num_tokens", axis=1, inplace=True)
    print("number of rows after removing >8000:", len(df1))

    job_title = df1["job_title"][row_index]
    cleaned_description = df1["cleaned_description"][row_index]

    print("job_title:", job_title)
    print("cleaned_description:", cleaned_description)
    print("\n")

    system_message = (
        "You are a world class extractor of information in from messy job postings."
    )
    query = f"Job title:\n {job_title}\n Job posting:\n {cleaned_description}"

    start = time.time()
    job_details, error = api_function_call(
        system_message, query, model, JobDetails, args.max_retries, False
    )
    end = time.time()
    print("\n")
    response_dict = job_details.model_dump()
    for key, value in response_dict.items():
        print(f"- {key}: {value} \n")
    # print(job_details.model_dump_json(indent=2))

    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
    # print("output tokens:", len(encoding.encode(str(response_dict))))
    print("time taken:", end - start)


if __name__ == "__main__":
    # TODO: specify the range of rows to extract data from ex. 0-1000
    # TODO: make concurrent requests to the API, and save the output to a json file, while avoiding rate limits
    main()
