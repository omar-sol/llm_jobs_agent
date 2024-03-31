"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --model gpt-3.5-turbo-1106 \
  --max_requests_per_minute 10_000 \
  --max_tokens_per_minute 1_000_000 \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 0, 1, 2, ...)
    - Run main()
"""

# import aiohttp  # for making API calls concurrently
# import re  # for matching endpoint from request URL
import os  # for reading API key
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field

import tiktoken  # for counting tokens
import openai
from openai import AsyncOpenAI
import instructor
import pandas as pd
from dotenv import load_dotenv

from nested_structure import JobDetails

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    model: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # DEBUG, INFO, WARNING, ERROR, and CRITICAL
    logging.basicConfig(level=logging_level)

    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    logging.info(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
    aclient = instructor.apatch(AsyncOpenAI(api_key=OPENAI_API_KEY))

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()

    next_request: APIRequest | None = None  # variable to hold the next request to call

    # read existing results to avoid reprocessing
    processed_requests = read_existing_results(save_filepath)

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_unfinished = True  # refers to whether DataFrame rows are all processed
    logging.debug(f"Initialization complete.")

    # initialize file reading
    df1 = pd.read_json("data/formatted_jobs_feb5_24.json")
    df1["created_at"] = df1["created_at"].astype(str)
    df1 = df1.dropna(subset="cleaned_description")

    logging.debug("File read and dataframe created.")

    row_index = 0
    while True:
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
            elif file_unfinished:
                if row_index < len(df1):
                    try:
                        row_info_dict = df1.iloc[row_index].to_dict()
                        # Check if request has already been processed
                        current_request_url = row_info_dict["jobs_towardsai_url"]
                        if current_request_url in processed_requests:
                            row_index += 1
                            continue

                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            row_info_dict=row_info_dict,
                            token_consumption=num_tokens_consumed_from_request(
                                row_info_dict["cleaned_description"]
                            ),
                            attempts_left=max_attempts,
                            model=model,
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1

                        row_index += 1
                    except KeyError:
                        logging.debug("Invalid row index")
                        file_unfinished = False
                else:
                    logging.debug("All DataFrame rows processed")
                    file_unfinished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            # logging.debug(
            #     f"Request {next_request.task_id} requires {next_request_tokens} tokens"
            # )
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        client=aclient,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    token_consumption: int
    attempts_left: int
    row_info_dict: dict
    model: str
    result: list = field(default_factory=list)

    async def call_api(
        self,
        client: AsyncOpenAI,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        error = None
        query = f"Job title:\n {self.row_info_dict['title']}\n Job posting:\n {self.row_info_dict['cleaned_description']}"
        try:
            response = await client.chat.completions.create(
                model=self.model,
                response_model=JobDetails,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world class extractor of information in from messy job postings.",
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                max_retries=4,
            )
        except openai.BadRequestError as e:
            error = e
            status_tracker.num_api_errors += 1
            logging.exception(
                f"Invalid request: {self.task_id} to OpenAI API. See traceback:"
            )

        except openai.RateLimitError as e:
            error = e
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            logging.exception(
                f"RateLimit error {self.task_id} from OpenAI. See traceback:"
            )

        except Exception as e:
            error = e
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.task_id} failed after all attempts. Saving errors: {self.result}"
                )
                self.row_info_dict["extracted_info"] = [str(e) for e in self.result]
                data = self.row_info_dict

                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            response_dict = response.model_dump()
            self.row_info_dict["extracted_info"] = response_dict
            data = self.row_info_dict

            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    input_text: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
    messages = [
        {
            "role": "system",
            "content": "You are a world class extractor of information in from messy job postings.",
        },
        {"role": "user", "content": input_text},
    ]

    num_tokens = 0
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        for key, value in message.items():
            try:
                num_tokens += len(encoding.encode(value))
            except TypeError as e:
                logging.error(f"TypeError encountered in encoding: {e}")
                # You may choose to continue, break, return or handle it in any other way
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + 650


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def read_existing_results(filepath):
    if not os.path.exists(filepath):
        return set()

    processed_requests = set()
    with open(filepath, "r") as file:
        for line in file:
            data = json.loads(line)
            request_url = data[
                "jobs_towardsai_url"
            ]  # Adjust to match your data structure
            processed_requests.add(request_url)

    return processed_requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--requests_filepath", default="cleaned_data.csv")
    parser.add_argument(
        "--requests_filepath", default="data/formatted_jobs_feb5_24.json"
    )
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--max_requests_per_minute", type=int, default=5_000 * 0.75)
    parser.add_argument("--max_tokens_per_minute", type=int, default=160_000 * 0.75)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".json", "_results.jsonl")

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            model=args.model,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )
