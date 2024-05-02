import os
import logging

import tiktoken
import instructor
import openai
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def api_function_call(
    system_message: str,
    query: str,
    model: str = "gpt-4-turbo",
    response_model=None,
    max_retries: int = 0,
    stream: bool = False,
    max_tokens: int = 4000,
):

    client = instructor.from_openai(OpenAI())
    try:
        message_data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            "max_retries": max_retries,
            "stream": stream,
            "max_tokens": max_tokens,
            "response_model": response_model,
        }
        # if response_model is not None:
        # message_data["response_model"] = response_model

        if stream and response_model is not None:
            response = client.chat.completions.create_partial(**message_data)
            error = False
        else:
            response = client.chat.completions.create(**message_data)
            error = False

    except openai.BadRequestError:
        error = True
        logger.exception("Invalid request to OpenAI API. See traceback:")
        error_message = (
            "Something went wrong while connecting with OpenAI, try again soon!"
        )
        return error_message, error

    except openai.RateLimitError:
        error = True
        logger.exception("RateLimit error from OpenAI. See traceback:")
        error_message = "OpenAI servers seem to be overloaded, try again later!"
        return error_message, error

    except Exception as e:
        error = True
        logger.exception(
            "Some kind of error happened trying to generate the response. See traceback:"
        )
        error_message = (
            "Something went wrong with connecting with OpenAI, try again soon!"
        )
        return error_message, error

    if stream is True and response_model is None:
        # We are entering streaming mode, so here we're just wrapping the streamed
        # openai response to be easier to handle later
        def answer_generator():
            for chunk in response:
                token = chunk.choices[0].delta.content

                # Always stream a string, openAI returns None on last token
                token = "" if token is None else token

                yield token

        return answer_generator(), error

    else:
        return response, error
