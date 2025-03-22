import logging

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


system_message_validation = """You are a world-class expert who knows about every job on a job board. You can provide guidance and answer questions, but first, you need to validate if the query is in the context of jobs in general. So queries about skills, salaries, job types, and job locations are all valid."""


class QueryValidation(BaseModel):
    """
    Validate the user query. Ensure the query is related to a job board or a job counselor.
    """

    chain_of_thought: str = Field(
        description="Is the user query related to a job board or for a job counselor? Think step-by-step. Write down your chain of thought here.",
    )
    is_valid: bool = Field(
        description="Based on the previous reasoning, answer with True if the query is related to a job board. Answer False otherwise.",
    )
    reason: str = Field(
        description="Explain why the query is valid or not. What are the keywords that make it valid?",
    )


# -----------------------------------------------------------------------------------------------------

system_message_plan = """You are a world-class task-planning algorithm and developer capable of breaking down user questions into a solvable snippet of Python code.
You have a Pandas dataframe at your disposal. Remember that some values might be `None` or `NaN`.
The name of the dataframe is `df` and its case insensitive.
Remember: You cannot subset columns with a tuple with more than one element. Use a list instead.

Here are the headings and a brief description for each:
* Invoice: A unique identifier for each transaction.
* StockCode: A unique product identifier code.
* Description: The name/description of the product (some values may be missing when products lack formal descriptions but can still be identified by StockCode).
* Quantity: The number of items purchased (may include negative values representing returned items).
* InvoiceDate: The timestamp when the transaction occurred.
* Price: The unit price of the product (may include negative values for refunds, discounts, or adjustments).
* Customer ID: A unique identifier for the customer (missing values likely represent guest purchases without account registration).
* Country: The country where the transaction originated.

Here are more details created with df.info():

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 525461 entries, 0 to 525460
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   Invoice      525461 non-null  object        
 1   StockCode    525461 non-null  object        
 2   Description  522533 non-null  object        
 3   Quantity     525461 non-null  int64         
 4   InvoiceDate  525461 non-null  datetime64[ns]
 5   Price        525461 non-null  float64       
 6   Customer ID  417534 non-null  float64       
 7   Country      525461 non-null  object        
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 32.1+ MB

Note on empty values:
- Description: ~2,928 missing values (0.6%) - Products may still be identifiable by StockCode
- Customer ID: ~107,927 missing values (20.5%) - Likely represent guest purchases without account registration
- All other columns have complete data as they are essential for transaction tracking

Here are some rules to follow:
- You must use print statements to display relevant execution results.
- When computing over numerical values, make sure not to round the values.
"""


class TaskPlan(BaseModel):
    """- Generates Python code to be executed over a Pandas dataframe. Avoid including import statements.
    - If the query involves filtering a semantic column, provide variations of this phrase or similar terms that could mean the same thing.
    - You must use a print statement at the end to display the output but only print the relevant columns if necessary.
    """

    user_query: str = Field(
        description="The user query that you need to answer. This is the question you need to answer using the pandas dataframe.",
    )
    chain_of_thought: str = Field(
        description="How will you answer the user_query using the pandas dataframe. Think step-by-step. Write down your chain of thought and reasoning. What will you print as a result? Will the code be free of bugs?",
    )
    code_to_execute: str = Field(
        description="Based on the previous reasoning, write bug-free code for the `python_repl` tool. Make sure to write code without bugs. Avoid import statements. Print the relevant columns.",
    )
    is_code_bug_free: bool = Field(
        description="Reflect on the previously generated code, answer with True if the code is safe, will run without issues and answers the user query. Answer False otherwise. Does it have extra indentations?",
    )
    result: str = Field(
        default="",
        description="The result of the code execution. If the code has not been executed yet, leave this field empty.",
    )

    @model_validator(mode="after")
    def verify_code(self):
        logger.info("Verifying code")
        result = self.execute_code()
        if "Error" in result or "Exception" in result:
            self.is_code_bug_free = False
            logger.error(f"An error occurred: {result}")
            raise ValueError(f"An error occurred: {result}")
        logger.info(f"code execution result: {result}")
        self.result = result
        return self

    def execute_code(self):
        import io
        import sys
        from contextlib import redirect_stdout

        # Create a dictionary to hold the global variables
        globals_dict = {}
        # python_repl = PythonREPL(_globals=globals_dict)

        # Set up code to execute with imports and data loading
        import_and_load = """import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 400)
# Load the dataframe
df = pd.read_excel("exercise_2/online_retail_II.xlsx", sheet_name="Year 2009-2010")
"""
        code: str = import_and_load + self.code_to_execute
        logger.info(f"Code that will be executed: \n{code}")

        # Capture stdout to get the results
        output_buffer = io.StringIO()

        try:
            with redirect_stdout(output_buffer):
                exec(code, globals_dict)
            result = output_buffer.getvalue()
        except Exception as e:
            result = f"Error: {str(e)}"

        return result


# -----------------------------------------------------------------------------------------------------

system_message_synthesiser = """- You are a world-class data analyst, your task is to answer the user query in a way that is helpful and complete.  
- The answer must include all the information you have at your disposal.
- At your disposal, you have the results of executed Python code.
- The executed code was used a Python Pandas Dataframe containing online retrail data.
- Users do not see the code or its output. They only see your answer. Use the information to generate a complete and helpful reply.
- Use Markdown to format your answer. Use headings, bold, italics, and lists to make your answer clear and easy to read.
- Provide the user with all the information; do not cut down your answer.
- If the executed code result is empty, state that no information is available in the database.
"""


synthesiser_prompt = """user_query: {query}

exec_tool_output: {result} 

REMEMBER: That you are a data analyst. Give a complete answer to the user question.
Avoid short answers, avoid statements like '...and more.'
Instead provide complete and thorough answers.
"""


class SynthesiserResponse(BaseModel):
    """
    Generate and answer to the user. Use Markdown to format your answer. Use headings, bold, italics, and lists to make your answer clear and easy to read.
    Make sure to give a complete and helpful answer.
    If the exec_tool result is empty, state that no information is available in our database.
    """

    chain_of_thought: str = Field(
        description="Given the information, how will you answer the user query? Think step-by-step. Write down your chain of thought and reasoning.",
    )
    answer: str = Field(
        description="Answer given to the user. Based on the previous reasoning, generate a complete and helpful answer to the user. Use Markdown to format the text.",
    )
    reflect: str = Field(
        description="Did you give a complete answer?",
    )
