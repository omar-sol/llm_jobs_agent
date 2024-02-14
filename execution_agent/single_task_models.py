import logging
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError, model_validator
from langchain.utilities.python import PythonREPL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


system_message_validation = """You are a world-class expert that know about every job in a job board. You can provide guidance and answer questions, but first you need to validate if the query is in context of jobs in general. So queries about skills, salaries, job types, and locations are all valid."""


class QueryValidation(BaseModel):
    """
    Validates the user query. Makes sure the query is related to a job board or for a job counselor.
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


system_message_plan = """You are a world-class task-planning algorithm and developer capable of breaking down user questions into a list of tasks.
You have a Pandas dataframe at your disposal. Remember that some values might be `None` or `NaN`.
The name of the dataframe is `df.` and every value is lowercase.
Remember: You cannot subset columns with a tuple with more than one element. Use a list instead.

Here are more details about the Dataframe, along with the ways you can filter each column, either `precise` or `semantic`:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9105 entries, 0 to 9104
Data columns (total 26 columns):
 #   Column                Non-Null Count  Dtype   Type of filtering          
---  ------                --------------  -----   -----------------       
 0   job_id                9105 non-null   int64   precise
 1   creation_date         9105 non-null   datetime64[ns]  precise
 2   job_title             9105 non-null   object  semantic        
 3   job_skills            9105 non-null   object  semantic      
 4   company_id            9105 non-null   int64   precise       
 5   apply_url             9105 non-null   object  precise      
 6   city                  7913 non-null   object  semantic      
 7   country               8814 non-null   object  semantic      
 8   involves_ai           9105 non-null   bool    precise      
 9   company_name          9105 non-null   object  precise      
 10  job_listing_text      9105 non-null   object  semantic      
 11  jobs_towardsai_url    9105 non-null   object  precise      
 12  company_info          9105 non-null   object  semantic      
 13  role_description      9105 non-null   object  semantic      
 14  preferred_skills      6395 non-null   object  semantic      
 15  salary_reasoning      8008 non-null   object  precise      
 16  salary_frequency      2688 non-null   object  precise      
 17  salary_currency       2924 non-null   object  precise      
 18  job_type_answer       8073 non-null   object  precise      
 19  min_experience_years  5995 non-null   object  precise      
 20  regions_or_states     5497 non-null   object  semantic      
 21  continent             8178 non-null   object  semantic      
 22  experience_level      5519 non-null   object  precise      
 23  remote_answer         4494 non-null   object  precise      
 24  salary_min            2405 non-null   float64 precise      
 25  salary_max            2405 non-null   float64 precise      
dtypes: bool(1), datetime64[ns](1), float64(2), int64(2), object(20)
memory usage: 1.7+ MB
None
(9105, 26)

Here are the only valid values for some columns, which you can use to filter in a precise way: 
remote_answer: Literal["remote", "hybrid", "onsite"]
job_type_answer: Literal["full-time", "part-time", "contract", "internship", "freelance", "temporary", "other"]
experience_level: Literal["senior", "mid-level", "entry-level"]
salary_frequency: Literal["hourly", "monthly", "annually", "Not specified"]


Here are some rules to follow:
- You must use a print statement to display the output code. 
- If users ask for a list of jobs (rows in the dataframe), only include the relevant columns in the print statement but ALWAYS include the `jobs_towardsai_url` column. 
- If users ask a list of jobs, sort them by `creation_date` and only include the most recent 20 jobs.
- NEVER print the values in the `job_listing_text` column. only use it for filtering.

- When computing over salary values, group computations over the same currency and same salary frequency. Check the `salary_currency` and `salary_frequency` column.
- Check the currency with the `salary_currency` column if the question involves a salary computation. You cant't assume the currency. Average values over different currencies or salary frequencies are not valid.
- When asked about salary, keep the minimum and maximum salary values separate.
- When computing over numerical values, make sure not to round the values.

- When filtering for skills with keywords, use the `job_listing_text` column. Also provide variations of the keyword (e.g., "data scientist", "data science", "data analysis").
- When extracting job skills, use the `job_skills` column.
- Extracting job skills, might result in repeated skills, make sure to count them and return the most common skills.

- When filtering for experience level, use the `experience_level` column OR filter with 'junior', 'senior' in the `job_title`.
- When filtering semantic columns, capture all possible variations the keyword (e.g., "junior data scientist", "JR. Data Scientist", "entry-level data scientist")
"""


class TaskPlan(BaseModel):
    """- Generates Python code to be executed over a Pandas dataframe. Avoid including import statements.
    - If the query involves filtering a semantic column, provide variations of this phrase or similar terms that could mean the same thing.
    - You must use a print statement at the end to display the output but only print the relevant columns if necessary.
    """

    chain_of_thought: str = Field(
        description="How will you answer the user_query using the pandas dataframe. Think step-by-step. Write down your chain of thought and reasoning. What will you print as a result? Will the code be free of bugs?",
    )
    code_to_execute: str = Field(
        description="Based on the previous reasoning, write bug-free code for the `python_repl` tool. Make sure to write code without bugs. Avoid import statements. Print the relevant columns.",
    )
    is_code_bug_free: bool = Field(
        description="Based on the previously generated code, answer with True if the code is safe and will run without issues. Answer False otherwise. Does it have extra indentations?",
    )
    result: str = Field(
        default="",
        description="The result of the code execution. If the code has not been executed yet, leave this field empty.",
    )

    @model_validator(mode="after")
    def verify_code(self):
        result = self.execute_code()
        if "Error" in result or "Exception" in result:
            self.is_code_bug_free = False
            logger.error(f"An error occurred: {result}")
            raise ValueError(f"An error occurred: {result}")
        logger.info(f"code execution result: {result}")
        self.result = result
        return self

    def execute_code(self) -> str:
        python_repl = PythonREPL()
        import_and_load = """import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 400)
# Load the dataframe
df = pd.read_pickle('data/extracted_cleaned_df_feb5.pkl')
"""
        code: str = import_and_load + self.code_to_execute
        logger.info(f"code to execute: {code}")
        result: str = python_repl.run(code)
        return result


system_message_synthesiser = """- You are a world-class job counselor—your task is to understand the user question and give helpful, complete and friendly answers with the information you have.
- To help you answer the user question, you will be given the result of a `python_repl` tool. The code was used over a Python Pandas Dataframe containing job listing data.
- Users do not see the code or repl output. They will only see your answer.
- Use Markdown to format your answer. Use headings, bold, italics, and lists to make your answer easy to read.
- Never provide a direct link to the job board. If given to you, provide the `jobs_towardsai_url` link for each job.
- If the question asks about a list of jobs, please return a maximum of the twenty 20 first jobs with a summary. 
- If you are listing jobs, also provide the jobs_towardsai_url link for each of them so users can access the job listing themselves.
- If the python_repl did not produce a jobs_towardsai_url, do not link to any website, DO NOT create new links. 
- If you did not received an URL, do not create a new one. Avoid "any you may explore the job listings on the website" or similar sentences.
"""
