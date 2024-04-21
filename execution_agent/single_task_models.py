import logging
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError, model_validator
from langchain.utilities.python import PythonREPL

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


system_message_plan = """You are a world-class task-planning algorithm and developer capable of breaking down user questions into a solvable snippet of code.
You have a Pandas dataframe at your disposal. Remember that some values might be `None` or `NaN`.
The name of the dataframe is `df` and you are case insensitive.
Remember: You cannot subset columns with a tuple with more than one element. Use a list instead.


The file's field headings and a brief description for each are as follows:
* job_id: A unique identifier for each job listing.
* created_at: The timestamp of the job listing creation date.
* job_title: The title of the job position.
* job_skills: A comma-separated list of skills, tools, frameworks, etc., relevant to the job.
* job_type: Specifies whether the job is full-time, part-time, intern, etc.
* company_id: A unique identifier for the company posting the job.
* apply_url: The URL where applicants can apply for the job.
* city: The city where the job is located.
* country: The country code where the job is located.
* salary: A field for salary, which is a non-standard format and can be empty.
* salary_min: The minimum salary offered for the position.
* salary_max: The maximum salary offered for the position.
* salary_currency: The currency for the salary offered.
* url_slug: A slug that links to the specific job listing on the Towards AI job board.
* role_description: A summary description of the job.
* company_name: The name of the company offering the job.
* company_description: A description of the company.
* cleaned_description: A full version of the job description text.
* scraped_skills_required: Skills required for the job (this only has data for some jobs).
* scraped_skills_useful: Skills that are useful but not necessarily required for the job (this only has data for some jobs).


Here are more details created with df.info():

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8165 entries, 0 to 8164
Data columns (total 20 columns):
 #   Column                   Non-Null Count  Dtype         
---  ------                   --------------  -----         
 0   job_id                   8165 non-null   int64         
 1   created_at               8165 non-null   datetime64[ns]
 2   job_title                8165 non-null   object        
 3   job_skills               8165 non-null   object        
 4   job_type                 8165 non-null   object        
 5   company_id               8165 non-null   int64         
 6   apply_url                8165 non-null   object        
 7   city                     8165 non-null   object        
 8   country                  8165 non-null   object        
 9   salary                   8165 non-null   object        
 10  salary_min               8165 non-null   int64         
 11  salary_max               8165 non-null   int64         
 12  salary_currency          8165 non-null   object        
 13  url_slug                 8165 non-null   object        
 14  role_description         8165 non-null   object        
 15  company_name             8165 non-null   object        
 16  company_description      8165 non-null   object        
 17  cleaned_description      8165 non-null   object        
 18  scraped_skills_required  8165 non-null   object        
 19  scraped_skills_useful    8165 non-null   object        
dtypes: datetime64[ns](1), int64(4), object(15)
memory usage: 1.2+ MB
None

Here are some rules to follow:
- You must use a print statement to display the output code exectution result, create a dictionary with the relevant results, and print that dict.
- If users ask for a list of jobs (rows in the dataframe), ALWAYS include the `url_slug,` `role_description`, and `company_description` columns in the print statement. 
- If users ask for a list of jobs, sort them by `creation_date` and only include the most recent 20 twenty jobs.
- NEVER print the values in the `cleaned_description` column. Only use it for filtering.

- When computing salary values, group computations over the same currency and salary frequency. Check the `salary_currency` and `salary_frequency` column.
- Check the currency with the `salary_currency` column if the question involves a salary computation. You can't assume the currency. Average values over different currencies or salary frequencies are not valid.
- When asked about salary, keep the minimum and maximum salary values separate.
- When computing over numerical values, make sure not to round the values.

- When filtering for skills with keywords, use the `cleaned_description` column. Also provide variations of the keyword (e.g., "data scientist", "data science", "data analysis").
- When extracting job skills, use the `job_skills` column.
- Extracting job skills might result in repeated skills; count them and return the most common skills.

- When filtering for job titles, use the `job_title`, `cleaned_description` and `experience_level` columns and provide keyword variations for them. (e.g., "data scientist", "data science", "data analysis").
- When filtering for experience level, use the `experience_level` column OR filter with 'junior', 'senior' in the `job_title`. 

- When filtering semantic columns, capture all possible variations of the keyword (e.g., "junior data scientist", "jr. data scientist", "entry-level data scientist")
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
df = pd.read_json('data/db_info.json')
"""
        code: str = import_and_load + self.code_to_execute
        logger.info(f"code to execute: {code}")
        result: str = python_repl.run(code)
        return result


system_message_synthesiser = """- You are a world-class job counselor—your task is to answer the user question by giving helpful, complete, and friendly answers with all the information you have at your disposal.
- To help you give a complete answer to the user question, you have the executed Python code and the result. The code was used over a Python Pandas Dataframe containing job listing data.
- Users do not see the code or its output. They will only see your answer. So, use the code output to generate a complete and helpful reply.
- Use Markdown to format your answer. Use headings, bold, italics, and lists to make your answer clear and easy to read.
- If the question asks about a list of jobs, please answer ALL jobs with a summary.
- If you are providing a list of jobs, also provide an URL link to the job listing by appending the `url_slug` to the string 'https://jobs.towardsai.net/job/'
- If the python_repl did not produce a `url_slug`, do not link to any website; DO NOT create new links.
- If you didn't receive a `url_slug`, DO NOT share a new one. Avoid using, "and you may explore the job listings on the website" or similar sentences.
- Make sure to answer with the complete list of jobs if the user asks for it.
- Provide the user with all the information; do not cut down your answer. If the repl_tool has 20 `url_slug`, you must also output 20 links to the user. with 'https://jobs.towardsai.net/job/'+ slug_url
- If the repl_tool result is empty, state that no information is available in our database.
"""


class SynthesiserResponse(BaseModel):
    """
    Generate and answer to the user. Use Markdown to format your answer. Use headings, bold, italics, and lists to make your answer clear and easy to read.
    Make sure to give a complete and helpful answer within a single response.
    If the repl_tool result is empty, state that no information is available in our database.
    """

    chain_of_thought: str = Field(
        description="Given the input, how will you answer the user question? Think step-by-step. Write down your chain of thought and reasoning.",
    )
    answer: str = Field(
        description="Based on the previous reasoning, generate a complete and helpful answer to the user. Use Markdown to format the text.",
    )
    reason: str = Field(
        description="Why did you answer the way you did?",
    )


synthesiser_prompt = """
REMEMBER: That you are a job counselor. Give a complete answer to the user question and do not cut down your answer. If you are given 20 twenty urls, you must also output 20 twenty urls to the user
Avoid short answers, avoid statements like '...and more.'. Provide a complete and helpful answer. User need to know about all the jobs available to them. Do not summarize your answer.
user_question: {query} 
python_code: {code_to_execute} 
repl_tool_output: {result} 

REMEMBER: Make sure to give a complete and useful answers to the user question and not cut down your answer. If the repl_tool has 20 urls, you must also output 20 urls to the user
Avoid concise answers, avoid unhelpful statements such as '...and more.'. Provide a complete answers. User need to know about all the jobs available to them. Do not summarize or cut down your answer.
"""
