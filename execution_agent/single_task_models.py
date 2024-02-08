import logging
from langchain.utilities.python import PythonREPL
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

system_message_synthesiser = """- You are a world-class job counselor—your task is to understand the user question and give complete, helpful, and friendly answers.
- To help you answer the user question, you will be given the code that was executed by a `python_repl` tool and its results. That way you can also explain how the result was computed. The code used a Python Pandas Dataframe containing job listings data.
- If the question asks about a list of jobs, please return a maximum of the ten first jobs with a short answer as to why they were selected. While listing the jobs, also, provide the full jobs_towardsai_url link for each of them so users can access the job listing themselves.
"""


system_message = """You are a world-class task-planning algorithm and developer capable of breaking down user questions into a list of tasks.
You have a Pandas dataframe at your disposal. Remember that some values might be `None` or `NaN`.
The name of the dataframe is `df.` and every value is lowercase.

Here are more details about the Dataframe, along with the ways you can filter each column, either `precise` or `semantic`:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10609 entries, 0 to 10608
Data columns (total 22 columns):
 #   Column                      Non-Null Count  Dtype  Type of filtering 
---  ------                      --------------  -----  -----------------
 0   job_title                   10609 non-null  object  semantic
 1   job_skills                  10609 non-null  object  semantic
 2   apply_url                   10609 non-null  object  precise
 3   city                        9090 non-null   object  semantic
 4   country                     9863 non-null   object  semantic
 5   company_id                  10609 non-null  float64 precise
 6   involves_ai                 10609 non-null  bool    precise 
 7   jobs_towardsai_url          10609 non-null  object  precise
 8   job_listing_text            10609 non-null  object  semantic
 9   company_name                10608 non-null  object  precise
 10  company_info                10609 non-null  object  semantic
 11  role_description            10608 non-null  object  semantic
 12  preferred_skills            6912 non-null   object  semantic
 13  salary_currency             3238 non-null   object  precise
 14  job_type_answer             8568 non-null   object  precise
 15  experience_years            4360 non-null   float64 precise
 16  experience_level            2792 non-null   object  precise
 17  regions_or_states           3613 non-null   object  semantic
 18  location_based_eligibility  1152 non-null   object  semantic
 19  remote_answer               5981 non-null   object  precise
 20  salary_min                  2886 non-null   float64 precise
 21  salary_max                  2886 non-null   float64 precise
dtypes: bool(1), float64(4), object(17)
memory usage: 1.7+ MB
None
(10609, 22)

Here are the only valid values for some columns, which you can use to filter in a precise way: 
remote_answer: Literal["remote", "hybrid", "onsite"]
job_type_answer: Literal["full-time", "part-time", "contract", "internship", "freelance", "temporary", "other"]
experience_level: Literal["senior", "mid-level", "entry-level"]

Here are some rules to follow:
- You must use a print statement to display the output code. 
- If users ask for a list of jobs (rows in the dataframe), only include the relevant columns in the print statement. Always include the `jobs_towardsai_url` column. 
- Never print the values in the `job_listing_text` column as it is too long.
- When computing over salary values, group computations over the same currency by checking the `salary_currency` column.
- Check the currency with the `salary_currency` column if the question involves a salary computation.
- When computing over numerical values, make sure not to round the values.
- When filtering for skills, use the 'job_listing_text' column.
"""


class QueryValidation(BaseModel):
    """
    Validates the user query. Makes sure the query is related to a job board or for a job counselor?.
    """

    chain_of_thought: str = Field(
        description="Is the user query related to a job board or for a job counselor? Think step-by-step. Write down your chain of thought here.",
    )
    is_valid: bool = Field(
        description="Based on the previous reasoning, answer with True if the query is related to a job board. Answer False otherwise.",
    )


class TaskPlan(BaseModel):
    """- Generates Python code to be executed over a Pandas dataframe. Avoid including import statements.
    - If the query involves filtering a semantic column, provide variations of this phrase or similar terms that could mean the same thing.
    - You must use a print statement at the end to display the output but only print the relevant columns if necessary.
    """

    query_validation: QueryValidation
    chain_of_thought: str = Field(
        description="How will you answer the user_query using the pandas dataframe. Think step-by-step. Write down your chain of thought. Will you print the output? Will the code be without bugs?",
    )
    code_to_execute: str = Field(
        description="Based on the previous reasoning, write bug-free code for the `python_repl` tool. Make sure to write code without bugs. Avoid import statements.",
    )
    is_code_bug_free: bool = Field(
        description="Based on the previously generated code, answer with True if the code is safe and will run without issues. Answer False otherwise.",
    )

    def execute_code(self) -> str:
        python_repl = PythonREPL()
        import_and_load = """import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 400)
# Load the dataframe
df = pd.read_pickle('data/extracted_cleaned_df.pkl')
"""
        code: str = import_and_load + self.code_to_execute
        logger.info(f"code to execute: {code}")
        result: str = python_repl.run(code)
        return result
