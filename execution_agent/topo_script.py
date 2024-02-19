import asyncio
from typing import List, Generator, Literal
import time
import argparse

from openai import OpenAI
from pydantic import Field, BaseModel
import instructor

client = instructor.patch(OpenAI())


system_message = """- You are a world-class task-planning algorithm capable of breaking down user questions into a simple graph of tasks.
- By executing every task, you can answer the user's question.
- Provide a correct compute graph with suitable specific tasks to ask and relevant subtasks. 
- Before generating the list of tasks, think step by step to understand the problem better.
- You have a Pandas dataframe at your disposal. Remember that some values might be None or NaN.
- The name of the dataframe is `df.` and every value is lowercase.

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

- When generating code, you must include a print statement to display the result. 
- When computing the salary values, group computations over the same currency by checking the `salary_currency` column.
- When computing over numerical values, make sure not to round the values.
- Each task returns a result that the next task can use. That means you don't have to filter the dataframe multiple times for the same column.
- A task cannot be 'semantic' and use a 'precise' column and vice versa.
- A precise task that generates code cannot filter or use a 'semantic' column and vice versa.
- Always return all the columns, not just the job_title, when asked for jobs.
"""


class ValidationModel(BaseModel):
    """
    Validates the user query.
    """

    user_query: str = Field(
        description="The user's query. This is the question you must answer.",
    )
    chain_of_thought: str = Field(
        description="Is the user query related to a job-board or for a job counselor?. Think step-by-step. Write down your chain of thought here.",
    )
    is_valid: bool = Field(
        description="Based on the previous reasoning, answer with True if the query is related to a job-board. answer False otherwise.",
    )


class TaskResult(BaseModel):
    task_id: int
    result: str


class TaskResults(BaseModel):
    results: List[TaskResult]


# class Precise(BaseModel):
#     """
#     Class representing the code to execute on the Pandas dataframe.
#     """

#     code: str = Field(
#         description="The code to execute on the Pandas dataframe. Make sure to return the dataframe at the end of the code.",
#     )


# class Semantic(BaseModel):
#     """
#     Class representing the semantic task.
#     """

#     text_to_embed: str = Field(
#         description="Sentence to embed if the task is semantic.",
#     )
#     column_to_filter: str = Field(
#         description="Column to filter on if the task is semantic.",
#     )


class Task(BaseModel):
    """
    Class representing a single task in a task plan.
    """

    id: int = Field(..., description="Unique id of the task")
    task_reasoning: str = Field(
        description="Explain how you can determine the task to perform. Think step-by-step. Write down your reasoning here.",
    )
    task: str = Field(
        description="Using the reasoning above, describe the task to perform.",
    )
    tasks_to_before: List[int] = Field(
        default_factory=list,
        description="""List of the IDs of tasks that need to be executed before this question.""",
    )

    async def aexecute(self, with_results: TaskResults) -> TaskResult:
        """
        Executes the task by asking the question and returning the answer.
        """

        # We do nothing with the subtask answers, since this is an example however
        # we could use intermediate results to compute the answer to the main task.
        return TaskResult(task_id=self.id, result=f"`{self.task}`")


class TaskPlan(BaseModel):
    """
    Generates a plan of tasks representing a tree of tasks and subtasks that will answer the user's query.
    Make sure every task is in the tree, and every task is done only once.
    """

    # validation_model: ValidationModel
    user_query: str = Field(
        description="The user's query. This is the question you must answer with the task plan.",
    )
    task_plan_reasoning: str = Field(
        description="Explain how you can determine the graph task plan. Think step-by-step. Write down your reasoning here.",
    )
    task_graph: List[Task] = Field(
        ...,
        description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.",
    )

    def _get_execution_order(self) -> List[int]:
        """
        Returns the order in which the tasks should be executed using topological sort.
        Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
        """
        tmp_dep_graph = {item.id: set(item.tasks_to_before) for item in self.task_graph}

        def topological_sort(
            dep_graph: dict[int, set[int]],
        ) -> Generator[set[int], None, None]:
            while True:
                ordered = set(item for item, dep in dep_graph.items() if len(dep) == 0)
                if not ordered:
                    break
                yield ordered
                dep_graph = {
                    item: (dep - ordered)
                    for item, dep in dep_graph.items()
                    if item not in ordered
                }
            if len(dep_graph) != 0:
                raise ValueError(
                    f"Circular dependencies exist among these items: {{{', '.join(f'{key}:{value}' for key, value in dep_graph.items())}}}"
                )

        result = []
        for d in topological_sort(tmp_dep_graph):
            result.extend(sorted(d))
        return result

    async def execute(self) -> dict[int, TaskResult]:
        """
        Executes the tasks in the task plan in the correct order using asyncio and chunks with answered dependencies.
        """
        execution_order = self._get_execution_order()
        tasks = {q.id: q for q in self.task_graph}
        task_results = {}
        while True:
            ready_to_execute = [
                tasks[task_id]
                for task_id in execution_order
                if task_id not in task_results
                and all(
                    subtask_id in task_results
                    for subtask_id in tasks[task_id].tasks_to_before
                )
            ]
            # prints chunks to visualize execution order
            print(ready_to_execute)
            computed_answers = await asyncio.gather(
                *[
                    q.aexecute(
                        with_results=TaskResults(
                            results=[
                                result
                                for result in task_results.values()
                                if result.task_id in q.tasks_to_before
                            ]
                        )
                    )
                    for q in ready_to_execute
                ]
            )
            for answer in computed_answers:
                task_results[answer.task_id] = answer
            if len(task_results) == len(execution_order):
                break
        return task_results


Task.model_rebuild()
TaskPlan.model_rebuild()


def task_planner(question: str, model, max_retries) -> TaskPlan:
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_model=TaskPlan,
        messages=messages,
        max_tokens=1000,
        max_retries=max_retries,
    )

    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tags from a specified row in a CSV file."
    )
    parser.add_argument("--query", type=str, help="Question to ask", required=False)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        help="Model to use for the OpenAI API",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Number of retries for the OpenAI API",
    )

    args = parser.parse_args()
    query = args.query

    model_mapping = {
        "gpt3.5": "gpt-3.5-turbo-1106",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpt4": "gpt-4-0125-preview",
    }
    model = model_mapping[args.model]

    # query = "what is the average salary of senior data scientists?"
    query = "What are the key differences between a data scientist and a data engineer?"
    print(query)
    print("model:", model)

    start = time.time()
    plan = task_planner(query, model, args.max_retries)
    end = time.time()

    print(plan.model_dump_json(indent=2))
    print(plan._get_execution_order())
    print(plan.task_graph)

    print("time taken for API call:", end - start)
