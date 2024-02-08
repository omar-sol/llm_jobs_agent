import asyncio
from typing import List, Generator, Literal
import time
import argparse

from openai import OpenAI
from pydantic import Field, BaseModel
import instructor

client = instructor.patch(OpenAI())


system_message = """- You are a world-class task-planning algorithm capable of breaking down user questions into a simple graph of tasks.
- By executing every task, we can answer the user's question. 
- Provide a correct compute graph with suitable specific tasks to ask and relevant subtasks. 
- Before generating the list of tasks, think step by step to understand the problem better.
- You have a Pandas dataframe at your disposal. Remember that some values might be None or NaN.
- The name of the dataframe is `df.` and every value is lowercase.

- Here are more details about the Dataframe, along with the ways you can filter each column, either 'precise' or 'semantic':
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

- Here are the only valid values for some columns, which you can use to filter in a 'precise' way: 
remote_answer: Literal["remote", "hybrid", "onsite"]
job_type_answer: Literal["full-time", "part-time", "contract", "internship", "freelance", "temporary", "other"]
experience_level: Literal["senior", "mid-level", "entry-level"]

- You must use a print statement to display the output when generating code. 
- When computing the salary values, group computations over the same currency by checking the `salary_currency` column.
- When computing over numerical values, make sure not to round the values.
- Each task returns a result that the next task can use. That means you don't have to filter the dataframe multiple times for the same column.
- A task cannot be 'semantic' and use a 'precise' column and vice versa.
- A precise task that generates code cannot filter or use a 'semantic' column and vice versa.
- Always return all the columns, not just the job_title, when asked for jobs."""


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


class Precise(BaseModel):
    """
    Class representing the code to execute on the Pandas dataframe.
    """

    code: str = Field(
        description="The code to execute on the Pandas dataframe. Make sure to return the dataframe at the end of the code.",
    )


class Semantic(BaseModel):
    """
    Class representing the semantic task.
    """

    text_to_embed: str = Field(
        description="Sentence to embed if the task is semantic.",
    )
    column_to_filter: str = Field(
        description="Column to filter on if the task is semantic.",
    )


class Task(BaseModel):
    """
    Class representing a single task in a task plan.
    Types of tasks:
    - precise: Task needs numeric computation or precise string filtering to answer.
    - semantic: Task needs semantic understanding to answer, such as filtering with unprecise strings.
    """

    id: int = Field(..., description="Unique id of the task")
    task_type_reasoning: str = Field(
        description="Explain how you can determine the task type. Think step-by-step. Write down your reasoning here.",
    )

    task_type: Precise | Semantic

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
        return TaskResult(task_id=self.id, result=f"`{self.task_type}`")


class TaskPlan(BaseModel):
    """
    Generates a plan of tasks representing a tree of tasks and subtasks.
    Make sure every task is in the tree, and every task is done only once.
    """

    validation_model: ValidationModel
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

    query = "what is the average salary of senior data scientists?"
    print(query)
    print("model:", model)

    start = time.time()
    plan = task_planner(query, model, args.max_retries)
    end = time.time()

    print(plan.model_dump_json(indent=2))
    print(plan._get_execution_order())
    print(plan.task_graph)

    print("time taken for API call:", end - start)
