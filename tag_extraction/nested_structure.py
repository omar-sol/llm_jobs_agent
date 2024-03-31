from pydantic import BaseModel, Field
from typing import Literal


class SkillsModel(BaseModel):
    """
    Extracts the explicitly stated required/essential and preferred/desired skills for this role.
    Focuses only on specific technical skills and tools; excludes degrees and general experience.
    """

    chain_of_thought: str = Field(
        min_length=1,
        description="If there are any explicit specific statements regarding the required/non-negotiable and preferred/desired skills for the role (languages, tools and software libraries), write them here.",
    )
    required_skills: list[str] = Field(
        description="Based on the previous reasoning, state the specified required technical skills, such as tools, and software libraries for the role.",
        examples=[
            ["Not specified"],
            ["XGBoost", "Hugging Face", "PyTorch", "SQL"],
        ],
    )
    preferred_skills: list[str] = Field(
        description="Based on the previous reasoning, state the Preferred/Desired technical skills, tools, and software libraries.",
        examples=[
            ["Not specified"],
            ["NLP", "Customer Success Engineer", "Kubernetes"],
        ],
    )


class SalaryModel(BaseModel):
    """
    Extracts the salary details, allowing for reasonable inferences from the job posting, and includes any salary-related eligibility requirements.
    """

    salary_reasoning: str = Field(
        description="Are there any explicit statements regarding the salary details and salary-related eligibility requirements? Write the statements here.",
    )
    salary_numerical: list[float] | str = Field(
        description="Based on the previous reasoning, state the salary numerical amount or range [min, max], using explicit mentions, in numerical values. Exclude other compensation details.",
        examples=[
            [70_000],
            [80_000, 150_000],
            "Not specified",
        ],
    )
    salary_frequency: Literal["hourly", "monthly", "annually", "Not specified"] = Field(
        description="State the frequency of the salary amount or range.",
    )
    salary_currency: str = Field(
        description="State the currency of the salary amount or range. If the symbol $ is used, you can assume 'USD'",
        examples=["USD", "CAD", "EUR", "GBP", "JPY", "Not specified"],
    )


class JobTypeModel(BaseModel):
    """
    Extracts the job type, only allowing for logical inferences based on clear and strong context clues in the job posting (e.g., 'annual salary' suggests full-time).
    """

    job_type_reasoning: str = Field(
        description="Explain how you determine the job type for this role using context clues. Inferences must be based on strong and explicit clues. Avoid assumptions if the information is ambiguous or missing.",
    )
    job_type_answer: str = Field(
        description="Based on the previous reasoning, state the job type for this role/position. Only use the inferred type if it's strongly supported by the clues.",
        examples=[
            "Not specified",
            "full-time",
            "contract",
            "internship",
            "part-time",
            "temporary",
            "freelance",
            "Other",
        ],
    )


class RemoteModel(BaseModel):
    remote_reasoning: str = Field(
        description="Are there any statements regarding remote work or telework? Only write the statements but answer below.",
    )
    remote_answer: Literal["Remote", "Hybrid", "Onsite", "Not specified"] = Field(
        description="Using the statements, State if the work is remote, hybrid or onsite. 'Not specified' if it's not specified."
    )


class LocationModel(BaseModel):
    """Extracts the office location information, allowing for reasonable inferences from the job posting."""

    chain_of_thought: str = Field(
        description="Write any explicit statements regarding the office/onsite locations. Write the statements here.",
    )
    city: list[str] = Field(
        min_length=1,
        description="City of office location. Use the previous statements. Include multiple cities if applicable. Only cities are allowed. 'Not specified' if not mentioned.",
        examples=[["Tokyo", "San Francisco, Montreal"], ["Not specified"]],
    )
    regions_or_states: list[str] = Field(
        min_length=1,
        description="The region or state of the office location. Deduce from city location. 'Not specified' if not specified and can't be deduced. Avoid abbreviations.",
        examples=[["Quebec", "California", "Massachusetts"], ["Not specified"]],
    )
    country: list[str] = Field(
        min_length=1,
        description="The country of the office location. Infer from city or state information. If city is Miami, country is USA.",
        examples=[["India", "United States", "Germany"], ["Not specified"]],
    )
    continent: list[str] = Field(
        min_length=1,
        description="Continents of the office location, use the countries or cities information.",
        examples=[["North America", "Europe", "Asia"], ["Not specified"]],
    )


class InvolvesAIModel(BaseModel):
    chain_of_thought: str = Field(
        description="Are there any explicit statements regarding if the role involves working with AI, machine learning or data science.",
    )
    involves_ai: Literal["True", "False", "Not specified"] = Field(
        description="State if the role involves working with AI, machine learning, or data science. Can be extrapolated from the skills mentioned in the job posting and your knowledge of AI, ML, and data science.",
    )


class JobDetails(BaseModel):
    """
    Extracts and summarizes information from a whole job posting.
    """

    company_info: str = Field(
        description="Summarize the company information in the about us section. It's three to four sentences, focusing on how the company uses AI. ",
    )
    role_description: str = Field(
        description="Summarize the role/position details in four sentences. With a focus on how the role uses AI and which skills and tools are wanted or needed. (DO NOT mention location, salary info or company info here)",
    )
    skills_model: SkillsModel
    salary_model: SalaryModel
    job_type_model: JobTypeModel
    min_experience_years: int | str = Field(
        description="If mentioned, state the minimum required years of experience for this role. If '3+' only write 3.",
        examples=[
            3,
            5,
            "Not specified",
        ],
    )
    location_model: LocationModel
    # experience_level: str = Field(
    #     description="If mentioned, state career level for this role, such as 'Senior', 'Mid-level', 'Entry-level'. Look at the job title for clues.",
    #     examples=[
    #         "Senior",
    #         "Mid-level",
    #         "Entry-level",
    #         "Not specified",
    #     ],
    # )
    experience_level: Literal["Senior", "Mid-level", "Entry-level", "Not specified"] = (
        Field(
            description="State career level for this role. Look at the job title for clues.",
        )
    )

    involves_ai_model: InvolvesAIModel
    remote_model: RemoteModel
