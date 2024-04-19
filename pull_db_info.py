import os
import logging
import subprocess

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set paths to local file
DATASET_LOCAL_PATH = "local_db.json"

stub = modal.Stub(name="create_local_db")

image = (
    modal.Image.debian_slim(python_version="3.11.8", force_build=False)
    .apt_install("git")
    .pip_install(
        "-U",
        "git+https://github.com/towardsai/buster@async_client",
        "openai",
        "deeplake",
        "pandas",
        "tiktoken",
        "pymysql",
        "sqlalchemy",
    )
    .run_commands(
        [
            "git config --global user.name 'Modal Cron Job'",
            "git config --global user.email 'omarsolano27@gmail.com'",
        ]
    )
)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        logger.error(f"Error calculating number of tokens: {e}")
        return 0


def load_and_clean_data():
    """Load data from the database and clean it."""
    # Get data from the database
    df = get_data_from_db()

    # Convert datetime to string, else its not json serializable
    df["created_at"] = df["created_at"].astype(str)

    # Drop rows with missing 'cleaned_description' values
    df.dropna(subset=["cleaned_description"], inplace=True)

    # Rename the 'cleaned_description' column to 'job_listing_text'
    # df.rename(columns={"cleaned_description": "job_listing_text"}, inplace=True)

    # Count the number of tokens in the 'cleaned_description' column, for each row
    df["num_tokens"] = df["cleaned_description"].apply(
        lambda x: num_tokens_from_string(x, "cl100k_base")
    )
    # Drop rows with too many tokens
    df = df[df["num_tokens"] < 8191]
    df.drop("num_tokens", axis=1, inplace=True)

    logger.info(f"DataFrame cleaned. Rows: {len(df)}")
    return df


def get_data_from_db():
    """Load data from MySQL into a DataFrame, using SQLAlchemy."""

    import pandas as pd
    from sqlalchemy import create_engine

    DB_HOST = os.getenv("DB_HOST", "")
    DB_USER = os.getenv("DB_USER", "")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_NAME = os.getenv("DB_NAME", "")

    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(connection_string)

    sql = """
    SELECT 
        j.id as job_id,
        j.created_at as created_at,
        j.title as job_title,
        j.skills as job_skills,
        j.type as job_type,
        j.company_id as company_id,
        j.apply_url as apply_url,
        j.city as city,
        j.country as country,
        j.salary as salary,
        j.salary_min as salary_min,
        j.salary_max as salary_max,
        j.salary_currency as salary_currency,
        CONCAT('https://jobs.towardsai.net/job/', j.slug) as jobs_towardsai_url,
        j.description as role_description,
        c.name as company_name,
        c.description as company_description,
        s.cleaned_description as cleaned_description,
        s.skills_required as scraped_skills_required,
        s.skills_useful as scraped_skills_useful
    FROM jobs j
    JOIN companies c ON c.id = j.company_id  
    JOIN scraped_jobs s ON s.job_id = j.id
    WHERE j.approved = 1 AND j.ai = 1
    """

    try:
        logger.info("Loading data from MySQL...")
        df = pd.read_sql_query(sql, engine)
        logger.info(f"Data loaded successfully. DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        raise


def setup_ssh_for_git(ssh_private_key: str):
    """Setup SSH for Git."""

    if not ssh_private_key:
        logger.info("No SSH private key found. Skipping SSH setup.")
        return
    ssh_key_path = "/tmp/id_rsa"
    with open(ssh_key_path, "w") as file:
        file.write(ssh_private_key)
    os.chmod(ssh_key_path, 0o600)
    os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=no"

    logger.info("SSH setup complete.")


def clone_repo():
    """Clone the repository to the local environment."""

    auth_repo_url = "git@github.com:towardsai/jobs-board-rag-api.git"
    try:
        subprocess.run(["git", "clone", auth_repo_url, "repo_directory"], check=True)
        os.chdir("repo_directory")
        logger.info("Repository cloned successfully.")
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        raise


def run_git_commands():
    """Run git commands to push changes to the repository."""

    repo_url = "git@github.com:towardsai/jobs-board-rag-api.git"
    try:
        subprocess.run(["git", "add", "local_db"])
        subprocess.run(["git", "commit", "-m", "Update DB from cron job"])
        subprocess.run(["git", "push", repo_url, "main"])
        logger.info("Git push successful")
    except Exception as e:
        logger.error(f"Error running git commands: {e}")
        raise


@stub.function(
    image=image,
    secrets=[
        modal.Secret.from_name("mysql-secret-jobs"),
        modal.Secret.from_name("my-custom-secret"),
        modal.Secret.from_name("my-github-secret"),
    ],
    timeout=3000,
)
def create_local_db():

    logger.info("Script started.")
    # setup_ssh_for_git(os.getenv("SSH_PRIVATE_KEY", ""))
    # clone_repo()
    df = load_and_clean_data()
    df.to_json("data/db_info.json", orient="records", lines=False, indent=4)
    logger.info("Dataframe saved to db_info.json")

    # upload_to_vector_store(df, DATASET_LOCAL_PATH)
    # logger.info("DB updated successfully.")

    # run_git_commands()
    # logger.info("Git push executed successfully.")


# During Eastern Daylight Time (EDT): 3:00 p.m. UTC - 4 hours = 11:00 a.m. EDT
# runs at 11:00 a.m. (EDT) every Monday
# @stub.function(schedule=modal.Cron("0 15 * * 1"))
@stub.local_entrypoint()
def scheduled_main():
    create_local_db.local()
