import os
import OpenAI


SYSTEM_PROMPT = """You will be given text for a full job posting.
Answer only by quoting from the text and do not use our memory and do not invent anything.

You will need to extract or summarise unstructured data from the given text for several fields of information. 

1. Summarise the job description in Four sentences- with a focus on how the company and role uses AI and which skills and tools are wanted or needed.  (DO NOT mention location and salary info here) 
2. If there is salary info in the  text, output this with just the currency and the salary numerical amount (no extra info about the salary and benefits).
3. If there is info about the job type in the text (e.g. part-time, remote, full-time, contractor, intern among others) output this . Some times “full-time” can be extrapolated by references to for example “annual salary”. 
4. If the text specifies years of experience required; output this with no extra info in the format;
5. What specific skills, tools, software libraries, languages and frameworks are required for this job. 
6. What specific skills, tools, software libraries, languages and frameworks are mentioned as useful or good to have but not required for the job. 
7. If there is an about us or about the company section - summarise the company in three to four sentences, with a focus on how the company is using AI.  
8. Does this job involve working with AI, machine learning or data science? Base this answer on the skills mentioned and your knowledge of AI and data. 
9,10,11,12. If there is location info in the text, output this (there may be missing info here, in which case leave it blank unless you are certain of an extrapolation) valid extrapolations can be e.g. a) if california state is mentioned but no country mentioned you know it is in USA, if b) a city is called e.g. San Francisco and Country is called US, you know the state is California, Remember the difference between cities, states, countries and continents. If multiple locations are listed for the jobs, include them in the same output with a ":" separator;

Output your answers in the format;
1. Description; xyz
2. Salary amount or range; xyz (currency symbol and number only)
3. Job Type; xyz (all answers must be just one word or a compound word). 
4. Experience needed; xyz years
5. Skills required; xyz:abc (all that are mentioned, : separated). (Outputs in the list must be specific words or compound words here, no adjectives or background. )
6. Skils useful; xyz:abc (all that are mentioned, : separated) (Outputs in the list must be specific words or compound words here, no adjectives or background. )
7. Company info; xyz. 
8. AI relevant; (Yes or no)
9. City or cities; xyz 
10. Country or countries; xyz (just country name in full)
11. Regions/States (write the name in full)
12.  Remote; (Yes or no)
If any field is unavailable, output the number e.g. 4., but leave the entry blank. 

Note - DO NOT, use "and" in your ":" separated list responses, and do not combine things with e.g. "/", separate everything!
Here is an example full  text;
------------------------------------------------------
Research Engineer Apply locations Poland, Remote Poland, Warsaw time type Full time posted on Posted Today job requisition id JR1971657 NVIDIA's technology is at the heart of the AI revolution, touching people across the planet by powering everything from self-driving cars, robotics, and voice-powered intelligent assistants. Academic and commercial groups around the world are using NVIDIA GPUs to revolutionize deep learning and data science, and to power data centers.

We are looking for a Research Engineer to join the Algorithmic ML Optimization team that is developing new efficient ML architectures. In this role you will interact with the scientific community creating algorithms to accelerate the training and inference of ML models for Conversational AI.

What you’ll be doing:
Create new efficient ML architectures for Conversational AI (LLM, NLP, ASR, TTS, Multimodal/Vision)
Develop model optimization algorithms based on Neural Architecture Search, Pruning, Knowledge Distillation, etc.
Work in a dynamic, applied team of engineers and researchers
Work on large-scale multi-node ML models

What we need to see:

Master’s degree in Computer Science, Artificial Intelligence, Applied Math, or related field
Machine learning fundamentals (linear algebra, probability theory, optimization, supervised/unsupervised/self-supervised ML, etc.)
Hands-on experience with Deep Learning (Convolutional Neural Networks, Transformers, etc.)
Experience in Conversational AI technologies (LLM, NLP, ASR, TTS, Multimodal/Vision) or a related field.
Programming skills (Python, C/C++), algorithms & data structures, debugging, performance analysis, and design skills.
Experience with deep learning frameworks such as PyTorch or TensorFlow
Ability to work independently and handle your own work effort.
Good communication and documentation habits.
Ways to stand out from the crowd:
Good track of publications in leading international conferences/journals
Experience with ML model optimization techniques such as Neural Architecture Search, Pruning, Distillation, Quantization, etc.
Knowledge of CPU and/or GPU architectures in the context of ML algorithms.
Ph. D. degree or equivalent experience.

NVIDIA is widely considered to be one of the technology world’s most desirable employers. We have some of the most forward-thinking and hardworking people in the world working for us. If you're creative and autonomous, we want to hear from you!

The base salary range for this full-time position is $152,800 - $183,360.  Compensation packages include base salary, equity, and benefits. The range displayed on each job posting reflects the minimum and maximum target for new hire salaries for the position, determined by work location and additional factors, including job-related skills, experience, interview performance, and relevant education or training. 

------------------------------------------------------
Here is an example of a good output
------------------------------------------------------

1. Description; NVIDIA seeks a Research Engineer for their Algorithmic ML Optimization team, tasked with developing new efficient ML architectures and optimization algorithms to accelerate training and inference of ML models, specifically for Conversational AI. The role requires involvement in creating algorithms, engaging with the scientific community, and working on large-scale multi-node ML models. Required expertise includes machine learning fundamentals, deep learning, and programming skills, ideally with Conversational AI (LLM, NLP, ASR, TTS, Multimodal/Vision). It involves a balance of independent work and collaboration, with a focus on performance analysis and documentation.
2. $152,800 - $183,360
3. Job Type; Full-time
4. 
5. Skills required; Machine Learning:Linear Algebra:Probability Theory:Optimization:Supervised Learning:Unsupervised Learning:Self-Supervised Learning:Deep Learning:Convolutional Neural Networks:Transformers:Conversational AI:LLM:NLP:ASR:TTS:Multimodal:Vision:Python:C/C++:PyTorch:TensorFlow
6. Skills useful; Neural Architecture Search:Pruning:Knowledge Distillation:Quantization:CPU architecture:GPU architecture
7. Company info; NVIDIA stands at the forefront of the AI revolution, impacting lives worldwide through its GPU technology which powers a wide array of applications, from self-driving cars and robotics to voice assistants. Recognized globally, NVIDIA's GPUs are instrumental for deep learning, data science, and powering data centers, while the company cultivates an image as one of the most attractive employers in the technology sector.
8. AI relevant; Yes
9. City or cities; Warsaw
10. Country or countries; Poland
11. Regions/States;
12. Remote; Yes

------------------------------------------------------
Now here is the text to use: 
------------------------------------------------------
"""


def extract_tags_with_prompt(cleaned_text: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": cleaned_text,
                },
            ],
        )
        return response
    except Exception as e:
        print(e)
        return "Error generating post with the OpenAI API."
