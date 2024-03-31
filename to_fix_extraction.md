# To Fix in extraction process

- it allows strings for 'Not specified' when it should be a number or a list
  - fix "experience_years": '1+' to 1

- index[9010] remote_model is not a logical answer, must be clearer with gpt3.5. Says possible 'remote', when it should be 'hybrid'

- index[10348] salary might be extracted like this: [115000, 135300, 165600, 194700] instead of a min and max. How to deal with such cases?
  - These are salaries from different regions, within the same listing text. No solution for this yet.

- python tag_extraction/extract_data_single_call.py --index 8533 --model gpt3.5 --max_retries 2

- index[10587] 'city': 'sioux falls, sd, scottsdale, az, troy, mi, franklin, tn, dallas, tx', 'country': 'united states, united states, united states, united states, united states'.  

- city values can be 'remote', they should be 'Not specified' instead of 'remote'.
  - Will need to be filtered out in the dataset.

- if 'junior' is in the job_tile, then 'experience_level' must be 'entry_level' and not None.

- if 'salary_numerical' is not None, then 'salary_currency' can be 'USD'

- many fields are 'Not specified' with 3.5 as it not able to do inferences from the text. 3.5 needs explicit statements. While gpt4 can do it.
