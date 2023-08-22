import time

import ai21
import matplotlib.pyplot as plt
import openai
import pandas as pd
from Access_to_models import chatgpt
from Access_to_models import chatgpt_topp


# Prompt to generate creative uses for an item
aut_prompt = 'What are some creative uses for a item? The goal is to come up with creative ideas, ' \
             'which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or' \
             ' different. List num_use_cases creative uses for a item.'


def save_response_aut(prompt, item, filename, num_use_cases, num_responses, temperature):
    """
        Generate and save creative responses for a given item.

        Parameters:
            prompt (str): The base prompt for generating creative responses.
            item (str): The item for which creative responses are generated.
            filename (str): The base filename for saving the responses to a CSV file.
            num_use_cases (int): The number of creative use cases to include in the prompt.
            num_responses (int): The number of responses to generate and save.
            temperature (float): The temperature for the response generation (higher values make output more random).

        Returns:
            None
    """
    prompt = prompt.replace('item', item)
    prompt = prompt.replace('num_use_cases', str(num_use_cases))
    print(prompt)
    columns = ['id', 'item', 'response', 'condition']
    df = pd.DataFrame(columns=columns)
    counter = 1
    for i in range(num_responses):
        print(i)
        try:
            response = chatgpt(prompt, temperature)
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(20)
            response = chatgpt(prompt, temperature)
        response = response.split('\n')
        for j in response:
            if j not in ['', ' ']:
                df.loc[len(df)] = [counter, item, j, 'creative']
                counter += 1
                df.to_csv(filename + '_' + item + '.csv')


items = ['book', 'fork', 'tin can', 'box', 'rope', 'brick']


# save_response_aut(aut_prompt, items[0], 'aut_gpt_test', 10, 25, '') # example run

