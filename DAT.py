import random
import re
import time

import openai
import pandas as pd

from Access_to_models import chatgpt



disassociationprompt = "Please name 10 words that are as different from each other as possible, " \
                       "in all meanings and uses of " \
                       "the words. Follow the following rules: " \
                       "Only single words in English, only nouns (e.g., things, objects, concepts), " \
                       "no proper nouns (e.g., no specific people or places) and no specialised vocabulary " \
                       "(e.g., no technical terms). "

disassociationprompt2 = "Please name 10 words that are as different from each other as possible, " \
                       "in all meanings and uses of " \
                       "the words. Maximize the unrelatedness of the words! Follow the following rules: " \
                       "Only single words in English, only nouns (e.g., things, objects, concepts), " \
                       "no proper nouns (e.g., no specific people or places) and no specialised vocabulary " \
                       "(e.g., no technical terms). "

more_creative = 'Try to be more creative!'


def dat_txt_tsv(filename):
    """
    Convert a comma-separated text file to a tab-separated values (TSV) file.

    This function takes a filename of a comma-separated text file and converts it to a tab-separated values (TSV) file.

    Parameters:
        filename (str): The name of the comma-separated text file.

    Returns:
        None
    """
    columns = ['id']
    for i in range(1, 11):
        columns.append('word' + str(i))
    df = pd.DataFrame(columns = columns)
    print(columns)
    with open(filename, 'r') as f:
        for counter, line in enumerate(f):
            line = line.strip()
            if len(line.split(',')[0:11]) == 10:
                df.loc[len(df)] = [counter] + line.split(',')[0:11]
                print([counter] + line.strip().split(',')[0:11])
            else:
                diff = 10 - len(line.split(',')[0:11])
                if diff > 0:
                    curr_row = line.split(',')[0:11]
                    for i in range(diff):
                        curr_row.append('')
                    print([counter] + curr_row, 'diff')
                    df.loc[len(df)] = [counter] + curr_row
                else:
                    df.loc[len(df)] = [counter] + line.split(',')[0:11+diff]
    print(df)
    df.to_csv(filename + '.tsv', sep='\t', index=False)

def dat(prompt, filename_raw, num_responses, temperature):
    """
        Generate and save responses for a given DAT prompt.

        This function generates responses using the specified prompt and saves them to a file.

        Parameters:
            prompt (str): The prompt for generating the responses.
            filename_raw (str): The name of the file to save the raw responses.
            num_responses (int): The number of responses to generate and save.
            temperature (float): The temperature for response generation (higher values make output more random).

        Returns:
            None
    """
    # df = pd.DataFrame(columns = ['word ' + str(i) for i in range(1, 11)])
    try:
        with open(filename_raw, 'r') as file:
            lines = len(file.readlines())
            print('curr num of lines', lines)
    except FileNotFoundError:
        lines = 0
    if lines < num_responses:
        # response = a21_model(prompt)
        try:
            response = chatgpt(prompt, temperature)
            # response = chatgpt2(prompt, top_p)
            print(response)
            response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
            response = response.split('\n')
            with open(filename_raw, 'a+') as f:
                string_to_write = ",".join(response)
                # Write the string to the file
                if lines == 0:
                    f.write(string_to_write)
                else:
                    f.write('\n' + string_to_write)
            dat(prompt, filename_raw, num_responses, temperature)
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(20)
            dat(prompt, filename_raw, num_responses, temperature)
            # response = chatgpt(prompt, temperature)
    else:
        return None
    # dat_txt_tsv(filename_raw)



def save_multiple_response_dat(prompts, filename, filename2,  num_responses):
    """
       Generate and save multiple responses for given DAT prompts.

       This function generates responses using the specified prompts and saves them to separate files.

       Parameters:
           prompts (list): A list of prompts for generating the responses.
           filename (str): The name of the file to save the responses from the first prompt.
           filename2 (str): The name of the file to save the responses from the second prompt.
           num_responses (int): The number of responses to generate and save.

       Returns:
           None
    """
    # df = pd.DataFrame(columns = ['word ' + str(i) for i in range(1, 11)])
    for i in range(num_responses):
        raw_response = chatgpt(prompts[0], [])
        response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), raw_response))
        response = response.split('\n')
        with open(filename, 'a+') as f:
            print(i)
            print(response)
            # Convert the list of strings to a single string separated by commas
            string_to_write = ",".join(response)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)
        response2 = chatgpt(prompts[1], message_history=[{'role': 'user', 'content': prompts[0]}, {"role": "assistant", "content": raw_response}])
        response2 = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response2))
        response2 = response2.split('\n')
        with open(filename2, 'a+') as f:
            print(i)
            print(response2)
            # Convert the list of strings to a single string separated by commas
            string_to_write = ",".join(response2)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)


def filter_responses(filename):
    """
        Filter and modify responses in the given file.

        This function reads responses from the specified file, filters out certain unwanted strings,
        and writes the modified responses back to the file.

        Parameters:
            filename (str): The name of the file containing the responses.

        Returns:
            None
    """
    # Open the file in read mode
    lines_lst = []
    with open(filename, 'r') as file:
        for line in file:
            new_list = []
            curr_list = line.strip().split(',')
            # print(curr_list)
            for i in curr_list:
                if not (i.startswith('Certainly') or i.startswith('Sure') or i.endswith(':') or i.endswith('!') or i.startswith('here') or i in ['', ',', 'Okay']):
                    new_list.append(i)
                    # print(i)
                else:
                    pass
                    # print(i)
            # print(new_list)
            lines_lst.append(new_list)
    file.close()
    # Open the file in write mode and write the modified contents back to the file
    with open(filename, 'w') as f:
        for i, response in enumerate(lines_lst):
            print(response)
            # Convert the list of strings to a single string separated by commas
            string_to_write = ",".join(response)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)
    f.close()

# example run
# dat(disassociationprompt, 'dat_test', 100, '')
# dat_txt_tsv('dat_test')
