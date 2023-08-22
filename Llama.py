import re
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import pandas as pd
from llama_cpp import Llama
import torch


# print('loading model')
#
#
# model_path = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
#
# llm = Llama(model_path=model_path)

def llama2_quant(prompt):
    output = llm(prompt)
    print('output', output['choices'][0]['text'])
    return output['choices'][0]['text']


# model_path = "models_hf/7b-chat"
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(model_path)
#

forward_flow_prompt = "Starting with the word seed_word, name the next word that follows in your mind from the previous " \
                      "word. Please put down only single words, and do not use proper nouns " \
                      "(such as names, brands, etc.). Name 20 words in total and separate the words by comma (,)."
disassociationprompt = "Please name 10 words that are as different from each other as possible, " \
                       "in all meanings and uses of " \
                       "the words. Follow the following rules: " \
                       "Only single words in English, only nouns (e.g., things, objects, concepts), " \
                       "no proper nouns (e.g., no specific people or places) and no specialised vocabulary " \
                       "(e.g., no technical terms)."
dat_prompt = "Name ten words that are as different from each other as possible."

# Prompt to generate creative uses for an item
aut_prompt = 'What are some creative uses for a item? The goal is to come up with creative ideas, ' \
             'which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or' \
             ' different. List num_use_cases creative uses for a item.'


# def llama2_hf(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     generate_ids = model.generate(inputs.input_ids)
#     print(generate_ids)
#     output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     print(output)
#     return output

def dat(prompt, filename_raw, num_responses):
    """
        Generate and save responses for a given DAT prompt.

        This function generates responses using the specified prompt and saves them to a file.

        Parameters:
            prompt (str): The prompt for generating the responses.
            filename_raw (str): The name of the file to save the raw responses.
            num_responses (int): The number of responses to generate and save.

        Returns:
            None
    """
    # df = pd.DataFrame(columns = ['word ' + str(i) for i in range(1, 11)])
    for i in range(num_responses):
        # response = llama2_hf(prompt)
        response = llama2_quant(prompt)
        # response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
        response = response.split('\n')
        with open(filename_raw, 'a+') as f:
            print(response)
            responses_filtered = []
            for word in response:
                if word.startswith("1.") or word.startswith("2.") or word.startswith("3.") \
                        or word.startswith("4.") or word.startswith("5.") or word.startswith("6.") \
                        or word.startswith("7.") or word.startswith("8.") or word.startswith("9.") \
                        or word.startswith("10."):
                    word = re.sub(r'[\d. ]+', '', word)
                    responses_filtered.append(word)
            string_to_write = ",".join(responses_filtered)
            # Write the string to the file
            f.write('\n' + string_to_write)
            # response = chatgpt(prompt, temperature)


def ff(prompt, filename_raw, num_responses, seedword):
    """
        Generate and save responses for a given FF prompt.

        This function generates responses using the specified prompt and saves them to a file.

        Parameters:
            prompt (str): The prompt for generating the responses.
            filename_raw (str): The name of the file to save the raw responses.
            num_responses (int): The number of responses to generate and save.

        Returns:
            None
    """
    prompt = prompt.replace('seed_word', seedword)
    print('prompt', prompt)
    # df = pd.DataFrame(columns = ['word ' + str(i) for i in range(1, 11)])
    count = 0
    for i in range(num_responses):
        print('COUNT', count)
        # response = llama2_hf(prompt)
        response = llama2_quant(prompt)
        # response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
        response = response.split(',')
        with open(filename_raw, 'a+') as f:
            if len(response) >= 15:
                print('SAVED!!')
                count += 1
                f.write('\n')
                f.write(",".join(response))
            else:
                response = ','.join(response)
                response = response.split('-')
                if len(response) >= 15:
                    count += 1
                    print('-', response)
                    f.write('\n')
                    f.write(",".join(response))
        if count >= 100:
            return None
            # response = chatgpt(prompt, temperature)


def save_response_ff2(filename, responses, seedword):
    """
        Save responses to a file in CSV format.

        This function saves the responses to a CSV file with each response in a separate row.

        Parameters:
            filename (str): The name of the file to save the responses.
            responses (list): A list containing the responses to be saved.
            seedword (str): The seedword used for the responses.

        Returns:
            None
    """
    max_length = max(len(lst) for lst in responses)
    df = pd.DataFrame(columns=['Subject#'] + ['Word ' + str(i) for i in range(1, max_length + 1)])
    # print(df, max_length)
    for counter, i in enumerate(responses):
        if len(i) < max_length:
            for j in range(max_length - len(i)):
                i.append('')
            print(i, 'new')
        df.loc[len(df)] = [counter] + i
    # print(df)
    df.to_csv(filename + '.csv')  #


def save_response_aut(prompt, item, filename, num_use_cases, num_responses):
    """
        Generate and save creative responses for a given item.

        Parameters:
            prompt (str): The base prompt for generating creative responses.
            item (str): The item for which creative responses are generated.
            filename (str): The base filename for saving the responses to a CSV file.
            num_use_cases (int): The number of creative use cases to include in the prompt.
            num_responses (int): The number of responses to generate and save.

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
        response = llama2_quant(prompt)
        # response = a21_model(prompt)
        # print(response)
        # print(response)
        # response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
        # print(response)
        response = response.split('\n')
        print(response)
        for j in response:
            if j not in ['', ' ']:
                df.loc[len(df)] = [counter, item, j, 'creative']
                counter += 1
                df.to_csv(filename + '_' + item + '.csv', sep='#')


def save_response_rat(filename):
    """
       Save responses generated by the RAT (Remote Associates Test) prompt.

       This function generates responses to RAT prompts and saves them to a file.

       Parameters:
           filename (str): The name of the file to save the responses.

       Returns:
           None
       """
    dfsol = pd.read_csv('data_changed_nan.txt', sep=' ', index_col=False)
    # df2 = pd.read_csv('data_changed_nan.txt', sep=' ')['Solutions']
    # print(df.to_string())
    # print(df2.to_string())
    try:
        with open(filename, 'r') as file:
            lines = len(file.readlines())
            print('curr num of lines', lines)
    except FileNotFoundError:
        lines = 0
    # Get the next row using the index
    # print('ri', lines, dfsol.iloc[lines+1]['RemoteAssociateItems'])
    if lines < len(dfsol):
        pass
    else:
        print('None')
        return None
    # print('i', i, 'lines', lines)
    curr_row = dfsol.iloc[lines]
    curr_words = curr_row['RemoteAssociateItems']
    prompt = 'What word connects curr_items? Only name the connecting word and do not explain your answer.'
    prompt = prompt.replace('curr_items', curr_words)
    print('prompt', prompt)
    response = llama2_quant(prompt)
    with open(filename, 'a+') as f:
        if lines != 0:
            f.write('\n' + prompt + ' # ' + response.replace('\n', ' '))
        else:
            f.write(prompt + ' # ' + response.replace('\n', ' '))
    save_response_rat(filename)


def evaluate_multiple_respones(filename_responses, filename_data, filename_eval):
    """
        Evaluate the accuracy of RAT responses against the solutions.

        This function reads RAT responses and their corresponding solutions from files, and evaluates the accuracy
        of the responses against the solutions. The evaluation result is saved to a new file.

        Parameters:
            filename_responses (str): The name of the file containing the RAT responses.
            filename_data (str): The name of the file containing the RAT data with solutions.
            filename_eval (str): The name of the file to save the evaluation results.

        Returns:
            None
    """
    df1 = pd.read_csv(filename_data, sep=' ')['Solutions']
    df2 = pd.read_csv(filename_responses, sep='#', names=['prompt', 'responses'], encoding='windows-1252',
                      encoding_errors='ignore')['responses']
    df4 = pd.read_csv(filename_data, sep=' ')['RemoteAssociateItems']
    # print(df2, df1)
    # print(df2, df1)
    right_false = []
    counter_right = 0
    for i, word in enumerate(df2):
        one_is_right = False
        curr_sol = df1[i].lower()
        if len(word.split(' ')) > 1:
            for j in word.split(' '):
                j = re.sub(r'[^a-zA-Z]', '', j)
                if j.lower() == curr_sol:
                    print(j.lower(), curr_sol, i)
                    one_is_right = True
            right_false.append(one_is_right)
        else:
            curr_response = word.lower()
            if curr_sol == curr_response:
                # print(curr_sol, curr_response)
                print(curr_response, curr_sol)
                right_false.append(True)
            else:
                if curr_sol.startswith(curr_response):
                    counter_right += 1
                    right_false.append(True)
                else:
                    right_false.append(False)
                # print(curr_sol, curr_response)
            # print(i)
            # print(curr_sol, curr_response)
            # pass
            # print(curr_sol, curr_response)
    df3 = pd.DataFrame()
    df3['response_filtered'] = df2.replace(',', ' ')
    df3['solution'] = df1
    df3['eval'] = right_false
    df3['RATItems'] = df4
    df3['prompt'] = pd.read_csv(filename_responses, sep='#', names=['prompt', 'response'], encoding='windows-1252',
                                encoding_errors='ignore')['prompt']
    df3['response_raw'] = df3['prompt'] = \
        pd.read_csv(filename_responses, sep='#', names=['prompt', 'response'], encoding='windows-1252',
                    encoding_errors='ignore')['response']
    # print(df3)
    df3.to_csv(filename_eval, sep='#')
    # print(i, counter_right)
    print(df3['eval'].sum())


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
    df = pd.DataFrame(columns=columns)
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
                    df.loc[len(df)] = [counter] + line.split(',')[0:11 + diff]
    print(df)
    df.to_csv(filename.replace('.txt', '.tsv'), sep='\t', index=False)

# example runs
# DAT
# dat(disassociationprompt, 'test_dat', 10)
# dat_txt_tsv('test_dat')
# FF
# ff(forward_flow_prompt, 'ff_test', 5, 'bear')
# with open('ff_test','r') as f:
#     listl=[]
#     for line in f:
#         strip_lines=line.strip()
#         listli=strip_lines.split(',')
#         listli = [l.strip() for l in listli]
#         listl.append(listli)
#     print(listl)
# save_response_ff2('ff_test2_', listl, 'bear')
# AUT
# save_response_aut(aut_prompt, 'book', 'aut_test', 10, 3)

# RAT
# save_response_rat('rat_test')
