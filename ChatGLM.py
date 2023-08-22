import re
from transformers import AutoTokenizer, AutoModel
import pandas as pd

forward_flow_prompt = "Starting with the word seed_word, name the next word that follows in your mind from the previous " \
                      "word. Please put down only single words, and do not use proper nouns " \
                      "(such as names, brands, etc.). Name 19 words in total and separate the words by comma (,)."
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
items = ['book', 'fork', 'tin can', 'box', 'rope', 'brick']
aut_prompt = 'What are some creative uses for a item? The goal is to come up with creative ideas, ' \
             'which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or' \
             ' different. List num_use_cases creative uses for a item.'


def chatglm(user_input, message_history=None):
    """
      Generates a response from a chatbot model.

      Args:
          user_input (str): The user's input message.
          message_history (list, optional): List of previous messages. Defaults to None.

      Returns:
          str: The model's response.
      """
    if message_history is None:
        message_history = []
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, user_input, history=message_history)
    print(user_input, response)
    return response


def dat_txt_tsv(filename, filename2):
    """
       Converts a txt data file to a TSV format.

       Args:
           filename (str): The name of the input CSV file.
           filename2 (str): The name of the output TSV file.
       """
    columns = ['id']
    for i in range(1, 11):
        columns.append('word' + str(i))
    df = pd.DataFrame(columns=columns)
    print(columns)
    with open(filename, 'r', encoding='utf8') as f:
        for counter, line in enumerate(f):
            print([counter] + line.strip().split(',')[1:11])
            line = line.strip()
            df.loc[len(df)] = [counter] + line.split(',')[1:11]
    print(df)
    df.to_csv(filename2, sep='\t', encoding='utf8')


def dat(prompt, filename_raw, filename2, num_responses):
    """
      Generates responses to a prompt and saves them to a file.

      Args:
          prompt (str): The prompt for which responses are generated.
          filename_raw (str): The name of the raw response data file (CSV).
          filename2 (str): The name of the processed response data file (TSV).
          num_responses (int): The number of responses to generate.
      """
    for i in range(num_responses):
        response = chatglm(prompt)
        response = ''.join(filter(lambda x: (not x.isdigit() and x not in ['.', ' ']), response))
        response = response.split('\n')
        with open(filename_raw, 'a+', encoding='utf8') as f:
            print(i)
            string_to_write = ",".join(response)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)
    dat_txt_tsv(filename_raw, filename2)


def filter_responses(filename, filename_new):
    """
        Filters and cleans responses in a text file and saves the cleaned data to a new file.

        Args:
            filename (str): The name of the input response data file.
            filename_new (str): The name of the output cleaned response data file.
    """
    # Open the file in read mode
    lines_lst = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            new_list = []
            curr_list = line.strip().split(',')
            # print(curr_list)
            for i in curr_list:
                i = re.sub('[\u4e00-\u9fff]+', '', i)
                i = re.sub('-[^ ]*', '', i)
                # i = i.replace('(', '')
                # i = i.replace(')', '')
                i = re.sub('[^a-zA-Z]+', '', i)
                # print(filtered_string)
                if not (i.startswith('These') or i.startswith('Note') or i.endswith(':') or i.endswith(
                        '!') or i.startswith('here') or i in ['', ',', 'Okay'] or i.startswith('suchas')):
                    new_list.append(i)
                    # print(i)
                else:
                    pass
                    # print(i)
            # print(new_list)
            lines_lst.append(new_list)
    file.close()
    # Open the file in write mode and write the modified contents back to the file
    with open(filename_new, 'w', encoding='utf8') as f:
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


def save_response_rat(filename):
    """
        Generates and saves responses for the Remote Associates Test (RAT).

        Args:
            filename (str): The name of the output response data file.
    """
    df = pd.read_csv('data_changed_2.txt', sep=' ')['RemoteAssociateItems']
    for i, word in enumerate(df):
        prompt = 'What word connects curr_items? Only name the connecting word and do not explain your answer.'
        prompt = prompt.replace('curr_items', word)
        print(prompt)
        response = chatglm(prompt, [])
        # response = a21_model(prompt)
        # print(response)
        with open(filename, 'a+', encoding='utf8') as f:
            if i != 0:
                f.write('\n' + prompt + ' # ' + response.replace('\n', ' '))
            else:
                f.write(prompt + ' # ' + response.replace('\n', ' '))


def save_response_aut(prompt, item, filename, num_use_cases, num_responses):
    """
       Generates and saves creative uses for an item.

       Args:
           prompt (str): The prompt for generating creative uses.
           item (str): The item for which creative uses are generated.
           filename (str): The name of the output response data file.
           num_use_cases (int): The number of creative use cases to generate.
           num_responses (int): The number of responses to generate for each use case.
    """
    prompt = prompt.replace('item', item)
    prompt = prompt.replace('num_use_cases', str(num_use_cases))
    print(prompt)
    columns = ['id', 'item', 'response', 'condition']
    df = pd.DataFrame(columns=columns)
    counter = 1
    for i in range(num_responses):
        print(i)
        response = chatglm(prompt, [])
        response = response.split('\n')
        # print(response)
        for j in response:
            if j not in ['', ' ']:
                df.loc[len(df)] = [counter, item, j, 'creative']
                counter += 1
                df.to_csv(filename + '_' + item + '.csv')


def save_response_ff2(filename, responses, seedword):
    """
       Saves responses for forward flow prompts to a CSV file.

       Args:
           filename (str): The name of the output response data file.
           responses (list): List of responses for forward flow prompts.
           seedword (str): The seed word used in the prompts.
       """
    max_length = max(len(lst) for lst in responses)
    df = pd.DataFrame(columns=['Subject#'] + ['Word ' + str(i) for i in range(1, max_length + 2)])
    # print(df, max_length)
    for counter, i in enumerate(responses):
        if len(i) < max_length:
            for j in range(max_length - len(i)):
                i.append('')
            print(i, 'new')
        df.loc[len(df)] = [counter, seedword] + i
    # print(df)
    df.to_csv(filename + '.csv')


def save_response_ff(prompt, seedword, filename, num_responses):
    """
    Generates and saves responses for forward flow prompts.

    Args:
        prompt (str): The forward flow prompt.
        seedword (str): The seed word used in the prompt.
        filename (str): The name of the output response data file.
        num_responses (int): The number of responses to generate.
    """
    prompt = prompt.replace('seed_word', seedword)
    print(prompt)
    # print(df)
    responses = []
    for i in range(num_responses):
        response = chatglm(prompt, [])
        print(i)
        response = ''.join(filter(lambda x: (not x.isdigit() and x not in ['.', ' ']), response))
        # print(response)
        response = response.split(',')
        # response = response.split('\n')
        if len(response) == 1:
            response = "".join(response)
            response = response.split('\n')
            if len(response) == 1:
                response = "".join(response)
                response = response.split('->')
                if len(response) == 1:
                    response = "".join(response)
                    response = response.split('-')
        # print([i] + response)
        # df.loc[len(df)] = [i, seedwords[0]] + response
        # print(df)
        responses.append(response)
        save_response_ff2(filename, responses, seedword)

# AUT
# save_response_aut(aut_prompt, 'book', 'aut_test', 10, 25)

# DAT
# dat(disassociationprompt, 'dat_test', 'dat_test2', 100)
# filter_responses('dat_test', 'dat_test_filtered')

# FF
# save_response_ff(forward_flow_prompt, 'bear', 'ff_test', 100)

# RAT
# save_response_rat('rat_test')
