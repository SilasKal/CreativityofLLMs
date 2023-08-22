from sklearn.decomposition import PCA
import os

import pandas as pd
import seaborn as sns
import venn as venn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
import venn
from Access_to_models import get_embedding_gensim_list

def get_colordict(palette,number,start):
    pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
    color_d = dict(enumerate(pal, start=start))
    return color_d

def get_df_dat_gpt(filename):
    df = pd.read_csv(filename, sep=',', header=None)
    columns = ['Word {}'.format(i) for i in range(1, 11)]
    df.columns = columns
    return df
def visualize_word_freq(dataframe1, model, columns=['Word {}'.format(i) for i in range(1, 11)]):
    word_freq_dict = {}
    # Iterate over the columns in dataframe1
    for column in columns:
        # Iterate over the rows in dataframe1
        for word in dataframe1[column]:
            # Check if the word is not NaN (missing value)
            if pd.notnull(word):
                # Increment the word frequency in the dictionary
                word = word.lower()
                word_freq_dict[word] = word_freq_dict.get(word, 0) + 1

    # Create a new DataFrame from the word frequency dictionary
    dataframe2 = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])

    # Sort dataframe2 by frequency in descending order
    dataframe2 = dataframe2.sort_values('Frequency', ascending=False)

    # Reset the index of dataframe2
    dataframe2 = dataframe2.reset_index(drop=True)
    print(dataframe2)
    dataframe2= dataframe2[dataframe2['Word'].str.len() <= 20]
    word_freq_dict = dict(zip(dataframe2['Word'], dataframe2['Frequency']))

    # Create a word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Generate the word cloud
    wordcloud.generate_from_frequencies(word_freq_dict)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # plt.show()
    # create index list for slicing
    index_list = [[i[0], i[-1] + 1] for i in np.array_split(range(100), 5)]

    n = dataframe2['Frequency'].max()
    color_dict = get_colordict('viridis', n, 1)

    fig, axs = plt.subplots(1, 5, figsize=(16, 8), facecolor='white', squeeze=False)
    for col, idx in zip(range(0, 5), index_list):
        df = dataframe2[idx[0]:idx[-1]]
        if not df.empty:
            # df = dataframe2
            label = [w + ': ' + str(n) for w, n in zip(df['Word'], df['Frequency'])]
            color_l = [color_dict.get(i) for i in df['Frequency']]
            x = list(df['Frequency'])
            y = list(range(len(df)))

            sns.barplot(x=x, y=y, data = df, alpha=0.9, orient='h',
                        ax=axs[0][col], palette=color_l)
            axs[0][col].set_xlim(0, n + 1)  # set X axis range max
            axs[0][col].set_yticklabels(label, fontsize=12)
            axs[0][col].spines['bottom'].set_color('white')
            axs[0][col].spines['right'].set_color('white')
            axs[0][col].spines['top'].set_color('white')
            axs[0][col].spines['left'].set_color('white')
    plt.tight_layout()
    plt.savefig('top50words_' + model  + '.png')
    plt.show()
    print('return', dataframe2.iloc[[i for i in range(1, 51)]])
    # return dataframe2.head(3)
    return dataframe2.iloc[[i for i in range(1, 51)]]
    # return dataframe2.head(50)
def visualize_directory_ff(directory, filter):
    for filename in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, filename)):
                    if (filename.startswith(filter)):
                        print(directory + '/' + filename)
                        visualize_word_freq(pd.read_csv(directory + '/' + filename))
def visualize_directory_dat(directory, filter, model):
    for filename in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, filename)):
                    if (filename.startswith(filter)):
                        print(directory + '/' + filename)
                        return visualize_word_freq(get_df_dat_gpt(directory + '/' + filename), model)
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def plot_venn_diagramm(df1, df2, df3, df4,df5, item, labels=['Human Study', 'ChatGLM', 'Jurassic-2-Jumbo-Instruct', 'GPT-3.5-turbo', 'LLama 2' ]):

    # Assuming you have four DataFrames called df1, df2, df3, and df4

    # Get the unique values from each DataFrame
    set1 = set(df1['Word'])
    set2 = set(df2['Word'])
    set3 = set(df3['Word'])
    set4 = set(df4['Word'])
    set5 = set(df5['Word'])
    # create set unions
    data_dict = {}
    for i, n in enumerate(labels):
        if i == 0:
            data_dict[n] = set1
        if i == 1:
            data_dict[n] = set2
        if i == 2:
            data_dict[n] = set3
        if i == 3:
            data_dict[n] = set4
        if i == 4:
            data_dict[n] = set5
    # print(data_dict)
    llm_sets = [set2, set3, set4, set5]
    llm_names = ["ChatGLM", "J2-Jumbo-Instruct", "GPT", "LLama"]
    intersection_result = set1 & (set2 | set3 | set4 | set5)
    print("Intersection human and LLMs :", len(intersection_result), intersection_result)
    intersection_result = set2 &  (set3 | set4 | set5)
    print("Intersection human and LLMs :", len(intersection_result), intersection_result)
    for i in range(len(llm_sets)):
        intersection_result = set1 & llm_sets[i]
        print("Intersection between human :",  llm_names[i], len(intersection_result))
    for i in range(len(llm_sets)):
        for j in range(i + 1, len(llm_sets)):
            similarity = jaccard_similarity(llm_sets[i], llm_sets[j])
            intersection_result = llm_sets[i] & llm_sets[j]
            print(f"Jaccard Similarity between {llm_names[i]} and {llm_names[j]}:", similarity, len(intersection_result))

    jaccard_llm_human = [
        jaccard_similarity(set2, set1),
        jaccard_similarity(set3, set1),
        jaccard_similarity(set4, set1),
        jaccard_similarity(set5, set1)
    ]


    print("Jaccard Similarities between LLMs and Humans:")
    print(jaccard_llm_human)
    # plot the Venn diagram
    venn.venn(data_dict)
    title = 'Overlap FF Responses with the Seed Word ' + item
    plt.title(title)
    # display the plot
    plt.legend(labels, loc='upper left', bbox_to_anchor=(0, 1.1)).set_visible(False)
    plt.savefig(title + '.png')
    plt.show()


def visualize_embeddings(data, threed=True):

    # Define the texts and their corresponding top ten words
    top_words = [i.tolist() for i in data]

    # Initialize an empty list to store the embeddings
    embeddings = []
    # Extract the word embeddings for each top word
    for i, words in enumerate(top_words):
        print('i', i)
        # for vec in get_embedding_gensim_list(words):
        #     embeddings.append(vec)
        embeddings += [vec for vec in get_embedding_gensim_list(words)]
        # for word in words:
        #     embedding = get_embedding_gensim(word)
        #     # embedding = np.random.random((300,))
        #     if embedding is None:
        #         embedding = np.zeros((300,))
        #     embeddings.append(embedding)
    # for words in top_words:
        # embeddings.append([get_random_embedding(word) for word in words])
    #     embeddings.append([get_embedding_gensim(word) for word in words])

    tsne = TSNE(perplexity=30, metric='cosine')
    embeddings = np.array(embeddings)
    flat_embeddings = tsne.fit_transform(embeddings)
    # pca = PCA(n_components=2)
    # print(embeddings)
    # flat_embeddings = pca.fit_transform(embeddings)

    # Create a color map for each text
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    # Plot the embeddings in a scatter plot
    models = ['j2', 'glm', 'gpt', 'human']
    plt.figure(figsize=(20, 8))
    for i, text in enumerate(data):
        start_idx = i * 10
        end_idx = start_idx + 10
        curr_label = f' {models[i]} \n {text.iloc[[i for i in range(0, 9)]]}'
        # ax.scatter(flat_embeddings[start_idx:end_idx, 0], flat_embeddings[start_idx:end_idx, 1], flat_embeddings[start_idx:end_idx, 2], c=[colors[i]], label=curr_label)
        plt.scatter(flat_embeddings[start_idx:end_idx, 0], flat_embeddings[start_idx:end_idx, 1], c=[colors[i]], label=curr_label)

    # Add legend and labels to the plot
    plt.legend()
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.subplots_adjust(right=0.8)  # Adjust the right margin to make space for the legend

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Word Embeddings Scatter Plot')
    plt.show()
    if threed:
        pca = PCA(n_components=3)
        print(embeddings)
        flat_embeddings = pca.fit_transform(embeddings)
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(projection='3d')
        for i, text in enumerate(data):
            start_idx = i * 10
            end_idx = start_idx + 10
            curr_label = f' {models[i]} \n {text.iloc[[i for i in range(0, 9)]]}'
            ax.scatter(flat_embeddings[start_idx:end_idx, 0], flat_embeddings[start_idx:end_idx, 1],
                       flat_embeddings[start_idx:end_idx, 2], c=[colors[i]], label=curr_label)
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.show()




