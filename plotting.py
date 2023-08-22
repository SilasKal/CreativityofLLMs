import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import ttest_ind
import openpyxl

# DAT

def plot_dat(filename_study, filename_gpt, filename_glm, filename_j2, filename_llama2):
    gpt = pd.read_csv(filename_gpt, names=['dat'])['dat']
    study_data = pd.read_csv(filename_study, sep='\t')['dat']
    # study1a = pd.read_csv('study1a.tsv', sep='\t')['dat']
    # study1b = pd.read_csv('study1b.tsv', sep='\t')['dat']
    # study1c1 = pd.read_csv('study1c.tsv', sep='\t')['dat.0']
    # study1c2 = pd.read_csv('study1c.tsv', sep='\t')['dat.2']
    # study2 = pd.read_csv('study2.tsv', sep='\t')['dat']
    j2_jumbo = pd.read_csv(filename_j2, sep='\t')['dat']
    glm = pd.read_csv(filename_glm,sep='\t', names=['dat'])
    glm = pd.to_numeric(glm['dat'], errors='coerce').dropna()
    llama2 = pd.read_csv(filename_llama2, sep='\t')['dat']
    llama2_all = pd.read_csv(filename_llama2, sep='\t')
    study_mean = study_data.mean()
    glm_mean = glm.mean()
    j2_mean = j2_jumbo.mean()
    gpt_mean = gpt.mean()
    llama2_mean = llama2.mean()

    study_std = study_data.std()
    glm_std = glm.std()
    j2_std = j2_jumbo.std()
    gpt_std = gpt.std()
    llama2_std =llama2.std()
    t_statistic, p_value = ttest_ind(gpt, study_data)
    # Print the results
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)
    if p_value < 0.05:
        print("Result: Significant")
    else:
        print("Result: Not Significant")

    # Printing the means
    print("Study Data Mean:", study_mean)
    print("GLM Mean:", glm_mean)
    print("J2 Jumbo Mean:", j2_mean)
    print("GPT Mean:", gpt_mean)
    print("llama2 Mean:", llama2_mean)
    #
    # # Printing the standard deviations
    print("Study Data Standard Deviation:", study_std)
    print("GLM Standard Deviation:", glm_std)
    print("J2 Jumbo Standard Deviation:", j2_std)
    print("GPT Standard Deviation:", gpt_std)
    print("llama2 std:", llama2_std)

    # # Printing the standard deviations
    study_max = max(study_data)
    glm_max = max(glm)
    j2_max = max(j2_jumbo)
    gpt_max = max(gpt)
    llama2_max = max(llama2)

    print("Study Data Maximum Score:", study_max)
    print("GLM Maximum Score:", glm_max)
    print("J2 Jumbo Maximum Score:", j2_max)
    print("GPT Maximum Score:", gpt_max)
    print("llama2 Maximum Score:", llama2_max)

    max_index = llama2_all['dat'].idxmax()
    # Get the row with the maximum score
    row_with_max_score = llama2_all.loc[max_index]
    # Print the row
    print("Row with Maximum Score in llama2 Data:")
    print(row_with_max_score)
    # Creating the bar plot with error bars representing standard deviation
    data_means = [study_mean, glm_mean, j2_mean, gpt_mean, llama2_mean]
    data_std = [study_std, glm_std, j2_std, gpt_std, llama2_std]
    columns = ['Human Study', 'ChatGLM', 'J2', 'GPT-3.5-turbo', 'LLama 2']

    # Choosing different colors for the bars
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    plt.bar(columns, data_means, yerr=data_std, capsize=5, color=colors)
    plt.ylabel('Mean DAT score')
    # plt.title('Divergent Association Task')
    # plt.savefig('dat_study2_barplot_with_std')
    plt.show()


def plot_dat_temp(filename_1='', filename_2='', filename_3='', filename_4='', filename_5='', filename_6='', filename_7='', filename_8='', filename_9='', temps=''):
    data = []
    filenames = [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6, filename_7, filename_8,
                 filename_9]
    for filename in filenames:
        if filename:
            dat = pd.read_csv(filename, sep='\t')['dat']
            data.append(dat)
    default = pd.read_csv('gpt_dat_responses_temp/default_values.txt', names=['dat'])['dat']
    data.append(default)
    temps.append(3)
    data_means = [d.mean() for d in data]
    data_std = [d.std() for d in data]
    data_max = [max(d) for d in data]
    data_min = [min(d) for d in data]
    print('means', data_means)
    print('std', data_std)
    print('max', data_max)
    print('min', data_min)
    print(temps)
    plt.bar(temps, data_means, yerr=data_std, width=0.1)
    plt.ylabel('Mean DAT score')
    plt.xlabel('Temperature')
    plt.title('Divergent Association Task')
    plt.savefig('dat_temp_barplot_with_std')
    plt.show()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            t_stat, p_value = ttest_ind(data[i], data[j])
            # print(f"T-test between {temps[i]} and {temps[j]}:")
            if p_value < 0.05 and (temps[i] == 3 or temps[j] == 3):
                print("There is a significant difference between the data sets.")
                print(f"T-test between {temps[i]} and {temps[j]}:")
                print(f"t-statistic: {t_stat}")
                print(f"p-value: {p_value}")
                print(data[i].mean(), data[j].mean())
            else:
                pass
                # print("There is no significant difference between the data sets.")
# FF

def plot_ff_temp(filename_1='', filename_2='', filename_3='', filename_4='', filename_5='', filename_6='', filename_7='', filename_8='', filename_9='', temps=''):
    data = []
    filenames = [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6, filename_7, filename_8,
                 filename_9]
    for filename in filenames:
        if filename:
            dat = pd.read_csv(filename)['Flow']
            data.append(dat)
    temps.append(3)
    data.append(pd.read_csv('gpt_ff_responses_temp/summary_default.csv')['Flow'])
    data_means = [d.mean() for d in data]
    data_std = [d.std() for d in data]
    data_max = [max(d) for d in data]
    data_min = [min(d) for d in data]
    print('means', data_means)
    print('std', data_std)
    print('max', data_max)
    print('min', data_min)
    print(temps)
    bar_width = 0.1  # You can adjust this value
    plt.bar(temps, data_means, yerr=data_std, width=bar_width)
    plt.ylabel('Max DAT score')
    plt.xlabel('Temperature')
    plt.title('FF')
    plt.show()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            t_stat, p_value = ttest_ind(data[i], data[j])
            # print(f"T-test between {temps[i]} and {temps[j]}:")
            if p_value < 0.05 and (temps[i] == 3 or temps[j] == 3):
                print("There is a significant difference between the data sets.")
                print(f"T-test between {temps[i]} and {temps[j]}:")
                print(f"t-statistic: {t_stat}")
                print(f"p-value: {p_value}")
                print(data[i].mean(), data[j].mean())
            else:
                pass
                # print("There is no significant difference between the data sets.")

def compare_ff_data(items, filenamegpt, filename_glm, filename_j2, filename_llama2):
    # all_data = pd.DataFrame()
    data = []
    std1 = []
    std2 = []
    std3 = []
    std4 = []
    std5 = []
    for item in items:
        # study2 = pd.read_spss('Study 4 - Actors and MTurkers AmPsy.sav')
        study2 = pd.read_spss('FF_data_humans/Study 2 - Representative Sample of Americans.sav') # seed words = toaster
        # study2.to_excel('study2_ff.xlsx') # Wordlist
        # print(study2['Wordlist'])
        study2_data = study2.loc[study2['Wordlist'].str[2:len(item)+2] == item]
        study2_data = study2_data['ForwardFlow']
        glm_data = pd.read_csv(filename_glm + item + '.csv')['Flow']
        j2_data = pd.read_csv(filename_j2 + item + '.csv')['Flow']
        gpt_data = pd.read_csv(filenamegpt + item + '.csv')['Flow']
        llama2_data = pd.read_csv(filename_llama2 + item + '.csv')['Flow']
        # all_data['ff_glm_' + item] = glm_data
        # all_data['ff_gpt_' + item] = gpt_data
        # all_data['ff_j2_' + item] = j2_data
        # all_data['ff_human_' + item] = study2_data
        # all_data.to_excel('all_data.xlsx')
        std1.append(study2_data.std())
        std2.append(glm_data.std())
        std3.append(j2_data.std())
        std4.append(gpt_data.std())
        std5.append(llama2_data.std())
        data.append([item, study2_data.mean(), glm_data.mean(), j2_data.mean(), gpt_data.mean(), llama2_data.mean()])
        # print('mean', [item, study2_data.mean(), glm_data.mean(), j2_data.mean(), gpt_data.mean(), llama2_data.mean()])
        # print('std', [item, study2_data.std(),glm_data.std(), j2_data.std(), gpt_data.std(), llama2_data.std()])
        max_study2_data = np.max(study2_data)
        max_glm_data = np.max(glm_data)
        max_j2_data = np.max(j2_data)
        max_gpt_data = np.max(gpt_data)
        max_llama2_data = np.max(llama2_data)

        # print(round(max_study2_data, 5))
        # print( round(max_glm_data, 5))
        # print(round(max_j2_data, 5))
        # print(round(max_gpt_data, 5))
        # print(round(max_llama2_data, 5))
        t_statistic, p_value = ttest_ind(gpt_data, study2_data)
        # Print the results
        # print(item)
        # print("T-Statistic:", t_statistic)
        # print("P-Value:", p_value)
        # if p_value < 0.05:
        #     print("Result: Significant")
        # else:
        #     print("Result: Not Significant")
    plt.figure(figsize=(50, 10))
    df3 = pd.DataFrame(data=data, columns = ['item', 'study2', 'glm', 'j2', 'gpt', 'llama2'])
    df3.to_excel('df3.xlsx')
    ax = df3.plot(kind='bar', x='item', y=['study2', 'glm', 'j2', 'gpt', 'llama2'], yerr=[std1, std2, std3, std4, std5],
                  capsize=4, error_kw={'capthick': 0.75, 'elinewidth': 0.75})
    # ax = df3.plot(kind='bar', x='item', y=['study data', 'chatglm', 'j2', 'gpt'], yerr=[std1, std2, std3, std4])
    labels = items
    # ax = df3.plot(kind='bar', x='item', y=['study4_data', 'glm_data', 'j2_data', 'gpt_data'])
    ax.set_xticklabels(labels)
    # ax.legend(loc='upper left', framealpha=1, fontsize=8, bbox_to_anchor=(1, 1))
    # for text in ax.get_legend().get_texts():
    #     text.set_alpha(1)
    ax.legend().set_visible(False)
    # ax = df3.plot(kind='bar', x='item', y=['study4_data', 'glm_data', 'j2_data', 'gpt_data'])
    ax.set_xticklabels(labels)
    plt.tight_layout()
    # plt.title('Forward Flow', pad=20)
    plt.xlabel('Seed Word', labelpad=10)
    plt.ylabel('Mean Forward Flow', labelpad=10)
    plt.subplots_adjust(bottom=0.2, left=0.1, top=0.9)
    # plt.savefig('ff_comparison_item_glm_j2_gpt_study2_llama2')
    plt.show()


# RAT

def sum_up_rat_results():
    df4 = pd.DataFrame()
    df4['sum'] = pd.read_csv('gpt_rat_responses_temp/temp = 0.75/rat_responses_gpt_temp_0.75_eval', sep='#')['eval'].astype(int)
    df4['RATItems'] = pd.read_csv('gpt_rat_responses_temp/temp = 0.75/rat_responses_gpt_temp_0.75_eval', sep='#')['RATItems']
    for j in range(1, 5):
        df4['sum'] += pd.read_csv('gpt_rat_responses_temp/temp = 0.75/rat_responses_gpt_temp_0.75_' + str(j) + '_eval', sep='#')['eval'].astype(int)
        # print(df4)
    df4.to_csv('gpt_rat_responses_temp/temp = 0.75/evaluation_gpt_temp0.75_5_runs', sep='#')

def evaluation_analyze_rat():
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)
    df_glm = pd.read_csv('glm_rat_responses/evaluation_glm_10_runs', sep='#')
    df_j2 = pd.read_csv('j2_rat_responses/evaluation_j2_10_runs', sep='#')
    df_gpt = pd.read_csv('gpt_rat_responses/evaluation_gpt_10_runs', sep='#')
    df_llama2 = pd.read_csv('llama2_rat_responses/evaluation_llama2_10_runs', sep='#')
    # df_gpt_temp = pd.read_csv('gpt_rat_responses_temp/temp = 0/evaluation_gpt_temp0_5_runs', sep='#')
    # df_glm = pd.read_csv('gpt_rat_responses_temp/temp = 0/evaluation_gpt_temp0_5_runs', sep='#')
    # df_j2 = pd.read_csv('gpt_rat_responses_temp/temp = 0.25/evaluation_gpt_temp0.25_5_runs', sep='#')
    # df_gpt = pd.read_csv('gpt_rat_responses_temp/temp = 0.5/evaluation_gpt_temp0.5_5_runs', sep='#')
    # df_llama2 = pd.read_csv('gpt_rat_responses_temp/temp = 0.75/evaluation_gpt_temp0.75_5_runs', sep='#')
    # df_gpt_temp = pd.read_csv('gpt_rat_responses_temp/temp = 1/evaluation_gpt_temp1_5_runs', sep='#')
    # df_gpt_temp['sum'] = (df_gpt_temp['sum']/5) *100
    df_llama2['sum'] *=10
    df_gpt['sum'] *= 10
    df_glm['sum'] *= 10
    df_j2['sum'] *= 10
    # df_llama2['sum'] = (df_llama2['sum']/5) *100
    # df_gpt['sum'] = (df_gpt['sum']/5) *100
    # df_glm['sum'] = (df_glm['sum']/5) *100
    # df_j2['sum'] = (df_j2['sum']/5) *100
    df = pd.read_csv('RAT_data_humans/data_rat.txt', sep=' ')
    # print(df.to_string())
    solving2 = df['Solving2sec']
    solving7 = df['Solving7sec']
    solving15 = df['Solving15sec']
    solving30 = df['Solving30sec']
    counts = []
    std1 = []
    std2 = []
    std3 = []
    std4 = []
    std5 = []
    for i in [20, 40, 60, 80, 100]:
        curr_mean_over = df[(df['Solving30sec'] < i) & (df['Solving30sec'] >= i - 20)]
        # print('<', i, '>=', i-20)
        # curr_mean_over['Solving30sec'].to_csv('filter_human' + str(i) + '.csv')
        # df_gpt_temp_filter = df_gpt_temp[(df_gpt_temp['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_gpt_filter = df_gpt[(df_gpt['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_j2_filter = df_j2[(df_j2['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_glm_filter = df_glm[(df_glm['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_llama2_filter = df_llama2[(df_llama2['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_gpt_filter.to_csv('filter_gpt' + str(i) + '.csv')
        std1.append(curr_mean_over['Solving30sec'].std())
        std2.append(df_glm_filter['sum'].std())
        std3.append(df_j2_filter['sum'].std())
        std4.append(df_gpt_filter['sum'].std())
        std5.append(df_llama2_filter['sum'].std())
        counts.append([i, curr_mean_over['Solving30sec'].mean(), df_glm_filter['sum'].mean(),
                       df_j2_filter['sum'].mean(), df_gpt_filter['sum'].mean(), df_llama2_filter['sum'].mean()]) # mean
        # counts.append([i, curr_mean_over['Solving30sec'].count(), len(df_glm_filter[df_glm_filter['sum'] >= 1]),
                       # len(df_j2_filter[df_j2_filter['sum'] >= 1]), len(df_gpt_filter[df_gpt_filter['sum'] >= 1]),
                       # len(df_llama2_filter[df_llama2_filter['sum']>=1]), len(df_gpt_temp_filter[df_gpt_temp_filter['sum']>=1])]) # count
        # print(curr_mean_over['Solving30sec'].count())
    # print(counts)
        # counts.append([i,  curr_mean_over['Solving30sec'].mean(),(len(set(indexes))/curr_mean_over['Solving30sec'].count())*100])
    df3 = pd.DataFrame(data=counts, columns=['percentage', 'human study', 'glm',
                                             'j2', 'gpt', 'llama2'])
    # df3 = pd.DataFrame(data=counts, columns=['percentage', 'human', '0', '0.25',
    #                                          '0.5', '0.75', '1'])
    print(df3)
    # ax = df3.plot(kind='bar', x='percentage', y=[ 'human study', 'glm',
                                             # 'j2', 'gpt', 'llama2'], yerr=[std1, std2, std3, std4, std5])
    ax = df3.plot(kind='bar', x='percentage', y=['human study', 'glm',
                                                 'j2', 'gpt', 'llama2'])
    # ax = df3.plot(kind='bar', x='percentage', y=['human', '0', '0.25',
    #                                         '0.5', '0.75', '1'])
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    ax.set_xticklabels(labels)

    # plt.xlabel('Percentage of humans that solved these items')
    # plt.ylabel('Mean')
    ax.legend().set_visible(False)
    plt.xlabel('Percentage of Humans that Solved These Items')
    plt.ylabel('Mean Success Percentage')
    plt.subplots_adjust(bottom=0.25)
    # plt.savefig('rat_comparison_human_glm_10_runs_mean')
    plt.show()

def evaluation_analyze_rat_one_eval():
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)
    # df_0 = pd.read_csv('gpt_rat_responses_temp/temp = 0/rat_responses_gpt_temp_0_eval', sep='#')
    # df_1 = pd.read_csv('gpt_rat_responses_temp/temp = 0.25/rat_responses_gpt_temp_0.25_eval', sep='#')
    # df_2 = pd.read_csv('gpt_rat_responses_temp/temp = 0.5/rat_responses_gpt_temp_0.5_eval', sep='#')
    # df_3 = pd.read_csv('gpt_rat_responses_temp/temp = 0.75/rat_responses_gpt_temp_0.75_eval', sep='#')
    # df_4 = pd.read_csv('gpt_rat_responses_temp/temp = 1/rat_responses_gpt_temp_1_1_eval', sep='#')
    # df_5 = pd.read_csv('gpt_rat_responses_temp/temp = 1.25/rat_responses_gpt_temp_1.25_eval', sep='#')
    # df_6 = pd.read_csv('gpt_rat_responses_temp/temp = 1.5/rat_responses_gpt_temp_1.5_eval', sep='#')
    # df_7 = pd.read_csv('gpt_rat_responses_temp/temp = 1.75/rat_responses_gpt_temp_1.75_eval', sep='#')
    # df_8 = pd.read_csv('gpt_rat_responses_temp/temp = 2/rat_responses_gpt_temp_2_eval', sep='#')
    df_0 = pd.read_csv('gpt_rat_responses_topp/rat_responses_gpt_topp_0.1_eval', sep='#')
    df_1 = pd.read_csv('gpt_rat_responses_topp/rat_responses_gpt_topp_0.25_eval', sep='#')
    df_2 = pd.read_csv('gpt_rat_responses_topp/rat_responses_gpt_topp_0.5_eval', sep='#')
    df_3 = pd.read_csv('gpt_rat_responses_topp/rat_responses_gpt_topp_0.75_eval', sep='#')
    df_4 = pd.read_csv('gpt_rat_responses_temp/temp = 1/rat_responses_gpt_temp_1_eval', sep='#')
    df = pd.read_csv('data_changed_2.txt', sep=' ')
    counts = []
    for i in [20, 40, 60, 80, 100]:
        curr_mean_over = df[(df['Solving30sec'] < i) & (df['Solving30sec'] >= i - 20)]
        # print('<', i, '>=', i-20)
        curr_mean_over['Solving30sec'].to_csv('filter_human' + str(i) + '.csv')
        df_0_filter = df_0[(df_0['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_1_filter = df_1[(df_1['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_2_filter = df_2[(df_2['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_3_filter = df_3[(df_3['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        df_4_filter = df_4[(df_4['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_5_filter = df_5[(df_5['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_6_filter = df_6[(df_6['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_7_filter = df_7[(df_7['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_8_filter = df_8[(df_8['RATItems'].isin(curr_mean_over['RemoteAssociateItems']))]
        # df_2_filter.to_csv('filter_gpt' + str(i) + '.csv')
        # counts.append([i, curr_mean_over['Solving30sec'].count(), df_0_filter['eval'].sum(),
        #                df_1_filter['eval'].sum(), df_2_filter['eval'].sum(),
        #                df_3_filter['eval'].sum(), df_4_filter['eval'].sum(),
        #                df_5_filter['eval'].sum(), df_6_filter['eval'].sum(),
        #                df_7_filter['eval'].sum(), df_8_filter['eval'].sum()]) # mean
        counts.append([i, curr_mean_over['Solving30sec'].count(), df_0_filter['eval'].sum(),
                       df_1_filter['eval'].sum(), df_2_filter['eval'].sum(),
                       df_3_filter['eval'].sum(), df_4_filter['eval'].sum()])  # mean
    # df3 = pd.DataFrame(data=counts, columns=['percentage', 'max count', '0', '0.25',
    #                                              '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2'])
    df3 = pd.DataFrame(data=counts, columns=['percentage', 'max count', '0.1', '0.25',
                                             '0.5', '0.75', '1'])
    print(df3)
    fig, ax = plt.subplots(figsize=(70, 10))
    df3.plot(kind='bar', x='percentage', y=['max count', '0.1', '0.25',
                                                 '0.5', '0.75', '1'])
    # df3.plot(kind='bar', x='percentage', y=['max count', '0', '0.25',
                                            # '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2'])
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    ax.set_xticklabels(labels)
    plt.xlabel('Percentage of humans that solved these items')
    plt.ylabel('Count')
    plt.subplots_adjust(bottom=0.125)
    legend = plt.legend(loc='upper right')
    legend.get_title().set_fontsize('9')  # Set the title font size
    for item in legend.get_texts():
        item.set_fontsize('8')
    plt.savefig('rat_comparison_human_glm_10_runs_mean', bbox_inches='tight')
    plt.show()


# AUT
def bar_plot_multiple_items_aut(items, filenamegpt, filename_glm, filename_j2, filename_study, filename_llama):
    data = []
    std1 = []
    std2 = []
    std3 = []
    std4 = []
    std5 = []
    for item in items:
        study = pd.read_csv(filename_study + item + '.csv')['SemDis_glove_c_m'].dropna()
        # print(filename_study, filename_glm, filename_j2, filenamegpt, filename_llama)
        glm_data = pd.read_csv(filename_glm + item + '.csv')['SemDis_glove_c_m'].dropna()
        j2_data = pd.read_csv(filename_j2 + item + '.csv')['SemDis_glove_c_m'].dropna()
        gpt_data = pd.read_csv(filenamegpt + item + '.csv')['SemDis_glove_c_m'].dropna()
        llama_data = pd.read_csv(filename_llama + item + '.csv')['SemDis_GloVe_c_m'].dropna()
        study_all = pd.read_csv(filename_study + item + '.csv').dropna()
        # print(filename_study, filename_glm, filename_j2, filenamegpt, filename_llama)
        glm_data_all = pd.read_csv(filename_glm + item + '.csv')
        j2_data_all = pd.read_csv(filename_j2 + item + '.csv')
        gpt_data_all = pd.read_csv(filenamegpt + item + '.csv')
        llama_data_all = pd.read_csv(filename_llama + item + '.csv')
        std1.append(study.std())
        std2.append(glm_data.std())
        std3.append(j2_data.std())
        std4.append(gpt_data.std())
        std5.append(llama_data.std())
        data.append([item, study.mean(), glm_data.mean(), j2_data.mean(), gpt_data.mean(), llama_data.mean()])
        # max_row_study = study_all['SemDis_GloVe_c_m'].idxmax()
        # max_row_glm = glm_data_all['SemDis_GloVe_c_m'].idxmax()
        # max_row_j2 = j2_data_all['SemDis_GloVe_c_m'].idxmax()
        max_row_gpt = gpt_data_all['SemDis_glove_c_m'].idxmax()
        max_row_llama = llama_data_all['SemDis_GloVe_c_m'].idxmax()
        # print("Row with Maximum Score (study):", study.iloc[max_row_study])
        # print("Row with Maximum Score (glm_data):", glm_data.iloc[max_row_glm])
        # print("Row with Maximum Score (j2_data):", j2_data.iloc[max_row_j2])
        print("Row with Maximum Score (gpt_data):", gpt_data_all.loc[max_row_gpt]['response'], gpt_data_all.loc[max_row_gpt] ['SemDis_glove_c_m'])
        print(llama_data_all.loc[max_row_llama]['response'], llama_data_all.loc[max_row_llama]['SemDis_GloVe_c_m'])

        # print('mean', [item, study.mean(), glm_data.mean(), j2_data.mean(), gpt_data.mean(), llama_data.mean()])
        # print('max', [item, study.max(), glm_data.mean(), j2_data.max(), gpt_data.max(), llama_data.max()])
        # print('min', [item, study.min(), glm_data.min(), j2_data.min(), gpt_data.min(), llama_data.min()])
        # print('std', study.std(), glm_data.std(), j2_data.std(), gpt_data.std(), llama_data.std())
        # print(study, gpt_data)
        # t_statistic, p_value = ttest_ind(j2_datat, llama_data)
        # Print the results
        # print(item)
        # # print("T-Statistic:", t_statistic)
        # print("P-Value:", p_value)
        # if p_value < 0.05:
        #     print("Result: Significant")
        # else:
        #     print("Result: Not Significant")
    df3 = pd.DataFrame(data=data, columns=['item', 'study data', 'chatglm', 'j2', 'gpt', 'llama2'])
    plt.figure(figsize=(40, 15))
    ax = df3.plot(kind='bar', x='item', y=['study data', 'chatglm', 'j2', 'gpt', 'llama2'], yerr=[std1, std2, std3, std4, std5],
                  capsize=4, error_kw={'capthick': 0.75, 'elinewidth': 0.75})

    labels = df3['item'].to_list()
    ax.set_xticklabels(labels)
    ax.legend().set_visible(False)
    plt.xlabel('Item')
    plt.ylim(-0.09, 1.7)
    plt.ylabel('Mean semantic distance')
    plt.tight_layout()
    plt.savefig('aut_bar_all_items')
    plt.show()

def plot_aut_temp(filename_1='', filename_2='', filename_3='', filename_4='', filename_5='', filename_6='', filename_7='', filename_8='', filename_9='', temps=''):
    data = []
    filenames = [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6, filename_7, filename_8,
                 filename_9]
    for filename in filenames:
        if filename:
            # aut = pd.read_csv(filename + 'fork.csv')['SemDis_GloVe_c_m']
            aut = pd.read_csv(filename)['SemDis_GloVe_c_m']
            data.append(aut)

    data_means = [d.mean() for d in data]
    data_std = [d.std() for d in data]
    data_max = [max(d) for d in data]
    data_min = [min(d) for d in data]
    print('means', data_means)
    print('std', data_std)
    print('max', data_max)
    print('min', data_min)
    print(temps)
    bar_width = 0.1  # You can adjust this value
    plt.bar(temps, data_means, yerr=data_std, width=bar_width)
    plt.ylabel('Mean AUT score')
    plt.xlabel('Temperature')
    plt.title('AUT')
    plt.show()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            t_stat, p_value = ttest_ind(data[i].dropna(), data[j].dropna())
            print(f"T-test between {temps[i]} and {temps[j]}:")
            # print(f"t-statistic: {t_stat}")
            # print(f"p-value: {p_value}")
            if p_value < 0.05:
                print("There is a significant difference between the data sets.")
            else:
                print("There is no significant difference between the data sets.")


# bar_plot_multiple_items_aut(['book', 'fork', 'tin_can', 'rope', 'brick', 'box'], 'gpt_aut_responses/SemDis_all_basic_', 'glm_aut_responses/SemDis_all_basic_',
#                         'j2_aut_responses/SemDis_all_basic_', 'AUT_data_humans_filtered/semdis_', 'llama2_aut_responses/SemDIs_all_basic_') # bar plot AUT with all items

# bar_plot_multiple_items_aut(['book', 'fork', 'tin_can'], 'gpt_aut_responses/SemDis_all_basic_', 'glm_aut_responses/SemDis_all_basic_',
#                         'j2_aut_responses/SemDis_all_basic_', 'AUT_data_humans_filtered/semdis_', 'llama2_aut_responses/SemDIs_all_basic_') # bar plot AUT with book, fork, tin can


# bar_plot_multiple_items_aut(['rope', 'brick', 'box'], 'gpt_aut_responses/SemDis_all_basic_', 'glm_aut_responses/SemDis_all_basic_',
#                         'j2_aut_responses/SemDis_all_basic_', 'AUT_data_humans_filtered/semdis_', 'llama2_aut_responses/SemDIs_all_basic_') # bar plot AUT with rope, brick, box



# compare_ff_data(['bear', 'candle', 'snow', 'toaster', 'paper', 'table'], 'gpt_ff_responses/summary_',
#                 'glm_ff_responses/summary_glm_', 'j2_ff_responses/summary_j2_', 'llama2_ff_responses/summary_') # bar plot FF with all seed words


# plot_dat('DAT_data_humans/study2.tsv', 'gpt_dat_responses/scores_100.txt', 'glm_dat_responses/dat-scored.tsv', 'j2_dat_responses/dat-scored.tsv', 'llama2_dat_responses/dat_scored_llama2.tsv') # DAT plot


# evaluation_analyze_rat() # full RAT plot

