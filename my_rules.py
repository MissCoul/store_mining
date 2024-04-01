import argparse
import csv
import operator
import os
from collections import defaultdict
from itertools import combinations, chain
import time
import pandas as pd
from csv import writer
import matplotlib.pyplot as plt

info = {}
headers = ['Min Support', 'Min Confidence', 'Time', 'Rules']
# bar_plot = {'Min Support': [], 'Min Confidence': [], 'Time': [], 'Rules': []}
bar_plot = {}
bar_for_program = {}
output_dir = './output results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
glob_df = pd.read_csv("small.txt", sep=" ", index_col=False)
df_high = glob_df['1'].value_counts()
df_high = df_high.sort_values(ascending=True)
print("frequent item is", df_high.iloc[0])


def preprocess_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    unique_itemset = set()

    items = defaultdict(list)
    for line in lines:
        split_data = line.split(' ')
        items[split_data[0]].append(split_data[1].replace('\n', ''))
        if not split_data[1] in unique_itemset:
            unique_itemset.add(split_data[1])

    path = f'{output_dir}items.csv'
    csv_file = open(path, 'w')

    for key, value in items.items():
        v = ','.join(value)
        csv_file.write(f'{v}\n')

    csv_file.close()
    info['Number of transaction'] = len(items.keys())
    print(f'length of the transactions is {len(items.keys())}')
    info['Number of items'] = len(unique_itemset)
    print(f'The items are {len(unique_itemset)}')
    return path


class Apriori:
    def __init__(self, minSupport, minConfidence):
        self.support_count = defaultdict(int)
        self.minSupport = minSupport - 1
        self.minConfidence = minConfidence

    def read_transactions_from_file(self, transaction_file):
        with open(transaction_file, 'r') as infile:
            transactions = [set(line.rstrip('\n').split(','))
                            for line in infile]

            return transactions

    def get_one_itemset(self, transactions):
        one_itemset = set()
        for transaction in transactions:
            for item in transaction:
                one_itemset.add(frozenset([item]))

        return one_itemset

    def self_cross(self, Ck, itemset_size):
        Ck_plus_1 = {itemset1.union(itemset2)
                     for itemset1 in Ck for itemset2 in Ck
                     if len(itemset1.union(itemset2)) == itemset_size}
        return Ck_plus_1

    def prune_Ck(self, Ck, Lk_minus_1, itemset_size):
        Ck_ = set()
        for itemset in Ck:
            Ck_minus_1 = list(combinations(itemset, itemset_size - 1))
            flag = 0
            for subset in Ck_minus_1:
                if not frozenset(subset) in Lk_minus_1:
                    flag = 1
                    break
            if flag == 0:
                Ck_.add(itemset)
        return Ck_

    def get_min_supp_itemsets(self, Ck, transactions):
        itemset_st = time.time()
        temp_freq = defaultdict(int)

        for transaction in transactions:
            for itemset in Ck:
                if itemset.issubset(transaction):
                    temp_freq[itemset] += 1
                    self.support_count[itemset] += 1
        N = len(transactions)
        Lk = [itemset for itemset, freq in temp_freq.items()
              if freq > self.minSupport]
        itemset_et = time.time()
        elapsed_time = itemset_et - itemset_st
        return set(Lk), elapsed_time

    def apiori(self, transactions):
        freq_itemset_count = 0
        total_exec_time = 0
        print(
            f'minsup : {int(self.minSupport + 1)}     minconf : {self.minConfidence}')
        info['Minsup'] = int(self.minSupport + 1)
        info['minconf'] = self.minConfidence
        bar_plot['Min Support'] = int(self.minSupport + 1)
        bar_plot['Min Confidence'] = self.minConfidence
        # bar_plot.append(int(self.minSupport + 1))
        # bar_plot.append(self.minConfidence)
        K_itemsets = dict()
        Ck = self.get_one_itemset(transactions)
        Lk, elapsed_time_1 = self.get_min_supp_itemsets(Ck, transactions)
        k = 2
        info['Number of frequent 1-itemset'] = len(Lk)
        print(f'1-itemsets:: {len(Lk)}')
        bar_for_program['Frequent items'] = len(Lk)
        freq_itemset_count += len(Lk)

        total_exec_time += elapsed_time_1
        while len(Lk) != 0:
            K_itemsets[k - 1] = Lk
            Ck = self.self_cross(Lk, k)
            Ck = self.prune_Ck(Ck, Lk, k)

            Lk, elapsed_time_2 = self.get_min_supp_itemsets(Ck, transactions)

            if len(Lk) == 0:
                break
            print(f'{k}-itemsets: {len(Lk)}')
            bar_for_program[f'{k}-itemsets'] = len(Lk)
            text = f'Number of frequent {k}-itemset'
            info[text] = len(Lk)
            freq_itemset_count += len(Lk)
            total_exec_time += elapsed_time_2
            k += 1
        highest_k = k
        text_1 = f'The length of the largest {highest_k -1} -itemset'
        info['Time in seconds to find the frequent itemsets'] = total_exec_time
        bar_plot['Time'] = total_exec_time
        info[text_1] = highest_k - 1
        info['Total number of frequent itemsets'] = freq_itemset_count
        print(f'Frequent items are {freq_itemset_count}')
        return K_itemsets

    def subsets(self, iterable):
        list_ = list(iterable)
        subsets_ = chain.from_iterable(combinations(
            list_, len) for len in range(len(list_) + 1))
        subsets_ = list(map(frozenset, subsets_))

        return subsets_

    def get_rules(self, K_itemsets):
        confidence_st = time.time()
        rules = list()
        highest_rule = ''
        highest_confidence = 0
        for key, k_itemset in K_itemsets.items():
            if key > 1:
                for itemset in k_itemset:
                    sub_itemsets = {subset for subset in self.subsets(itemset) if
                                    (subset != set() and len(subset) != len(itemset))}
                    for subset in sub_itemsets:
                        left = subset
                        right = itemset.difference(subset)
                        confidence = self.support_count[itemset] / \
                            self.support_count[left]

                        if confidence >= highest_confidence:
                            highest_confidence = confidence
                            highest_rule = (
                                list(left), list(right), confidence)
                        if confidence > self.minConfidence:
                            rules.append((list(left), list(right), confidence))

        rules.sort(key=operator.itemgetter(2), reverse=True)
        confidence_et = time.time()
        confidence_time = confidence_et - confidence_st
        no_of_high_confidence_rules = len(rules)
        info['Number of high confidence rules'] = no_of_high_confidence_rules
        bar_plot['Rules'] = no_of_high_confidence_rules
        print(f'Number of high confidence rules {no_of_high_confidence_rules}')
        info['The rule with the high confidence'] = highest_rule
        print(f'highest_confidence {highest_confidence}')
        info["Time in  seconds to find the cofidence rules"] = confidence_time
        print(f'highest_confidence time {confidence_time}')
        return rules

    def write_info(self, K_itemsets, rules, transactions_count):
        os.remove(f'{output_dir}items.csv')
        N = transactions_count
        outfile_path = f'./output results/3_freq_items.txt'
        with open(outfile_path, 'w') as outfile:
            tot_itemset_count = 0
            for key, values in K_itemsets.items():
                count = 0
                for value in values:

                    support = self.support_count[value] / N
                    support_ct = self.support_count[value]
                    count += 1
                    tot_itemset_count += 1
                    list_values = list(value)
                    itemset_joined_string = ' '.join(
                        [str(v) for v in list_values])
                    outfile.write(f'{itemset_joined_string} | {support_ct}\n')
        if not self.minConfidence == -1:
            outfile_path = f'./output results/3_freq_rules.txt'

            with open(outfile_path, 'w') as outfile:
                for rule in rules:
                    support_l = self.support_count[frozenset(rule[0])]
                    support_r = self.support_count[frozenset(rule[1])]
                    rule_l = ' '.join([str(v) for v in rule[0]])
                    rule_r = ' '.join([str(v) for v in rule[1]])
                    outfile.write(
                        f'{rule_l} | {rule_r} | {support_l} | {round(rule[2], 4):.4f}\n')

        outfile_path = f'./output results/Item_info.txt'
        with open(outfile_path, 'w') as outfile:
            for key, value in info.items():

                k = key
                val = value
                outfile.write(
                    f'{k} : {val} \n')


parser = argparse.ArgumentParser(
    description='Apriori Algorithms')
parser.add_argument('data_file', type=str, help='File with data')
parser.add_argument('minsup', type=float, help='Minimum support count')
parser.add_argument('minconf', type=float, help='Minimum confidence for rules')

args = parser.parse_args()

in_transaction_file = preprocess_file(args.data_file)
info['Input file'] = args.data_file
ap = Apriori(minSupport=args.minsup, minConfidence=args.minconf)
transactions = ap.read_transactions_from_file(in_transaction_file)
K_itemsets = ap.apiori(transactions)
rules = ap.get_rules(K_itemsets)
ap.write_info(K_itemsets, rules, len(transactions))

# bar plot

frequ = list(bar_for_program.values())
freq_key = list(bar_for_program.keys())

plt.bar(freq_key, frequ)
plt.xlabel('Frequent itemset')
plt.ylabel('Itemset')
plt.savefig('./output results/for bar/Freq_plot_items.png')


read_csv_path = './output results/for bar/bar.csv'
if not os.path.exists('./output results/for bar/'):
    os.makedirs('./output results/for bar/')
    with open(read_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvfile.close()
