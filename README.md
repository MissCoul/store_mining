# Apriori Algorithm Implementation

This repository contains an implementation of the Apriori algorithm, a popular algorithm for finding frequent itemsets in transactional databases. The algorithm is used for association rule mining in data mining and is commonly applied in market basket analysis.

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Output](#output)
5. [Contributing](#contributing)

## Introduction

The Apriori algorithm implemented in this codebase is written in Python. It takes transactional data as input and generates frequent itemsets and association rules based on specified minimum support and confidence thresholds.

## Usage

To run the code, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Execute the script `apriori.py` with the following command:

python apriori.py data_file minsup minconf

Replace `data_file` with the path to your transactional data file, `minsup` with the minimum support count, and `minconf` with the minimum confidence for generating association rules.

## Dataset

The dataset used for this implementation should be in a text file format, where each line represents a transaction, and items within transactions are separated by commas or spaces.

## Output

The algorithm generates several output files in the `output results` directory:

- `3_freq_items.txt`: Contains frequent itemsets along with their support counts.
- `3_freq_rules.txt`: Contains association rules with their antecedents, consequents, support counts, and confidence values.
- `Item_info.txt`: Provides information about the dataset and algorithm execution, including the number of transactions, items, frequent itemsets, high-confidence rules, and execution time.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


