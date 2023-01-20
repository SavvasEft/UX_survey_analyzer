# Unlock the Power of Open-Ended Feedback: Discover Insights with this Survey Analysis Tool

This tool allows the user to gain a deeper understanding of customer feedback.The user can easily identify topics covered in open-ended questions.
This tool is perfect for UX researchers looking to gain a deeper understanding of customer feedback and minimize the tedious manual analysis of open-ended answers.

## Main characteristics:

### Feature analysis of answers
- Effortlessly Identify Keywords and Features in Open-Ended Survey Responses
- Uncover Hidden Patterns in Your Survey Data
- Remove Noise and Zero-Information Features for Cleaner Data Analysis
- Group and Replace Features for More Focused Analysis

### Clustering analysis
- Cluster Similar Responses for a Clear Understanding of Feedback
- Viewing the Overall Distribution of Clusters and Answer Counts
- Uncover the Topics of Feedback by drilling down into Individual Clusters

### Prepare reports
Have all the information you need in a Clustered_data.xlsx file. This includes:
- The identified cluster for each answer, and a cluster-score per answer ('Question_1' sheet)
- The identified clusters with their main feature ('model' sheet)
- A clustering report, that allows you to evaluate the clustering performed ('clustering_report' sheet)

### Requirements:
Four files are necessary:
   - answers to be analyzed in raw_data.xlsx file named 'raw_data.xlsx' of the following structure:
      - question is on the first row
      - answers for one open-ended question below the question
      - one answer per row, all in the first column

   - 'parameters.xlsx': a file that has the values for the necessary parameters for the program to run. These parameters are:
      - PRINT_INFO : print details - usefull when tuning with dictionaries (True/False)             
      - CLUSTER_NO : number of clusters (integer)             
      - TOP_FEATURES : number of top features to keep (integer)
      - REPORT : Produce clustering report file named 'clustering_report.xlsx' (True/False)
      - SAVE_FEATURES : Saves identified features in features.xlsx (True/False)
      - PREDICT : Option to predict the cluster of an input sentence by the user and it's score (True/False) - currently not working correctly
      - N_GRAMS_MIN : Define minimum N-gram to consider as feature (int)
      - N_GRAMS_MAX : Define maximum N-gram to consider as feature (int)
      - PORTER_OR_LEMMATIZER : choose porter stemmer (1) or WordNetLemmatizer (2)
      - SAVE_VIZ : save vizualizations in current folder (True/False)
   - 'topic_dictionary.xlsx'

   - 'stop_word_list.xlsx'

- dependencies:
   python 3, numpy, pandas, nltk, scikitlearn, matplotlib