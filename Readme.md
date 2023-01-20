{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Unlock the Power of Open-Ended Feedback: Discover Insights with this Survey Analysis Tool\n",
    "\n",
    "This tool allows the user to gain a deeper understanding of customer feedback.The user can easily identify topics covered in open-ended questions.\n",
    "This tool is perfect for UX researchers looking to gain a deeper understanding of customer feedback and minimize the tedious manual analysis of open-ended answers.\n",
    "\n",
    "## Main characteristics:\n",
    "\n",
    "### Feature analysis of answers\n",
    "- Effortlessly Identify Keywords and Features in Open-Ended Survey Responses\n",
    "- Uncover Hidden Patterns in Your Survey Data\n",
    "- Remove Noise and Zero-Information Features for Cleaner Data Analysis\n",
    "- Group and Replace Features for More Focused Analysis\n",
    "\n",
    "### Clustering analysis\n",
    "- Cluster Similar Responses for a Clear Understanding of Feedback\n",
    "- Viewing the Overall Distribution of Clusters and Answer Counts\n",
    "- Uncover the Topics of Feedback by drilling down into Individual Clusters\n",
    "\n",
    "### Prepare reports\n",
    "Have all the information you need in a Clustered_data.xlsx file. This includes:\n",
    "- The identified cluster for each answer, and a cluster-score per answer ('Question_1' sheet)\n",
    "- The identified clusters with their main feature ('model' sheet)\n",
    "- A clustering report, that allows you to evaluate the clustering performed ('clustering_report' sheet)\n",
    "\n",
    "### Requirements:\n",
    "Four files are necessary:\n",
    "   - answers to be analyzed in raw_data.xlsx file named 'raw_data.xlsx' of the following structure:\n",
    "      - question is on the first row\n",
    "      - answers for one open-ended question below the question\n",
    "      - one answer per row, all in the first column\n",
    "\n",
    "   - 'parameters.xlsx': a file that has the values for the necessary parameters for the program to run. These parameters are:\n",
    "      - PRINT_INFO : print details - usefull when tuning with dictionaries (True/False)             \n",
    "      - CLUSTER_NO : number of clusters (integer)             \n",
    "      - TOP_FEATURES : number of top features to keep (integer)\n",
    "      - REPORT : Produce clustering report file named 'clustering_report.xlsx' (True/False)\n",
    "      - SAVE_FEATURES : Saves identified features in features.xlsx (True/False)\n",
    "      - PREDICT : Option to predict the cluster of an input sentence by the user and it's score (True/False) - currently not working correctly\n",
    "      - N_GRAMS_MIN : Define minimum N-gram to consider as feature (int)\n",
    "      - N_GRAMS_MAX : Define maximum N-gram to consider as feature (int)\n",
    "      - PORTER_OR_LEMMATIZER : choose porter stemmer (1) or WordNetLemmatizer (2)\n",
    "      - SAVE_VIZ : save vizualizations in current folder (True/False)\n",
    "   - 'topic_dictionary.xlsx'\n",
    "\n",
    "   - 'stop_word_list.xlsx'\n",
    "\n",
    "- dependencies:\n",
    "   python 3, numpy, pandas, nltk, scikitlearn, matplotlib"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)]"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "81794d4967e6c3204c66dcd87b604927b115b27c00565d3d43f05ba2f3a2cb0d"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
