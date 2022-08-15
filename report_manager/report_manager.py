#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getopt
import warnings
import pandas as pd
import logging
import time
import os
import os.path
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats
from scipy.stats import linregress
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import pandas_profiling as pp
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class ReportManager:
    def __init__(self):

        # Report Merger
        self.file_1 = None
        self.file_2 = None
        self.df_1 = None
        self.df_2 = None
        self.df_1_join_keys = None
        self.df_2_join_keys = None
        self.join_type = "inner"
        self.report_name = "joined_report_export"
        self.report_name_csv = str(report_name) + ".csv"
        self.report_name_xlsx = str(report_name) + ".xlsx"
        self.save_directory = os.getcwd()
        self.csv_export = save_directory + '\\' + report_name_csv
        self.excel_export = save_directory + '\\' + report_name_xlsx
        self.df_final = None

        # Pandas Profiling
        self.file_raw = None
        self.df_raw = None
        self.report_name = "pandas-profiling_export"
        self.report_title = "Title"
        self.report_name_html = str(report_name) + ".html"
        self.save_directory = os.getcwd()
        self.export = save_directory + '\\' + report_name
        self.profile = None

        # Custom Report
        self.csv_path = os.getcwd()
        self.plot_path = os.getcwd()
        self.report_path = os.getcwd()
        self.clean_csv_path = os.getcwd()
        self.data = None
        self.nan_prop = None
        self.categorical_variable = None
        self.numerical_variable = None

        # Plot
        self.num_var_combination = None
        self.cat_var_combination = None
        self.catnum_combination = None

    def set_file_1(self, file_1):
        self.file_1 = file_1

    def set_file_2(self, file_2):
        self.file_2 = file_2

    def set_join_type(self, join_type):
        print("Join Type Selected: ", join_type)
        self.join_type = join_type

    def get_join_type(self):
        return self.join_type

    def set_report_name(self, report_name):
        if report_name != "":
            print("Report Name: ", report_name)
            self.report_name = report_name
            self.csv_export = self.save_directory + '\\' + str(self.report_name)+".csv"
            self.excel_export = self.save_directory + '\\' + str(self.report_name)+".xlsx"
            self.report_name = new_report_name
            self.report_name_html = str(new_report_name) + ".html"
            self.export = self.save_directory + '\\' + str(self.report_name) + ".html"
        else:
            print("Report Name was Blank")

    def get_report_name(self):
        return self.report_name

    def set_save_directory(self, directory):
        self.save_directory = directory
        print("New Directory: ",  self.save_directory)
        self.csv_export = self.save_directory + '\\' + self.report_name_csv
        self.excel_export = self.save_directory + '\\' + self.report_name_xlsx
        self.set_pandas_save_directory(directory)
        self.set_csv_path(directory)
        self.set_plot_path(directory)
        self.set_report_path(directory)
        self.set_clean_csv_path(directory)

    def set_pandas_save_directory(self, save_path):
        self.save_directory = save_path
        print("New Directory: ", self.save_directory)
        self.export = self.save_directory + '\\' + self.report_name_html

    # Load Files to Dataframe 1
    def load_dataframe_1(self):
        print("Loading Data to Dataframe 1")
        try:
            if self.file_1.endswith('.csv'):
                print("File 1 is a CSV")
                self.df_1 = pd.read_csv(self.file_1, engine='python')
            elif self.file_1.endswith('.xlsx'):
                print("File 1 is a Excel")
                self.df_1 = pd.read_excel(self.file_1)
            self.df_1.columns = self.df_1.columns.str.replace(' ', '_')
            #self.df_1 = self.df_1.astype(str)
            print("DTYPES Dataframe 1: ", self.df_1.dtypes)
            #self.df_1.columns = self.df_1.columns.str.strip()
            print(self.df_1)
        except pd.errors.ParserError:
            print("Error")
            return 1

    # Load Files to Dataframe
    def load_dataframe_2(self):
        print("Loading Data to Dataframe 2")
        try:
            if self.file_2.endswith('.csv'):
                print("File 2 is a CSV")
                self.df_2 = pd.read_csv(self.file_2, engine='python')
            elif self.file_2.endswith('.xlsx'):
                print("File 2 is a Excel")
                self.df_2 = pd.read_excel(self.file_2)
            self.df_2.columns = self.df_2.columns.str.replace(' ', '_')
            #self.df_2 = self.df_2.astype(str)
            print("DTYPES Dataframe 2: ", self.df_2.dtypes)
            #self.df_2.columns = self.df_2.columns.str.strip()
            print(self.df_2)
            return 0
        except pd.errors.ParserError:
            print("Error")
            return 1

    # Set Data Frame Column Type
    def set_columndtype(self, file, column, data_type):
        try:
            if file == 1:
                print("Changing Dtype File 1")
                if data_type == "string":
                    self.df_1[column] = self.df_1[column].astype(str)
                    return 0
                if data_type == "date":
                    self.df_1[column] = pd.to_datetime(self.df_1[column])
                    return 0
                if data_type == "int":
                    self.df_1[column] = self.df_1[column].astype(int)
                    return 0
            elif file == 2:
                print("Changing Dtype File 2")
                if data_type == "string":
                    self.df_2[column] = self.df_2[column].astype(str)
                    return 0
                if data_type == "date":
                    self.df_2[column] = pd.to_datetime(self.df_2[column])
                    return 0
                if data_type == "int":
                    self.df_2[column] = self.df_2[column].astype(int)
                    return 0
        except Exception as e:  # print(e)(pd.io.sql.DatabaseError, AttributeError) as Exception_e:
            print("Error Setting Data Type: ", e)
            return 1
        print("Completed Successfully")

    def get_features(self, df):
        concat_str = ""
        for col in df.columns:
            concat_str = concat_str + " " + str(col)
        return concat_str

    def get_df1(self):
        return self.df_1

    def get_df2(self):
        return self.df_2

    def set_df1_join_keys(self, df_1_join_keys):
        self.df_1_join_keys = df_1_join_keys

    def set_df2_join_keys(self, df_2_join_keys):
        self.df_2_join_keys = df_2_join_keys

    def get_df1_join_keys(self):
        return self.df_1_join_keys

    def get_df2_join_keys(self):
        return self.df_2_join_keys

    # Join Based off Condition
    def join_data(self, df_1_join_keys=None, df_2_join_keys=None):
        # You can use this function by calling it with your join keys already set. Otherwise you can manaully set them using the setters.
        if df_1_join_keys and df_2_join_keys:
            self.set_df1_join_keys(df_1_join_keys)
            self.set_df2_join_keys(df_2_join_keys)
        if self.join_type == "append":
            if len(self.df_1.columns) == len(self.df_2.columns):
                print("DataFrame 1 Before Append: ", self.df_1)
                self.df_final = self.df_1.append(self.df_2)
                print("DataFrame 1 After Append: ", self.df_1)
                #self.df_final = self.df_1.copy()
                print(self.df_final)
            else:
                print("Files Have Different Column Lengths")
                return "Files Have Different Column Lengths"
        else:
            print("Joining data")
            try:
                if len(self.df_1_join_keys) > 0 and len(self.df_2_join_keys) > 0:
                    self.df_final = pd.merge(self.df_1, self.df_2, left_on=self.df_1_join_keys, right_on=self.df_2_join_keys, how=self.join_type)
                else:
                    print("One of your keys was empty")
            except Exception as e:
                print("Error: ", e)
        print(self.df_final)

    def set_csv_path(self, new_csv_path):
        self.csv_path = new_csv_path

    def set_plot_path(self, new_plot_path):
        self.plot_path = new_plot_path

    def set_report_path(self, new_report_path):
        self.report_path = new_report_path

    def set_clean_csv_path(self, new_clean_csv_path):
        self.clean_csv_path = new_clean_csv_path

    def run_analysis(self):
        try:
            # Creating a txt file in the report_path
            print("Creating txt file in the report_path")
            filename = os.path.join(self.report_path, "report" + ".txt")

            # Assigning csv file to a variable call 'data'
            print("Assigning csv file to a variable call 'data'")
            self.data = self.df_raw.copy()

            # Create a function to separate out numerical and categorical data
            # Using this function to ensure that all non-numerical in a numerical column
            # and non-categorical in a categorical column is annotated

            print("Assigning Categorical and Numerical Variables")
            self.categorical_variable = self.cat_variable(self.data)
            self.numerical_variable = self.num_variable(self.data)
            print("Assigning Categorical and Numerical Variables Completed")

            # Assigning variable filename to report and enable writing mode
            print("Opening Report")
            report = open(filename, "w")
            print("Opening Report Complete")

            # Execute overview function in model module
            self.data = self.model_overview(self.data, self.numerical_variable, report)

            # Create a function to decide whether to drop all NA values or replace them
            # Drop it if NAN count < 5 %
            self.nan_prop = (self.data.isna().mean().round(2) * 100)  # Show % of NaN values per column

            cols_to_drop = self.drop_na()

            print("Dropped NA Values")
            self.data = self.data.dropna(subset=cols_to_drop)

            # Using Imputer to fill NaN values
            # Counting the proportion of NaN
            cols_to_fill = self.fill_na()

            cat_var_tofill = []
            num_var_tofill = []

            for var in cols_to_fill:
                if var in self.categorical_variable:
                    cat_var_tofill.append(var)
                else:
                    num_var_tofill.append(var)

            print("Categorical Values")
            imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            try:
                self.data[cat_var_tofill] = imp_cat.fit_transform(self.data[cat_var_tofill])
            except ValueError:
                pass

            print("Number Values")
            imp_num = SimpleImputer(missing_values=np.nan, strategy='median')
            try:
                self.data[num_var_tofill] = imp_num.fit_transform(self.data[num_var_tofill])
            except ValueError:
                pass

            self.data = self.outlier()

            # Creating possible combinations among a list of numerical variables
            num_var_combination = list(itertools.combinations(self.numerical_variable, 2))

            # Creating possible combinations among a list of categorical variables
            cat_var_combination = list(itertools.combinations(self.categorical_variable, 2))

            # Creating possible combinations among a list of numerical and categorical variuable
            catnum_combination = list(itertools.product(self.numerical_variable, self.categorical_variable))

            print("Running Model")
            # Running the report now
            self.model_run(num_var_combination, catnum_combination, cat_var_combination, report, self.data)
            # Create an output file that shows cleaned data
            data2 = self.data.copy()
            data2.to_csv(r'{}/cleaned_csv.csv'.format(self.clean_csv_path), index=False)

            print("Running Plots")
            # Running plot class from Graph package
            self.plot.run(self.data, self.categorical_variable, self.numerical_variable, num_var_combination,
                          cat_var_combination, catnum_combination, self.plot_path)
            return 0
        except Exception as e:
            print("[ERROR]: ", e)
            return e

    @staticmethod
    def cat_variable(df):
        return list(df.select_dtypes(include=['category', 'object']))

    @staticmethod
    def num_variable(df):
        return list(df.select_dtypes(exclude=['category', 'object']))

    def drop_na(self):
        return [i for i, v in self.nan_prop.items() if 5 > v > 0]

    # Create a function to process outlier data
    def outlier(self):
        z = np.abs(stats.zscore(self.data[self.numerical_variable]))
        z_data = self.data[(z < 3).all(axis=1)]  # Remove any outliers with Z-score > 3 or < -3
        return z_data

    def fill_na(self):
        return [i for i, v in self.nan_prop.items() if v > 5]

    # Pandas Profiling Create Report
    def create_report(self, sample=None, minimal=False):
        # Create Profile Report off of dataframe
        self.df_raw.replace(['None', 'Null'], np.nan)
        self.df_raw.isnull().sum(axis=0).to_frame().rename(columns={0: 'Count_Nulls'})
        if sample is None:
            self.profile = pp.ProfileReport(self.df_raw, title=self.report_title, minimal=minimal,
                                            html={'style': {'full_width': True}}, progress_bar=True)
        else:
            self.profile = pp.ProfileReport(self.df_raw.sample(n=sample), title=self.report_title, minimal=minimal,
                                            html={'style': {'full_width': True}}, progress_bar=True)

    # Pandas Profiling Set File
    def set_file(self, file_raw):
        self.file_raw = file_raw

    def set_report_title(self, new_report_title):
        self.report_title = new_report_title

    def get_report_title(self):
        return self.report_title

    # Load Files to Dataframe
    def load_dataframe(self):
        print("Loading Data to Dataframe 1")
        try:
            if self.file_raw.endswith('.csv'):
                print("File 1 is a CSV")
                self.df_raw = pd.read_csv(self.file_raw, dtype=str, engine='python')
            elif self.file_raw.endswith('.xlsx'):
                print("File 1 is a Excel")
                self.df_raw = pd.read_excel(self.file_raw, dtype=str)
            self.df_raw.columns = self.df_raw.columns.str.replace(' ', '_')
            print(self.df_raw)
        except pd.errors.ParserError:
            print("Error")
            return 1

    @staticmethod
    def get_features(df):
        concat_str = ""
        for col in df.columns:
            concat_str = concat_str + " " + str(col)
        return concat_str

    def get_df(self):
        return self.df_raw

    def export_report(self):
        print("Exporting Data")
        self.profile.to_file(output_file=self.export)

    def model_overview(self, df, numerical_variable, report):
        data_head = df.head()
        data_shape = df.shape
        data_type = df.dtypes
        df = (df.drop(numerical_variable, axis=1).join(df[numerical_variable].apply(pd.to_numeric,
                                                                                    errors='coerce')))  # Converts any non-numeric values in a numerical column into NaN
        null_values = df.isnull().sum()
        zero_prop = ((df[df == 0].count(axis=0) / len(df.index)).round(2) * 100)
        data_summary = df.describe()
        report.write(
            """______Exploratory data analysis summary by Edator______\n\n\n\nThe first 5 rows of content comprise of:
            \n\n{}\n\n\nThere are a total of {} rows and {} columns.\n\n\nThe data type for each column is:\n\n{}\n\n\n
            Number of NaN values for each column:\n\n{}\n\n\n% of zeros in each column:\n\n{}\n\n\n
            The summary of data:\n\n{}""".format(data_head, data_shape[0], data_shape[1],
                                                 data_type, null_values, zero_prop, data_summary)
        )
        return df

    # Creating report for correlation
    def model_run(self, num_var_combination, catnum_combination, cat_var_combination, report, data):
        # Pearson correlation (Numerical)
        report.write("\n\n\n__________Correlation Summary (Pearson)__________")
        for i in num_var_combination:
            var1 = i[0]
            var2 = i[1]
            pearson_data = linregress(data[var1], data[var2])
            pearson_r2, pearson_pvalue = ((pearson_data[2] ** 2), pearson_data[3])
            report.write("\n\nThe Pearson R_Square and Pearson P-values between {} and {} are {} and {} respectively."
                         .format(var1, var2, pearson_r2, pearson_pvalue))

        # Spearsman correlation (Ordinal)
        report.write("\n\n\n\n__________Correlation Summary (Spearsman)__________")
        for q in num_var_combination:
            var1 = q[0]
            var2 = q[1]
            spearsman_data = stats.spearmanr(data[var1], data[var2])
            spearsman_r2, spearsman_pvalue = ((spearsman_data[0] ** 2), spearsman_data[1])
            report.write(
                "\n\nThe Spearsman R_Square and Spearsman P-values between {} and {} are {} and {} respectively."
                .format(var1, var2, spearsman_r2, spearsman_pvalue))

        # For numeric-categorical variables
        # ONE WAY ANOVA (Cat-num variables)
        report.write("\n\n\n\n__________Correlation Summary (One Way ANOVA)__________")
        for j in catnum_combination:
            var1 = j[0]
            var2 = j[1]
            lm = ols('{} ~ {}'.format(var1, var2), data=data).fit()
            table = sm.stats.anova_lm(lm)
            one_way_anova_pvalue = table.loc[var2, 'PR(>F)']
            report.write("\n\nThe One Way ANOVA P-value between {} and {} is {}."
                         .format(var1, var2, one_way_anova_pvalue))

        # For categorical-categorical variables
        # Chi-Sq test
        report.write("\n\n\n\n__________Correlation Summary (Chi Square Test)__________")
        for k in cat_var_combination:
            cat1 = k[0]
            cat2 = k[1]
            chi_sq = pd.crosstab(data[cat1], data[cat2])
            chi_sq_result = chi2_contingency(chi_sq)
            report.write("\n\nThe Chi-Square P-value between {} and {} is {}."
                         .format(cat1, cat2, chi_sq_result[1]))

        report.close()

    def plot_run(self, data, categorical_variable, numerical_variable, num_var_combination, cat_var_combination,
                 catnum_combination, plot_save_path):
        self.num_var_combination = num_var_combination
        self.cat_var_combination = cat_var_combination
        self.catnum_combination = catnum_combination
        # Set Unique categorical values that are < 5 as hue
        hue_lst = []
        for x in categorical_variable:
            if len(set(data[x])) <= 5:  # if we have less than 5 unique values, we will use it for hue attributes
                hue_lst.append(x)
        # Creating possible combinations among a list of numerical variables
        self.num_var_combination = list(itertools.combinations(numerical_variable, 2))
        # Creating possible combinations among a list of categorical variables
        self.cat_var_combination = list(itertools.combinations(categorical_variable, 2))
        # Creating possible combinations among a list of numerical and categorical variuable
        self.catnum_combination = list(itertools.product(numerical_variable, categorical_variable))

        # Using regplot for numerical-numerical variables
        num_var_hue_combination = list(itertools.product(self.num_var_combination, hue_lst))
        for i in num_var_hue_combination:
            var1 = i[0][0]
            var2 = i[0][1]
            hue1 = i[1]
            plot1 = sns.scatterplot(data=data, x=var1, y=var2, hue=hue1)
            fig1 = plot1.get_figure()
            fig1.savefig(plot_save_path + "/{} vs {} by {} scatterplot.png".format(var1, var2, hue1))
            fig1.clf()

        # Using countplot for categorical data
        for j in categorical_variable:
            plot2 = sns.countplot(data=data, x=j)
            fig2 = plot2.get_figure()
            fig2.savefig(plot_save_path + "/{}_countplot.png".format(j))
            fig2.clf()

        # Using boxplot for numerical + Categorical data
        for k in self.catnum_combination:
            num1 = k[0]
            cat1 = k[1]
            plot3 = sns.boxplot(data=data, x=cat1, y=num1)
            fig3 = plot3.get_figure()
            fig3.savefig(plot_save_path + "/{}_{}_barplot.png".format(num1, cat1))
            fig3.clf()

        # Creating heatmap to show correlation
        le = LabelEncoder()
        for cat in data[categorical_variable]:
            data[cat] = le.fit_transform(data[cat])
        plt.figure(figsize=(15, 10))
        corr_matrix = data.corr()
        plot4 = sns.heatmap(corr_matrix, annot=True)
        fig4 = plot4.get_figure()
        fig4.savefig(plot_save_path + "/heatplot.png")
        fig4.clf()

    def export_data(self, csv_flag):
        print("Exporting Data")
        if csv_flag == 1:
            self.df_final.to_csv(self.csv_export, index=False)
            print("Exported to CSV Complete!")
        else:
            # Export large data by creating xlsxwriter first and setting archivezip64 option
            self.df_final.to_excel(self.excel_export, index=False, engine='xlsxwriter')
            print("Exported to Excel Complete!")


def usage():
    print(f"Usage: \n"
          f"-h | --help [ See usage for script ]\n"
          f"-f | --file [ Subtitle File ]\n"
          f"-m | --mode [ \"+\"/\"-\" ]\n"
          f"-t | --time [ Time in seconds to shift ]\n"
          f"\n"
          f"report-manager --file Engrish.srt --mode + --time 5\n")


def report_manager(argv):
    report = ReportManager()
    report_name = "Sample"
    report_title = "Sample_Title"
    pandas_profiling_export = "./test"
    csv_path = "./test/test.csv"
    plot_path = "./test"
    report_path = "./test"
    clean_csv_path = "./test"
    mode = "pandas-profiling"
    # Parse args
    try:
        opts, args = getopt.getopt(argv, "hf:m:t:", ["help", "file=", "mode=", "time="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-m", "--mode"):
            mode = arg
            if str(mode) != "pandas-profiling" and str(mode) != "custom" and str(mode) != "merge":
                usage()
                sys.exit(2)
        elif opt in ("-t", "--time"):
            time = arg

    if mode == "custom":
        # Custom Report
        print("Generating custom report")
        report.set_csv_path(csv_path)
        report.set_plot_path(plot_path)
        report.set_report_path(report_path)
        report.set_clean_csv_path(clean_csv_path)
        report.run_analysis()
    elif mode == "pandas-profiling":
        # Pandas Profiling
        print("Running Pandas Profiling")
        report.set_file(csv_path)
        report.load_dataframe()
        report.set_report_title(report_title)
        report.set_report_name(report_name)
        sample_flag = None  # Set to the sample size if you would like to do it on a sample instead.
        minimal_flag = False  # Quicker run if set to true, but not everything is captured.
        report.create_report(sample_flag, minimal_flag)
        report.set_save_directory(pandas_profiling_export)
        report.export_report()
    elif mode == "merge":
        print("Implement Merging Reports")


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    report_manager(sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    report_manager(sys.argv[1:])
