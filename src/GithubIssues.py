import os
import sys 
import re
import requests 
import pandas as pd
import numpy as np
import calendar
from sqlalchemy import create_engine
from pandas.io.json import json_normalize
from dateutil.relativedelta import relativedelta


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GithubIssuesUtils:
    
    def stacktrace(self) -> str:
        filename    = (re.findall("([^/]*)$", sys._getframe(  ).f_back.f_code.co_filename))[0]
        caller      = sys._getframe().f_back.f_code.co_name
        line_number = sys._getframe().f_back.f_lineno
        return f"{bcolors.BOLD}[{filename} : {caller} : {line_number}]{bcolors.ENDC}"


    def exception(self, e: str, exception_type: str) -> str:
        exception_string  = f"{bcolors.FAIL}\n======== {exception_type} EXCEPTION - START ========\n"
        exception_string += str(e)
        exception_string += f"\n========= {exception_type} EXCEPTION - END =========\n{bcolors.ENDC}"
        return exception_string



class GithubIssuesAPI:

    def __init__(self):
        self.__next_page_url = None
        self.__response      = None
        self.__next_page     = 1
        self.__last_page     = "[unknown]"
        self.__utils         = GithubIssuesUtils()


    def __get_next_page_url(self) -> str:
        return self.__next_page_url
        
    def __set_next_page_url(self, url: str):
        valid_url = re.match("https:\/\/api.github.com\/repositories\/\d+\/issues", url)
        #if not valid_url:
        #    print(f"{self.__utils.stacktrace()} URL = {url}") 
        #    print(f"{self.__utils.stacktrace()} ERROR invalid URL. Valid URLs begin with 'https://api.github.com/repositories/<NUMERIC-ID>/issues'") 
        #    exit()
        self.__next_page_url = url
        
    next_page_url = property(__get_next_page_url, __set_next_page_url)


    def __get_response(self) -> requests.Response:
        return self.__response
        
    def __set_response(self, response: requests.Response):
        self.__response = response

    response = property(__get_response, __set_response)

    
    def get_next_page(self):
        if self.__next_page_url is None:
            print(f"{self.__utils.stacktrace()} ERROR The 'next_page_url' property has not been set. Please set it to a valid URL.") 
            exit()
        
        print(f"{self.__utils.stacktrace()} INFO processing page {self.__next_page} of {self.__last_page} pages...")
            
        self.__response = requests.get(url = self.__next_page_url)
        
        print(f"{self.__utils.stacktrace()} INFO remaining API call allowance = {self.__get_remaining_calls()}.")

        # check for valid response
        if not self.__response.headers['Status'].startswith("200 "):
            self.__process_response_error()
            exit()

        [self.__next_page_url] = re.findall("<([^<]*)>;\s+rel=.next",         self.__response.headers['Link'])
        [self.__next_page]     = re.findall("<[^<]*page=(\d+)>;\s+rel=.next", self.__response.headers['Link'])
        [self.__last_page]     = re.findall("<[^<]*page=(\d+)>;\s+rel=.last", self.__response.headers['Link'])


    def __process_response_error(self):
        json = self.__response.json()
        if 'message' in json:
            msg = json['message']
            if msg.startswith("API rate limit exceeded"):
                print(f"{utils.stacktrace()} WARNING Github API rate limit exceeded. Exiting program.")
            elif  msg.startswith("Not Found"):
                print(f"{utils.stacktrace()} URL = {self.next_page_url}")
                print(f"{utils.stacktrace()} ERROR Endpoint not found. Exiting program.")
            else:
                print(f"{utils.stacktrace()} unknown error. Exiting program.")
            print(self.__utils.exception(json['message'], 'REQUESTS'))                


    def __get_remaining_calls(self) -> int:
        return int(self.__response.headers['X-RateLimit-Remaining'])




class GithubIssuesDB:
    
    
    def __init__(self, db_name, db_table="issues", echo=False):
        db_connection_url = "sqlite:///" + str(db_name) + ".db"
        self.engine = create_engine(db_connection_url, echo=echo)
        self.repository_id = 0
        self.db_file = str(db_name) + ".db"
        self.db = db_name
        self.allowable_tables = ['issues','status']
        self.table = 'issues'
        self.status = 0
        self.set_table(db_table)


    
    def set_table(self, table_name):
        if table_name in self.allowable_tables: 
            self.table = table_name
            self.status = 0
        else:
            self.status = 1
            print(f"{self.stacktrace()} ERROR {self.status} '{table_name}' table is not supported.")
            print(f"{self.stacktrace()} INFO supported tables = {self.allowable_tables}.")
        return self.status


    def stacktrace(self):
        filename    = (re.findall("([^/]*)$", sys._getframe(  ).f_back.f_code.co_filename))[0]
        caller      = sys._getframe().f_back.f_code.co_name
        line_number = sys._getframe().f_back.f_lineno
        return f"{bcolors.BOLD}[{filename} : {caller} : {line_number}]{bcolors.ENDC}"


    def print_exception(self, e, exception_type):
        exception_string  = f"{bcolors.FAIL}\n======== {exception_type} EXCEPTION - START ========\n"
        exception_string += str(e)
        exception_string += f"\n========= {exception_type} EXCEPTION - END =========\n{bcolors.ENDC}"
        print (exception_string)


    def get_table(self):
        return self.table


    def set_repository_id(self, repository_id):
        self.repository_id = repository_id

        
    def get_repository_id(self):
        return self.repository_id

    
    def get_repo_name(self):
        try:
            sql = '''
                        SELECT repo_name FROM %s
                        WHERE repo_id = ?
                        LIMIT 1
            ''' % (self.table,)
            result = pd.read_sql(sql, self.engine, params=(self.repository_id,))
            self.status = 0
            return result.iloc[0,0]
        except Exception as e:
            self.status = 2
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
            return 'unknown'


    def drop_table(self):
        try:
            sql = "DROP TABLE %s" % (self.table,)
            self.engine.execute(sql)
            self.status = 0
        except Exception as e:
            self.status = 3
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        return self.status


    def drop_db(self):
        if os.path.exists(self.db_file):
            try:
                os.remove(self.db_file)
                self.status = 0
            except Exception as e:
                self.status = 15
                print(f"{self.stacktrace()} ERROR {self.status} cannot delete '{self.db_file}' database file.")
                self.print_exception(e, 'OS')
        else:
            self.status = 16
            print(f"{self.stacktrace()} WARN {self.status} '{self.db_file}' does not exist. Ignoring.")
        return self.status


    def create_table(self):
        if self.table == 'issues':
            try:
                sql = '''
                        CREATE TABLE issues(id           INTEGER PRIMARY KEY UNIQUE,
                                            repo_id      INTEGER,
                                            repo_name    TEXT,
                                            type         TEXT,
                                            state        TEXT,
                                            opened_date  TEXT,
                                            opened_month INTEGER,
                                            opened_year  INTEGER,
                                            closed_date  TEXT,
                                            closed_month INTEGER,
                                            closed_year  INTEGER)
                        WITHOUT ROWID
                '''
                self.engine.execute(sql)
                self.status = 0
            except Exception as e:
                self.status = 4
                print(f"{self.stacktrace()} ERROR {self.status} unable to create '{self.table}' table.\n")
                self.print_exception(e, 'SQL')
        else:
            self.status = 5
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table is not supported.")
        return self.status


    def issue_exists(self, issue_id):
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE id = ?
            ''' % (self.table,)
            results = pd.read_sql(sql, self.engine, params=(issue_id,))
            self.status = 0
            if results.iloc[0][0] > 0:
                return True
            else:
                return False
        except Exception as e:
            self.status = 6
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')


    def insert_issue(self, issue):
        df = json_normalize(issue)

        # the repository name can be found at the end of the 'repository_url' data item
        [self.repository_name] = re.findall("/([^/]+)$", df.at[0,"repository_url"])

        df['repo_id'] = self.repository_id
        df['repo_name'] = self.repository_name
        
        # all pull requests contain a 'pull_request' JSON object with a 'url' key...
        if 'pull_request.url' in df.columns:
            df['type'] = "pull_request"
        # ...otherwise, they are assumed to be 'issues'
        else:
            df['type'] = "issue"

        # rename a few JSON fields to match SQL table schema
        df.rename(columns={'number'     : 'id',
                           'created_at' : 'opened_date',
                           'closed_at'  : 'closed_date'}, inplace=True)
    
        # add month & year number fields, derived from relevant JSON datetime fields
        df['opened_month'] = pd.DatetimeIndex(df['opened_date']).month
        df['opened_year']  = pd.DatetimeIndex(df['opened_date']).year
        df['closed_month'] = pd.DatetimeIndex(df['closed_date']).month
        df['closed_year']  = pd.DatetimeIndex(df['closed_date']).year

        # Define a subset of the full dataframe for direct SQL insertion 
        record = df[['id', 'state', 'repo_id', 'repo_name', 'type',
                     'opened_date', 'opened_month', 'opened_year',
                     'closed_date', 'closed_month', 'closed_year']]

        try:
            record.to_sql(self.table, con=self.engine, if_exists='append', index=False)    
            self.status = 0
        except Exception as e:
            self.status = 7
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table insertion failure.")
            self.print_exception(e, 'SQL')    
        return self.status


    def count_issues(self, issue_type, year, month):
        return_val_open   = 'unknown'
        return_val_closed = 'unknown'
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_open = result_open.iloc[0,0]
        except Exception as e:
            self.status = 8
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL') 
        try:           
            sql = '''
                        SELECT Count(*) FROM issues
                        WHERE repo_id = ?
                        AND type = ?
                        AND closed_month = ?
                        AND closed_year = ?
            '''
            result_closed = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_closed = result_closed.iloc[0,0]
        except Exception as e:
            self.status = 9
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')

        return [return_val_open, return_val_closed]


    def get_start_date(self, issue_type):
        try:
            sql = '''
                        SELECT opened_date FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        ORDER BY opened_date
                        LIMIT 1
            ''' % (self.table,)
            result = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type))
            self.status = 0
            return result.iloc[0,0]
        except Exception as e:
            self.status = 10
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
            return "unknown"


    def get_end_date(self, issue_type):
        try:
            sql = '''
                        SELECT opened_date FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        ORDER BY opened_date DESC
                        LIMIT 1
            ''' % (self.table,)
            result = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type))
            self.status = 0
            return result.iloc[0,0]
        except Exception as e:
            self.status = 11
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
            return "unknown"


    def get_issue_ages(self, issue_type, month_end_date):
        try:
            sql = '''
                    SELECT opened_date FROM %s
                    WHERE repo_id = ?
                    AND type = ?
                    AND date(opened_date) < ?                                
                    AND (closed_date = "null" OR date(closed_date) > ?)
                    ORDER BY opened_date
            ''' % (self.table,)
            result = pd.DataFrame( pd.read_sql(sql,
                                               self.engine,
                                               parse_dates=['opened_date'],
                                               params=(self.repository_id, issue_type, month_end_date, month_end_date)) )
            self.status = 0
            return result.applymap(lambda x: (pd.to_datetime(month_end_date) - x).days)
        except Exception as e:
            self.status = 12
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
            return "unknown"
        
    
    def show_issues(self, issue_type):
        try:
            sql =   '''
                        SELECT  id, repo_name, type, state,
                                opened_month, opened_year,
                                closed_month, closed_year FROM %s
                        WHERE repo_id = ?
                        AND type = ?
            ''' % (self.table,)
            result = pd.DataFrame( pd.read_sql(sql,
                                               self.engine,
                                               params=(self.repository_id, issue_type)) )
            self.status = 0
            print(result)
        except Exception as e:
            self.status = 13
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
            
        return self.status


    def count_records(self, issue_type):
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE type = ?
            ''' % (self.table,)
            result = pd.read_sql(sql, self.engine, params=(issue_type,))
            self.status = 0
        except Exception as e:
            self.status = 14
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
        print(result.iloc[0,0])
        return result.iloc[0,0]



    def show_statistics(self, date_range):
        print(f"           \t          \t       ISSUES       \t PULL REQUESTS   ")
        print(f"REPOSITORY  \t  YYYY-MM  \t OPENED \t CLOSED \t OPENED \t CLOSED")
        print("=================================================================")
        repository_name = self.get_repo_name()
        for idx in date_range:
            [opened_issue_count, closed_issue_count] = self.count_issues('issue',
                                                                         idx.year, idx.month)
            [opened_pr_count, closed_pr_count]       = self.count_issues('pull_request',
                                                                         idx.year, idx.month)
            print("%10s  \t  %4d-%02d  \t  %4d  \t  %4d  \t %4d  \t  %4d"
                    % (repository_name, idx.year, idx.month,
                       opened_issue_count, closed_issue_count,
                       opened_pr_count, closed_pr_count))


class GithubIssuesData:

    
    def __init__(self):
        self.__status                  = 0
        self.__timespan_months         = 12
        self.utils                     = GithubIssuesUtils()
        self.__window_start_idx        = 0
        self.__window_end_idx          = self.__timespan_months
        self.init_arrays()


    def init_arrays(self, date_range: pd.DatetimeIndex = None):
        if date_range is None:
            length = self.__timespan_months
        else:
            length = date_range.size
        self.__plot_points             = np.arange(0,length)
        self.__opened_issue_counts     = np.zeros(length, dtype=int)
        self.__closed_issue_counts     = np.zeros(length, dtype=int)
        self.__total_open_issue_counts = np.zeros(length, dtype=int)
        self.set_month_labels(date_range, True)
        self.__issue_ages              = []
        

    def set_plot_window(self, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex) -> list:
        self.__window_start_idx = np.where(self.__month_labels ==
                                    calendar.month_abbr[start_date.month] + '-' + str(end_date.year)[2:])[0][0]
        self.__window_end_idx   = np.where(self.__month_labels ==
                                    calendar.month_abbr[end_date.month] + '-' + str(end_date.year)[2:])[0][0]
        return [self.__window_start_idx, self.__window_end_idx]

    def __get_plot_window(self) -> list:
        return [self.__window_start_idx, self.__window_end_idx]
        
    def __set_plot_window(self, window: list):
        if len(window) != 2:
            print(f"{self.stacktrace()} ERROR plot_window requires a 2 element list. you have supplied a len(window) element list.\n")
        self.__window_start_idx = window[0]
        self.__window_start_idx = window[1]

    plot_window = property(__get_plot_window, __set_plot_window)


    def __get_timespan_months(self) -> int:
        return self.__timespan_months
        
    def __set_timespan_months(self, months: int):
        self.__timespan_months = months

    timespan_months = property(__get_timespan_months, __set_timespan_months)


    def __get_plot_points(self) -> np.ndarray:
        return self.__plot_points
        
    def __set_plot_points(self, plot_points: np.ndarray):
        self.__plot_points = np.copy(plot_points)
        
    plot_points = property(__get_plot_points, __set_plot_points)


    def __get_month_labels(self) -> np.ndarray:
        return self.__month_labels

    def __set_month_labels(self, labels: np.ndarray):
        self.__month_labels = np.copy(labels)

    def set_month_labels(self, date_range: pd.DatetimeIndex = None, inclusive: bool = False):
        if date_range is None:
            length = self.__timespan_months
            ref_date = pd.to_datetime('now')
        else:
            length = date_range.size
            ref_date = date_range[-1]
        x = int(inclusive)
        months = np.array([ (ref_date - relativedelta(months=length-(i+x))).month
                            for i in range(length) ])
        years  = np.array([ str((ref_date - relativedelta(months=length-(i+x))).year)[2:]
                            for i in range(length) ])
        labels = np.array([ calendar.month_abbr[month] + '-' + year
                            for month,year in zip(months,years) ])
        labels[0] = ''
        self.__month_labels = np.copy(labels)
        
    month_labels = property(__get_month_labels, __set_month_labels)


    def __get_opened_issue_counts(self) -> np.ndarray:
        return self.__opened_issue_counts
        
    def __set_opened_issue_counts(self, counts: np.ndarray):
        self.__opened_issue_counts = np.copy(counts)

    opened_issue_counts = property(__get_opened_issue_counts, __set_opened_issue_counts)


    def __get_closed_issue_counts(self) -> np.ndarray:
        return self.__closed_issue_counts
        
    def __set_closed_issue_counts(self, counts: np.ndarray):
        self.__closed_issue_counts = np.copy(counts)

    closed_issue_counts = property(__get_closed_issue_counts, __set_closed_issue_counts)


    def __get_total_open_issue_counts(self) -> np.ndarray:
        return self.__total_open_issue_counts
        
    def __set_total_open_issue_counts(self, counts: np.ndarray):
        self.__total_open_issue_counts = np.copy(counts)

    total_open_issue_counts = property(__get_total_open_issue_counts, __set_total_open_issue_counts)


    def __get_issue_ages(self) -> list:
        return self.__issue_ages # [-(self.__timespan_months+1):]
        
    def __set_issue_ages(self, ages: list):
        self.__issue_ages = ages.copy()
    
    def issue_ages_append(self, ages: np.ndarray):
        self.__issue_ages.append(ages.copy())

    issue_ages = property(__get_issue_ages, __set_issue_ages)

