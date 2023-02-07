import os
import sys
import argparse
import re
import requests 
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.sql.expression import update
from pandas import json_normalize
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin

# required here for debug info reporting only
import subprocess
import platform
import struct
import locale
import sqlalchemy
import matplotlib


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
    
    def __init__(self):
        self.__status = 0
        self.__data_path = str(os.path.dirname(sys._getframe().f_code.co_filename)) + '/../data'


    def __get_data_path(self) -> str:
        return self.__data_path

    def __set_data_path(self, path : str):
        self.__data_path = path

    data_path = property(__get_data_path, __set_data_path)


    def __get_args(self) -> dict:
        return {'auth_token'     : self.__auth_token,
                'fqdn'           : self.__fqdn,
                'https'          : self.__https,
                'account'        : self.__account,
                'repo'           : self.__repo,
                'issue_type'     : self.__issue_type,
                'timespan'       : self.__timespan,
                'ref_date'       : self.__ref_date,
                'first_page'     : self.__first_page,
                'page_count'     : self.__page_count,
                'offset_month'   : self.__offset_month,
                'save_file'      : self.__save_file,
                'label_file'     : self.__label_file,
                'out_label_file' : self.__out_label_file,
                'num_labels'     : self.__num_labels,
                'info'           : self.__info
                }


    def __set_args(self, args : dict):
        for arg in args:
            # obtain 'mangled' version of private attribute name
            private_attr = '_GithubIssuesUtils__' + str(arg)
            setattr(self, private_attr, args[arg])

    args = property(__get_args, __set_args)


    def process_args(self) -> dict:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('-a', '--authtoken', type=str, default='',
                            help="Github personal access token (default='')")
        arg_parser.add_argument('-fqdn', '--fqdn', type=str, default='api.github.com',
                            help="fully qualified domain name of github server (default='api.github.com')")
        arg_parser.add_argument('-https', '--https', action='store_true',
                            help="use https")        
        arg_parser.add_argument('-u', '--user', type=str, default='matplotlib',
                            help="Github user name (default='matplotlib')")
        arg_parser.add_argument('-r', '--repo', type=str, default='matplotlib',
                            help="Github repo name (default='matplotlib')")
        arg_parser.add_argument('-t', '--type', type=str, choices=['issue','pr'], default='issue',
                            help="Issue type, one of ['issue','pr'] (default='issue')")
        arg_parser.add_argument('-m', '--months', type=int, default=12,
                            help="analysis timespan in months (default=12)")
        arg_parser.add_argument('-d', '--refdate', type=pd.Timestamp, default=datetime.now(),
                            help="reference date ('YYYY-MM', default=now)")
        arg_parser.add_argument('-f', '--firstpage', type=int, default=1,
                            help="first page of issues to request (default=1)")
        arg_parser.add_argument('-c', '--pagecount', type=int, default=10,
                            help="number of pages to request (default=10)")
        arg_parser.add_argument('-o', '--offsetmonth', type=str, choices=['opened','closed'], default="closed",
                            help="which month to offset issue closure in (default=closed")
        arg_parser.add_argument('-p', '--datapath', type=str, default=os.getcwd(),
                            help="path for SQLite db file (default=pwd)")
        arg_parser.add_argument('-s', '--savefile', type=str, nargs='?', action='store', const='metrics.png', default=None,
                            help="save result to a file (optional file name, default = metrics.png)")
        arg_parser.add_argument('-il', '--inlabfile', type=str, nargs='?', action='store', const='labels.csv', default=None,
                            help="read issue labels & groups from a CSV file (optional file name, default = labels.csv")
        arg_parser.add_argument('-ol', '--outlabfile', type=str, nargs='?', action='store', const='outlabels.csv', default=None,
                            help="write issue labels & groups to a CSV file (optional file name, default = outlabels.csv")
        arg_parser.add_argument('-nl', '--numlabs', type=int, default=12,
                            help="set number of labels to display in the top N frequency analysis display (default = 12)")
        arg_parser.add_argument('-i', '--info', action='store_true',
                            help="write environment info to debug_info.txt file for Github bug reporting")
        args = arg_parser.parse_args()        
        self.__auth_token = vars(args)['authtoken']
        self.__fqdn = vars(args)['fqdn']
        self.__https = vars(args)['https']
        self.__account = vars(args)['user']
        self.__repo = vars(args)['repo']
        self.__issue_type = vars(args)['type']
        self.__timespan = vars(args)['months']
        self.__ref_date = vars(args)['refdate']
        self.__first_page = vars(args)['firstpage']
        self.__page_count = vars(args)['pagecount']
        self.__offset_month = vars(args)['offsetmonth']
        self.__save_file = vars(args)['savefile']
        self.__label_file = vars(args)['inlabfile']
        self.__out_label_file = vars(args)['outlabfile']
        self.__num_labels = vars(args)['numlabs']
        self.__data_path = vars(args)['datapath']
        self.__info = vars(args)['info']
        # if -i/--info arg is supplied, write debug info out to a file and quit 
        if self.__info == True:
            self.write_debug_info()
            exit()
        return self.args


    def write_debug_info(self):
        """
    Class: GithubIssuesUtils
    Method: write_debug_info()
    Argument(s): None
    Return Value(s): None
    Output(s): a file (debug_info.txt)
    Interrogates and writes system, python, module & commit information to a file. 
    When reporting RepoDash bugs, run this function to obtain a detailed report 
    that you should provide to the library maintainer when submitting your github 
    issue. Please run this from within the top level folder for your cloned repo.
        """
        debug_info = []

        # get short commit hash
        commit = "unknown - please run this from within the top level folder for the repo"
        if os.path.isdir(".git"):
            try:
                pipe = subprocess.Popen(
                    'git rev-parse --short HEAD'.split(" "),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                so, serr = pipe.communicate()
            except (OSError, ValueError):
                pass
            else:
                if pipe.returncode == 0:
                    commit = so.decode('utf-8').strip().strip('"')

        debug_info.append(("commit", commit))

        try:
            (sysname, nodename, release, version, machine, processor) = platform.uname()

            # local timezone discovery only tested for Windows & Linux
            # if system is neither, then leave as "unknown"
            if sysname == 'Windows': 
                local_tz = subprocess.check_output(["tzutil", "/g"], shell=True)
                local_tz = local_tz.decode('utf-8').strip().strip('"')
            elif sysname == "Linux": 
                local_tz = subprocess.check_output(["date", "+%Z"])
                local_tz = local_tz.decode('utf-8').strip().strip('"')
            else:
                local_tz = "unknown"

            debug_info.extend(
                [
                    ('local timezone', f"{local_tz}"),
                    ('command line', ' '.join(map(str, sys.argv))),
                    ("python", ".".join(map(str, sys.version_info))),
                    ("python-bits", struct.calcsize("P") * 8),
                    ("OS", f"{sysname}"),
                    ("OS-release", f"{release}"),
                    ("Version", "{version}".format(version=version)),
                    ("machine", f"{machine}"),
                    ("processor", f"{processor}"),
                    ("byteorder", f"{sys.byteorder}"),
                    ("LC_ALL", f"{os.environ.get('LC_ALL', 'None')}"),
                    ("LANG", f"{os.environ.get('LANG', 'None')}"),
                    ("LOCALE", ".".join(map(str, locale.getlocale()))),
                    ('requests' , requests.__version__),
                    ('pandas' , pd.__version__),
                    ('sqlalchemy' , sqlalchemy.__version__),
                    ('numpy' , np.__version__),
                    ('matplotlib' , matplotlib.__version__),
                ]
            )
            f = open("debug_info.txt", "w")
            f.write("DEBUG INFO\n")
            f.write("----------\n")
            for item in debug_info:
                f.write(f"{item[0]}: {item[1]}\n")
            f.close()
        except (KeyError, ValueError):
            pass


    def get_plot_monthly_span(self, length_constraint = None):
        print('TIMESPAN' , self.__timespan , 'LENGTHCONSTRAINT' , length_constraint)
        plot_end_month = pd.Period(self.__ref_date, 'M') + 1
        offset = (self.__timespan
                  if length_constraint is None
                  else min(self.__timespan, length_constraint))
        plot_start_month = plot_end_month - offset
        return pd.date_range(start=pd.Period.to_timestamp(plot_start_month),
                             end=pd.Period.to_timestamp(plot_end_month),
                             closed=None, freq='M')
    

    def stacktrace(self) -> str:
        filename    = (re.findall("([^/]*)$", sys._getframe().f_back.f_code.co_filename))[0]
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
        self.__status        = 0
        self.__labels_url    = None
        self.__next_page_url = None
        self.__response      = None
        self.__next_page     = 0
        self.__last_page     = "[unknown]"
        self.__utils         = GithubIssuesUtils()


    def __get_status(self) -> int:
        return self.__status
        
    def __set_status(self, status_value: int):
        self.__status = status_value

    status = property(__get_status, __set_status)


    def __get_response(self) -> requests.Response:
        return self.__response
        
    def __set_response(self, response: requests.Response):
        self.__response = response

    response = property(__get_response, __set_response)


    def set_seed_url(self, args : dict, page_type='issues'):
        
        self.__last_page     = "[unknown]"
        self.__status = 0
        
        endpoint_url = (args['fqdn'] if 'fqdn' in args else self.__utils.args['fqdn'])
        if endpoint_url != "api.github.com":
            endpoint_url += "/api/v3"
        use_https = (args['https'] if 'https' in args else self.__utils.args['https'])
        protocol = ('https' if use_https else 'http')
        account = (args['account'] if 'account' in args else self.__utils.args['account'])
        repo = (args['repo'] if 'repo' in args else self.__utils.args['repo'])
        
        if page_type == "issues":
            page = (args['first_page'] if 'first_page' in args else self.__utils.args['first_page'])
            self.__next_page_url = f"{protocol}://{endpoint_url}/repos/{account}/{repo}/issues?state=all&direction=asc&per_page=100&page={page}"
        elif page_type == "labels":
            page = 1
            self.__next_page_url = f"{protocol}://{endpoint_url}/repos/{account}/{repo}/labels?page={page}"
        self.__next_page = page

    
    def get_next_page(self, args : dict):
        if self.__next_page_url is None:
            print(f"{self.__utils.stacktrace()} ERROR The 'next_page_url' property has not been set. Please set it to a valid URL.") 
            exit()
        
        print(f"{self.__utils.stacktrace()} INFO processing page {self.__next_page} of {self.__last_page} pages...")
        
        headers = ({"Authorization": f"token {args['auth_token']}"} if args['auth_token'] != '' else {})

        try:
            self.__response = requests.get(url = self.__next_page_url, headers = headers, timeout=3)
            self.__response.raise_for_status()
        except requests.exceptions.HTTPError:
            print (f"{self.__utils.stacktrace()} ERROR HTTP {self.__response.status_code} response code for {self.__next_page_url}")
            raise SystemExit()
        except requests.exceptions.ConnectionError:
            print (f"{self.__utils.stacktrace()} ERROR Connection error for {self.__next_page_url}")
            raise SystemExit()
        except requests.exceptions.Timeout:
            print(f"{self.__utils.stacktrace()} ERROR Timeout for {self.__next_page_url}")
            raise SystemExit()
        except requests.exceptions.TooManyRedirects:
            print(f"{self.__utils.stacktrace()} ERROR Too many redirects for {self.__next_page_url}")
            raise SystemExit()
        except requests.exceptions.RequestException:
            print(f"{self.__utils.stacktrace()} ERROR General request exception for {self.__next_page_url}")
            raise SystemExit()
            
        print(f"{self.__utils.stacktrace()} INFO remaining API call allowance = {self.__get_remaining_calls()}.")

        # check for valid response
        if not self.__response.status_code == 200:
            self.__status = 201
            self.__process_response_error()
            exit()

        # update page references from header information
        if 'Link' in self.__response.headers:
            if self.__next_page != self.__last_page:
                [self.__next_page_url] = re.findall("<([^<]*)>;\s+rel=.next",         self.__response.headers['Link'])
                [self.__next_page]     = re.findall("<[^<]*page=(\d+)>;\s+rel=.next", self.__response.headers['Link'])
                [self.__last_page]     = re.findall("<[^<]*page=(\d+)>;\s+rel=.last", self.__response.headers['Link'])
            else:
                self.__status = 202
                print(f"{self.__utils.stacktrace()} WARN reached last page of the issues list.")
                

    def __process_response_error(self):
        json = self.__response.json()
        if 'message' in json:
            msg = json['message']
            if msg.startswith("API rate limit exceeded"):
                print(f"{self.__utils.stacktrace()} WARNING Github API rate limit exceeded. Exiting program.")
            elif  msg.startswith("Not Found"):
                print(f"{self.__utils.stacktrace()} URL = {self.__next_page_url}")
                print(f"{self.__utils.stacktrace()} ERROR Endpoint not found. Exiting program.")
            else:
                print(f"{self.__utils.stacktrace()} unknown error. Exiting program.")
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
        self.allowable_tables = { 'issues' : 'issues',
                                  'labels' : 'issue_labels' }
        self.table = 'issues'
        self.status = 0
        self.__set_table(db_table)
        self.__reset_tables()

    def __reset_tables(self):
        '''
    ~~~ OBJECT-INTERNAL METHOD ~~~
    Class: GithubIssuesDB
    Method: __reset_tables()
    Argument(s): None
    Return Value(s): None
    Drops and re-creates empty issues & labels database tables. 
    This must be done with every run in this version of RepoDash. However, in future versions I 
    may implement the ability to run in update mode by specifying an existing issues database file. 
        '''
        try:
            for table_name in self.allowable_tables:
                self.drop_table(table_name)
                self.create_table(table_name)
            self.status = 0
        except Exception as e:
            self.status = 99
            print(f"{self.stacktrace()} ERROR {self.status} failed to reset db tables.")
            self.print_exception(e, 'SQL')

    def __set_table(self, table_name):
        if table_name in list(self.allowable_tables.values()):
            self.table = table_name
            self.status = 0
        else:
            self.status = 1
            print(f"{self.stacktrace()} ERROR {self.status} '{table_name}' table is not supported.")
            print(f"{self.stacktrace()} WARN {self.status} current table ('{self.table}') remains unchanged.")
            print(f"{self.stacktrace()} INFO supported tables = {list(self.allowable_tables.values())}.")
            exit()
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
        self.__set_table(self.allowable_tables['issues']) 
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


    def drop_table(self,type):
        status = self.__set_table(self.allowable_tables[type])
        print(f"{self.stacktrace()} INFO dropping '{self.table}' table...")
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
        print(f"{self.stacktrace()} INFO deleting '{self.db_file}' database file...")
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


    def create_table(self, type):
        self.__set_table(self.allowable_tables[type]) 
        if self.table == 'issues':
            try:
                sql = '''
                        CREATE TABLE issues(id            INTEGER PRIMARY KEY UNIQUE,
                                            repo_id       INTEGER,
                                            repo_name     TEXT,
                                            type          TEXT,
                                            label_count   INTEGER,
                                            labelstring   TEXT,
                                            assign_count  INTEGER,
                                            assignstring  TEXT,
                                            state         TEXT,
                                            opened_date   TEXT,
                                            opened_month  INTEGER,
                                            opened_year   INTEGER,
                                            closed_date   TEXT,
                                            closed_month  INTEGER,
                                            closed_year   INTEGER)
                        WITHOUT ROWID
                '''
                self.engine.execute(sql)
                self.status = 0
            except Exception as e:
                self.status = 4
                print(f"{self.stacktrace()} ERROR {self.status} unable to create '{self.table}' table.\n")
                self.print_exception(e, 'SQL')
        elif self.table == 'issue_labels':
            try:
                sql = '''
                        CREATE TABLE issue_labels(id            INTEGER PRIMARY KEY,
                                                  account       TEXT,
                                                  repo_name     TEXT,
                                                  label         TEXT,
                                                  label_group   TEXT,
                                                  description   TEXT)
                        WITHOUT ROWID
                '''
                self.engine.execute(sql)
                self.status = 0
            except Exception as e:
                self.status = 29
                print(f"{self.stacktrace()} ERROR {self.status} unable to create '{self.table}' table.\n")
                self.print_exception(e, 'SQL')
        else:
            self.status = 5
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table is not supported.")
        return self.status


    def issue_exists(self, issue_id):
        self.__set_table(self.allowable_tables['issues']) 
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


    def insert_label(self, label):
        self.__set_table(self.allowable_tables['labels']) 
        df = json_normalize(label)

        # rename a few JSON fields to match SQL table schema
        df.rename(columns={'name' : 'label'}, inplace=True)

        # the repository name can be found at the end of the 'repository_url' data item
        [self.account] = re.findall("/([^/]+)/[^/]+/labels", df.at[0,"url"])
        [self.repository_name] = re.findall("/([^/]+)/labels", df.at[0,"url"])

        df['account'] = self.account
        df['repo_name'] = self.repository_name
        df['label_group'] = df['label']

        # Define a subset of the full dataframe for direct SQL insertion 
        record = df[['id', 'account', 'repo_name', 'label', 'label_group', 'description']]

        try:
            record.to_sql(self.table, con=self.engine, if_exists='append', index=False)    
            self.status = 0
        except Exception as e:
            self.status = 30
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table insertion failure.")
            self.print_exception(e, 'SQL')
        return self.status


    def insert_issue(self, issue):
        self.__set_table(self.allowable_tables['issues']) 
        df = json_normalize(issue)

        assignstring = []
        for label in df['assignees'][0]:
            assignstring.append(str(label['login']))
        df['assign_count'] = len(assignstring)
        df['assignstring'] = ':'.join(assignstring)

        labelstring = []
        for label in df['labels'][0]:
            labelstring.append(str(label['name']))
        df['label_count'] = len(labelstring)
        df['labelstring'] = '::'.join(labelstring)

        # the repository name can be found at the end of the 'repository_url' data item
        [self.repository_name] = re.findall("/([^/]+)$", df.at[0,"repository_url"])

        df['repo_id'] = self.repository_id
        df['repo_name'] = self.repository_name
        
        # all pull requests contain a 'pull_request' JSON object with a 'url' key...
        if 'pull_request.url' in df.columns:
            df['type'] = "pr"
        # ...otherwise, they are assumed to be 'issues'
        else:
            df['type'] = "issue"

        # rename a few JSON fields to match SQL table schema
        df.rename(columns={'number'     : 'id',
                           'created_at' : 'opened_date',
                           'closed_at'  : 'closed_date'}, inplace=True)
 
        # remove time & timezone info from datestamp
        df['opened_date'] = pd.DatetimeIndex(df['opened_date']).date
        df['closed_date'] = pd.DatetimeIndex(df['closed_date']).date
 
        # add month & year number fields, derived from relevant JSON datetime fields
        df['opened_month'] = pd.DatetimeIndex(df['opened_date']).month
        df['opened_year']  = pd.DatetimeIndex(df['opened_date']).year
        df['closed_month'] = pd.DatetimeIndex(df['closed_date']).month
        df['closed_year']  = pd.DatetimeIndex(df['closed_date']).year

        # Define a subset of the full dataframe for direct SQL insertion 
        record = df[['id', 'state', 'repo_id', 'repo_name', 'type',
                     'label_count', 'labelstring',
                     'assign_count', 'assignstring',
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


    def get_labels(self):
        self.__set_table(self.allowable_tables['labels'])
        table = pd.DataFrame()
        try:
            # read all label records from db table
            select_sql = "SELECT * FROM %s" % (self.table,)
            table = pd.read_sql(select_sql, self.engine)
            self.status = 0
        except Exception as e:
            self.status = 31
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' SQL retrieval failed.\n")
            self.print_exception(e, 'SQL')
        return table


    def update_labels(self, df):
        self.__set_table(self.allowable_tables['labels']) 
        try:
            table = self.get_labels()
            # update label_group values from caller supplied mappings
            for index, row in df.iterrows():
                table.loc[table['label'] == row['label'], 'label_group'] = row['group']
            # replace all (updated) label records in db
            table.to_sql(self.table, self.engine, if_exists='replace', index=False)
            self.status = 0
        except Exception as e:
            self.status = 31
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' SQL update failed.\n")
            self.print_exception(e, 'SQL')
        return self.status


    def get_label_group(self, label):
        self.__set_table(self.allowable_tables['labels']) 
        try:
            sql = '''
                        SELECT label_group FROM %s
                        WHERE account = ?
                        AND repo_name = ?
                        AND label = ?
            ''' % (self.table,)
            results = pd.read_sql(sql, self.engine, params=(self.account, self.repository_name, str(label)))
            self.status = 0
            return results.iloc[0][0]
        except Exception as e:
            self.status = 32
            print(f"{self.stacktrace()} ERROR {self.status} issue label '{label}' does not exist.\n")
            self.print_exception(e, 'SQL')
            return "Unknown Type"


    def count_issues(self, issue_type, year, month):
        '''
    Class: GithubIssuesDB
    Method: count_issues()
    Argument(s):
        issue_type      string  ('issue' | 'pr')
        year            integer (DateTimeIndex.year format)
        month           integer (DateTimeIndex.month format)
    Return Value(s):
        [opened,closed] list(numpy.int32)
    Count the number of issues both opened and closed in a given month
        '''
        self.__set_table(self.allowable_tables['issues']) 
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
    

    def count_labels_point_in_time(self, issue_type, point_in_time):
        '''
    Class: GithubIssuesDB
    Method: count_labels_point_in_time()
    Argument(s):
        issue_type      string  ('issue' | 'pr')
        point_in_time   string  (YYYY-MM-DD format)
    Return Value(s):
        open_counts     Dataframe(string, numpy.int32)
    Count the use of labels for issues opened prior to and which remain open 
    at the stated point in time (e.g. month end)
        '''
        self.__set_table(self.allowable_tables['issues']) 
        open_labels = {}
        try:
            sql = '''
                    SELECT labelstring FROM %s
                    WHERE repo_id = ?
                    AND type = ?
                    AND date(opened_date) <= ?                                
                    AND (closed_date is NULL OR date(closed_date) > ?)
                    ORDER BY opened_date
            ''' % (self.table,)
            result = pd.read_sql(sql,
                                 self.engine,
                                 params=(self.repository_id, issue_type, point_in_time, point_in_time))
            for index, row in result.iterrows():
                for label in row[0].split('::'):
                    if label != '':
                        open_labels[label] = (1 if label not in open_labels else open_labels[label] + 1)
            self.status = 0
        except Exception as e:
            self.status = 26
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        return pd.DataFrame.from_dict(open_labels, orient='index', columns=['open_count'])


    def count_labels_monthly(self, issue_type, year, month):
        self.__set_table(self.allowable_tables['issues']) 
        opened_labels = {}
        open_labels = {}
        closed_labels = {}
        # count issues opened during the month
        try:
            sql = '''
                        SELECT labelstring FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
            ''' % (self.table,)
            result_opened = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            for index, row in result_opened.iterrows():
                for label in row[0].split('::'):
                    if label != '':
                        opened_labels[label] = (1 if label not in opened_labels else opened_labels[label] + 1)
            self.status = 0
        except Exception as e:
            self.status = 26
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        #count labels used for issues that were opened during the month and are currently still open
        try:
            sql = '''
                        SELECT labelstring FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND state = 'open'
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            for index, row in result_open.iterrows():
                for label in row[0].split('::'):
                    if label != '':
                        open_labels[label] = (1 if label not in open_labels else open_labels[label] + 1)
            self.status = 0
        except Exception as e:
            self.status = 27
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        # count labels used for issues that were closed during the month
        try:           
            sql = '''
                        SELECT labelstring FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND closed_month = ?
                        AND closed_year = ?
            ''' % (self.table,)
            result_closed = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            for index, row in result_closed.iterrows():
                for label in row[0].split('::'):
                    if label != '':
                        closed_labels[label] = (1 if label not in closed_labels else closed_labels[label] + 1)
            self.status = 0
        except Exception as e:
            self.status = 28
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')

        df_opened = pd.DataFrame.from_dict(opened_labels, orient='index', columns=['opened_count'])
        df_open = pd.DataFrame.from_dict(open_labels, orient='index', columns=['open_count'])
        df_closed = pd.DataFrame.from_dict(closed_labels, orient='index', columns=['closed_count'])

        return pd.concat([df_opened, df_open, df_closed], axis=1, sort=True, join='outer').fillna(0)

    def deprecated_count_labels(self, issue_type, year, month, label):
        self.__set_table(self.allowable_tables['issues']) 
        return_val_open   = 'unknown'
        return_val_closed = 'unknown'
        try:
            sql = '''
                        SELECT Count(*), INSTR(labelstring, "%s") as match FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND match > 0
            ''' % (label, self.table)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_open = result_open.iloc[0,0]
        except Exception as e:
            self.status = 17
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL') 
        try:           
            sql = '''
                        SELECT Count(*), INSTR(labelstring, "%s") as match FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND closed_month = ?
                        AND closed_year = ?
                        AND match > 0
            ''' % (label, self.table)
            result_closed = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_closed = result_closed.iloc[0,0]
        except Exception as e:
            self.status = 18
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        return [return_val_open, return_val_closed]


    def count_monthly_open(self, issue_type, year, month):
        '''
    Class: GithubIssuesDB
    Method: count_monthly_open()
    Argument(s):
        issue_type  string ('issue', 'pr')
        year        integer (DateTimeIndex.year format)
        month       integer (DateTimeIndex.month format)
    Return Value(s):
        count       numpy.int32
    Count the number of issues opened in a given month which remain open today
        '''
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND state = 'open'
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            count = result_open.iloc[0,0]
            self.status = 0
            return count
        except Exception as e:
            self.status = 21
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')


    def count_monthly_unlabelled(self, issue_type, year, month):
        '''
    Class: GithubIssuesDB
    Method: count_monthly_unlabelled()
    Argument(s):
        issue_type      string  ('issue' | 'pr')
        year            integer (DateTimeIndex.year format)
        month           integer (DateTimeIndex.month format)
    Return Value(s):
        count           numpy.int32
    Count the number of issues opened in a given month which remain
    open today and are also unlabelled
        '''
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND state = 'open'
                        AND label_count = 0
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            count = result_open.iloc[0,0]
            self.status = 0
            return count
        except Exception as e:
            self.status = 22
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')


    def count_monthly_unassigned(self, issue_type, year, month):
        '''
    Class: GithubIssuesDB
    Method: count_monthly_unassigned()
    Argument(s):
        issue_type      string  ('issue' | 'pr')
        year            integer (DateTimeIndex.year format)
        month           integer (DateTimeIndex.month format)
    Return Value(s):
        count           numpy.int32
    Count the number of issues opened in a given month which remain
    open today and which also have yet to be assigned to someone
        '''
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND state = 'open'
                        AND assign_count = 0
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            count = result_open.iloc[0,0]
            self.status = 0
            return count
        except Exception as e:
            self.status = 25
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')


    def count_labelled(self, issue_type, year, month):
        '''
    Class: GithubIssuesDB
    Method: count_labelled()
    Argument(s):
        issue_type      string  ('issue' | 'pr')
        year            integer (DateTimeIndex.year format)
        month           integer (DateTimeIndex.month format)
    Return Value(s):
        [opened,closed] list(numpy.int32)
    Count the number of issues both opened and closed in a given month
    which have been assigned labels
        '''
        self.__set_table(self.allowable_tables['issues']) 
        return_val_open   = 'unknown'
        return_val_closed = 'unknown'
        try:
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND opened_month = ?
                        AND opened_year = ?
                        AND label_count > 0
            ''' % (self.table,)
            result_open = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_open = result_open.iloc[0,0]
        except Exception as e:
            self.status = 23
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL') 
        try:           
            sql = '''
                        SELECT Count(*) FROM %s
                        WHERE repo_id = ?
                        AND type = ?
                        AND closed_month = ?
                        AND closed_year = ?
                        AND label_count > 0
            ''' % (self.table,)
            result_closed = pd.read_sql(sql, self.engine, params=(self.repository_id, issue_type, month, year))
            self.status = 0
            return_val_closed = result_closed.iloc[0,0]
        except Exception as e:
            self.status = 24
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.")
            self.print_exception(e, 'SQL')
        return [return_val_open, return_val_closed]
    

    def __get_monthly_span(self):
        [ start_month, end_month ] = self.get_month_range()
        print(f"{self.stacktrace()} INFO data span is start=" + str(start_month) + " , end=" + str(end_month))
        date_range = pd.date_range(start=start_month,
                                   end=pd.Period.to_timestamp(pd.Period(end_month, 'M') + 1),
                                   closed=None, freq='M')
        if date_range.size == 0:
            self.status = 19
            print(f"{self.stacktrace()} ERROR {self.status} record date range is empty")
        elif date_range.size == 1:
            self.status = 20
            print(f"{self.stacktrace()} ERROR {self.status} record date range only 1 month long. Minimum of 2 months history required")
            print(f"{self.stacktrace()} INFO record date range = {date_range}")
        else:
            self.status = 0
            return date_range
        exit()
    
    monthly_span = property(__get_monthly_span, None)


    def get_month_range(self):
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql = '''
                        SELECT MIN(opened_date), MAX(opened_date) FROM %s
                        WHERE repo_id = ?
                        ORDER BY opened_date
                        LIMIT 1
            ''' % (self.table,)
            result = pd.read_sql(sql, self.engine, params=(self.repository_id,))
            self.status = 0
            if result.empty:
                self.status = 10
                print(f"{self.stacktrace()} ERROR {self.status} SELECT statement returned no data. Cannot identify issue/pr date range\n")
                exit()
            return [ pd.to_datetime(result.iloc[0,0]).strftime('%Y-%m-%d'), pd.to_datetime(result.iloc[0,1]).strftime('%Y-%m-%d') ]
        except Exception as e:
            self.status = 11
            print(f"{self.stacktrace()} ERROR {self.status} general SQL error.\n")
            self.print_exception(e, 'SQL')
            exit()

    def get_issue_ages(self, issue_type, month_end_date):
        '''
    Class: GithubIssuesDB
    Method: get_issue_ages()
    Argument(s):
        issue_type          string  ('issue' | 'pr')
        month_end_date      string  (YYYY-MM-DD)
    Return Value(s):
        [age1, age2,...]    list(numpy.int32)
    Calculate the age of all issues that remain open at the end of a given
    month and return a list of those open issue ages
        '''
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql = '''
                    SELECT opened_date FROM %s
                    WHERE repo_id = ?
                    AND type = ?
                    AND date(opened_date) <= ?                                
                    AND (closed_date is NULL OR date(closed_date) > ?)
                    ORDER BY opened_date
            ''' % (self.table,)
            # get opened dates for all issues that remained open at the end of the given month
            result = pd.read_sql(sql,
                                 self.engine,
                                 parse_dates={'opened_date': {'format': '%Y-%m-%d', 'utc': True}},
                                 params=(self.repository_id, issue_type, month_end_date, month_end_date))
            self.status = 0
            
            month_end_date = pd.to_datetime(month_end_date, format='%Y-%m-%d', utc=True)
            if result.empty:
                month = calendar.month_abbr[month_end_date.month]
                year = month_end_date.year
                print(f"{self.stacktrace()} WARN no open {issue_type}s remained at month end for {month} {year}.")
                print(f"{self.stacktrace()} WARN returning an empty dataframe.")
                # create a minimal dataframe (required to keep box & violin plots happy)
                result = pd.DataFrame([float('nan'), float('nan')])
                return result
            # calculate ages (in days) for all issues that remained open in the given month 
            return result.applymap(lambda x: (month_end_date - x).days)
        except Exception as e:
            self.status = 12
            print(f"{self.stacktrace()} ERROR {self.status} '{self.table}' table does not exist.\n")
            self.print_exception(e, 'SQL')
            return "unknown"
        
    
    def show_issues(self, issue_type):
        self.__set_table(self.allowable_tables['issues']) 
        try:
            sql =   '''
                        SELECT  id, repo_name, type, state, opened_date, closed_date,
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
        self.__set_table(self.allowable_tables['issues']) 
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
        return result.iloc[0,0]



    def show_statistics(self, date_range):
        print(f"                             ISSUES             PULL REQUESTS        CURRENTLY      CURRENTLY OPEN AND   ")
        print(f"REPOSITORY  YYYY-MM    OPENED      CLOSED     OPENED      CLOSED       OPEN       UNLABELLED  UNASSIGNED ")
        print("==========================================================================================================")
        repository_name = self.get_repo_name()
        for idx in date_range:
            [opened_issue_count, closed_issue_count] = self.count_issues('issue',
                                                                         idx.year, idx.month)
            [opened_pr_count, closed_pr_count]       = self.count_issues('pull_request',
                                                                         idx.year, idx.month)
            open_count = self.count_monthly_open('issue', idx.year, idx.month)
            unlabelled_count = self.count_monthly_unlabelled('issue', idx.year, idx.month)
            unassigned_count = self.count_monthly_unassigned('issue', idx.year, idx.month)

            print("%10s  %4d-%02d     %4d       %4d       %4d       %4d         %4d         %4d        %4d"
                    % (repository_name, idx.year, idx.month,
                       opened_issue_count, closed_issue_count,
                       opened_pr_count, closed_pr_count,
                       open_count, unlabelled_count, unassigned_count))


class GithubIssuesData:

    def __init__(self):
        self.__status                  = 0
        self.__timespan_months         = 12
        self.__utils                   = GithubIssuesUtils()
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
        self.__open_issue_counts             = np.zeros(length, dtype=int)
        self.__unlabelled_issue_counts       = np.zeros(length, dtype=int)
        self.__unassigned_issue_counts       = np.zeros(length, dtype=int)
        self.__total_open_issue_counts       = np.zeros(length, dtype=int)
        self.__total_unlabelled_issue_counts = np.zeros(length, dtype=int)
        self.__total_unassigned_issue_counts = np.zeros(length, dtype=int)
        self.set_month_labels(date_range, True)
        self.__labels = pd.DataFrame()
        self.__groups = pd.DataFrame()
        self.__issue_ages = []
        

    def set_plot_window(self, args : dict) -> list:

        self.__status = 0
        self.__window_start_idx = None
        self.__window_end_idx = None
        
        ref_date = args['ref_date']
        timespan = args['timespan']
        
        search_string = calendar.month_abbr[pd.to_datetime(ref_date).month] + '-' + str(pd.to_datetime(ref_date).year)[2:]    
        date_search = np.where(self.__month_labels == search_string)
        self.__window_end_idx = (self.__month_labels.size-1 if date_search[0].size == 0 else date_search[0][0])
        if date_search[0].size == 0:
            print(f"{self.__utils.stacktrace()} WARN last plot month {search_string} is not within the data timespan")
            print(f"{self.__utils.stacktrace()} WARN setting last plot month to {self.__month_labels[self.__window_end_idx]}")
            self.__status = 102

        # now set the start month relative to the end month
        if self.__window_end_idx + 1 - timespan < 1:
            self.__window_start_idx = 1
            print(f"{self.__utils.stacktrace()} WARN unable to accommodate a {timespan} month plot window")
            print(f"{self.__utils.stacktrace()} WARN setting first plot month to {self.__month_labels[1]} ({self.__window_end_idx} months)")
            self.__status = (103 if self.__status == 102 else 101)
        else:
            self.__window_start_idx = self.__window_end_idx + 1 - timespan
            print(f"{self.__utils.stacktrace()} INFO setting first plot month to {self.__month_labels[self.__window_start_idx]}")

        return [self.__window_start_idx, self.__window_end_idx]
    

    def __get_plot_window(self) -> list:
        start_month = pd.to_datetime(self.__month_labels[self.__window_start_idx], format='%b-%y')
        end_month = pd.to_datetime(self.__month_labels[self.__window_end_idx], format='%b-%y')
        end_month = pd.Period.to_timestamp(pd.Period(end_month, 'M')+1)
        plot_window = pd.date_range(start=start_month, end=end_month, closed=None, freq='M')
        return plot_window

    plot_window = property(__get_plot_window, None)


    def calculate_monthly_issues_mix(self, args):
        '''
    Class: GithubIssuesData
    Method: calculate_monthly_issues_mix()
    Argument(s):
        args                    A dictionary of argument/value pairs                
    Return Value(s):
        [[-1:1], [-1:1],...]    list(numpy.int32)
    Calculate and return a list of the monthly mix of opened/closed issues
    covering the time span of the data collected
    range = [-1, 1], -ve => closed > opened, +ve => opened > closed
        '''
        [w_start, w_end] = self.set_plot_window(args)
        w_start -= 1    # we want 1 month prior as well
        w_end += 1      # account for open ended ranges

        # calculate monthly mix of opened/closed issues
        # range = [-1, 1], -ve => closed > opened, +ve => opened > closed
        w_opened = self.opened_issue_counts[w_start:w_end].astype('float')
        w_closed = self.closed_issue_counts[w_start:w_end].astype('float')
        monthly_sum          = np.add(w_opened, w_closed)
        monthly_issues_mix = -1 + (2 * np.true_divide(w_closed, monthly_sum,
                                                      out=np.full_like(w_closed, 0.5, dtype=np.float),
                                                      where=monthly_sum!=0))
        return monthly_issues_mix


    def __get_status(self) -> int:
        return self.__status
        
    def __set_status(self, status_value: int):
        self.__status = status_value

    status = property(__get_status, __set_status)


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


    def __get_open_issue_counts(self) -> np.ndarray:
        return self.__open_issue_counts
        
    def __set_open_issue_counts(self, counts: np.ndarray):
        self.__open_issue_counts = np.copy(counts)

    open_issue_counts = property(__get_open_issue_counts, __set_open_issue_counts)


    def __get_unlabelled_issue_counts(self) -> np.ndarray:
        return self.__unlabelled_issue_counts
        
    def __set_unlabelled_issue_counts(self, counts: np.ndarray):
        self.__unlabelled_issue_counts = np.copy(counts)

    unlabelled_issue_counts = property(__get_unlabelled_issue_counts, __set_unlabelled_issue_counts)


    def __get_unassigned_issue_counts(self) -> np.ndarray:
        return self.__unassigned_issue_counts
        
    def __set_unassigned_issue_counts(self, counts: np.ndarray):
        self.__unassigned_issue_counts = np.copy(counts)

    unassigned_issue_counts = property(__get_unassigned_issue_counts, __set_unassigned_issue_counts)


    def __get_total_open_issue_counts(self) -> np.ndarray:
        return self.__total_open_issue_counts
        
    def __set_total_open_issue_counts(self, counts: np.ndarray):
        self.__total_open_issue_counts = np.copy(counts)

    total_open_issue_counts = property(__get_total_open_issue_counts, __set_total_open_issue_counts)


    def __get_total_unlabelled_issue_counts(self) -> np.ndarray:
        return self.__total_unlabelled_issue_counts
        
    def __set_total_unlabelled_issue_counts(self, counts: np.ndarray):
        self.__total_unlabelled_issue_counts = np.copy(counts)

    total_unlabelled_issue_counts = property(__get_total_unlabelled_issue_counts, __set_total_unlabelled_issue_counts)


    def __get_total_unassigned_issue_counts(self) -> np.ndarray:
        return self.__total_unassigned_issue_counts
        
    def __set_total_unassigned_issue_counts(self, counts: np.ndarray):
        self.__total_unassigned_issue_counts = np.copy(counts)

    total_unassigned_issue_counts = property(__get_total_unassigned_issue_counts, __set_total_unassigned_issue_counts)


    def __get_labels(self) -> pd.DataFrame:
        return self.__labels
        
    def __set_labels(self, labels: pd.DataFrame):
        self.__labels = labels.copy()
    
    labels = property(__get_labels, __set_labels)


    def __get_groups(self) -> pd.DataFrame:
        return self.__groups
        
    def __set_groups(self, label_file: str):
        '''
    groups property setter
    parameter: a file path 
    reads in a label CSV file associating labels to groups
    populates a pandas dataframe, removing duplicates and skipping missing values
    assigns the resulting dataframe to the private __groups attribute
        '''
        df = pd.read_csv(label_file, names=['label','group'])
        df.drop_duplicates(subset=['label'], keep='first')
        df.dropna().reset_index()
        self.__groups = df


    groups = property(__get_groups, __set_groups)


    def group_labels(self, db):
        # add a group classification column
        self.__labels['label'] = self.__labels.index
        self.__labels['group'] = self.__labels['label'].apply(lambda x: (db.get_label_group(x)))
        # add an aggregated group count column
        self.__labels['group_open_count'] = self.__labels['group'].apply(lambda x: self.__labels.groupby(['group'])['open_count'].sum().get(x))


    def __get_issue_ages(self) -> list:
        return self.__issue_ages
        
    def __set_issue_ages(self, ages: list):
        self.__issue_ages = ages.copy()

    def issue_ages_append(self, ages: np.ndarray):
        self.__issue_ages.append(ages.copy())

    issue_ages = property(__get_issue_ages, __set_issue_ages)
