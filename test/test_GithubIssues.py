'''
RepoDash, by Laurence Molloy - (c) 2019-2023

Filename:   test_GithubIssues.py
Purpose:    test code that exercises all GithubISsues module functionality 
            including a range of edge cases

NOTES:
1. this is a work in progress and is known to be incomplete - needs more work
'''

import os
import sys
sys.path.insert(0, "../src/")
from GithubIssues import GithubIssuesAPI, GithubIssuesDB, GithubIssuesData
import pytest


def test_GA_db_drop():
    """
    database file removal
    When/if we encounter an ERROR 15, we'll need to cater for it
    """
    db = 'test'
    table = 'issues'

    # create the database handler object
    db = GithubIssuesDB(db, table, echo=False)
    assert db.status == 0  # SUCCESS

    # remove old db file if one exists
    if os.path.exists(db.db_file):
        status = db.drop_db()
        assert status == 0  # SUCCESS

    # create a table (this creates a new db file)
    status = db.create_table()
    assert status == 0  # SUCCESS

    # delete the db file that's just been created
    status = db.drop_db()
    assert status == 0  # SUCCESS

    # attempt deletion of (now) non-existent db file
    status = db.drop_db()
    assert status == 16  # WARNING 16: db file does not exist


def test_GA_bad_file_permissions():
    """
    failure to create a db file due to bad permissions
    """
    path = '/tmp/githubissues-test/'
    db = path + 'test'
    table = 'issues'

    # create a folder with zero permissions
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o000)

    # create the database handler object
    db = GithubIssuesDB(db, table, echo=False)
    assert db.status == 0  # SUCCESS

    # attempt to create a table (which would create a new db file)
    status = db.create_table()
    assert status == 4  # ERROR 4: unable to create db file

    # clean up after test
    os.chmod(path, 0o777)
    os.rmdir(path)


def test_GA_invalid_table_drop():
    """
    correct handling of invalid table drops
    """
    db = 'test'
    table = 'issues'

    # create the database handler object
    db = GithubIssuesDB(db, table, echo=False)
    assert db.status == 0  # SUCCESS

    # prepare test conditions - remove old db file if one exists
    if os.path.exists(db.db_file):
        status = db.drop_db()
        assert status == 0  # SUCCESS

    # create issues table (this creates a new db file)
    status = db.create_table()
    assert status == 0  # SUCCESS

    # drop issues table
    status = db.drop_table()
    assert status == 0  # SUCCESS

    # attempt to drop issues table again
    status = db.drop_table()
    assert status == 3  # ERROR 3: non-existent table

    # clean up after test
    status = db.drop_db()
    assert status == 0  # SUCCESS


def test_GA_duplicate_table():
    """
    creating a duplicate db table
    """
    db = 'test'
    table = 'issues'

    # create the database handler object
    db = GithubIssuesDB(db, table, echo=False)
    assert db.status == 0  # SUCCESS

    # prepare test conditions - remove old db file if one exists
    if os.path.exists(db.db_file):
        status = db.drop_db()
        assert status == 0  # SUCCESS

    # create issues table (this creates a new db file)
    status = db.create_table()
    assert status == 0  # SUCCESS

    # attempt to create another issues table
    status = db.create_table()
    assert status == 4  # ERROR 4: table already exists

    # clean up after test
    status = db.drop_db()
    assert status == 0  # SUCCESS


def test_GA_unsupported_table_name():
    """
    creating a table that is unsupported
    """
    db = 'test'
    table = 'unsupported_table_name'

    # create the database handler object
    db = GithubIssuesDB(db, table, echo=False)
    assert db.status == 1  # ERROR 1: unsupported table name

    # clean up after test
    status = db.drop_db()
    assert status == 16  # WARNING 16: db file does not exist


testdata = [
        {'timespan': 2, 'ref_date': '2011-03', 'status': 101, 'start': 1, 'end': 1},
        {'timespan': 3, 'ref_date': '2011-04', 'status': 101, 'start': 1, 'end': 2},
        {'timespan': 2, 'ref_date': '2011-04', 'status':   0, 'start': 1, 'end': 2},
        {'timespan': 3, 'ref_date': '2011-05', 'status': 103, 'start': 1, 'end': 2},
        {'timespan': 2, 'ref_date': '2011-05', 'status': 102, 'start': 1, 'end': 2},
        {'timespan': 1, 'ref_date': '2011-02', 'status': 102, 'start': 2, 'end': 2},
        {'timespan': 2, 'ref_date': '2011-02', 'status': 102, 'start': 1, 'end': 2},
        {'timespan': 6, 'ref_date': '2011-06', 'status': 103, 'start': 1, 'end': 2},
        {'timespan': 2, 'ref_date': '2011-06', 'status': 102, 'start': 1, 'end': 2}
]


@pytest.mark.parametrize("args", testdata)
def test_GID_analysis_period_range_checks(args):
    """
    tests for both in-range and out-of-range analysis periods
    (compares issues list time span with analysis period time span and adjusts with warnings)
    """
    db_name = 'test'
    table_name = 'issues'
    url_args = {'account': 'matplotlib', 'repo': 'matplotlib', 'first_page': 1}

    db = GithubIssuesDB(db_name, table_name, echo=False)
    assert db.status == 0  # SUCCESS

    # prepare test conditions - remove old db file if one exists
    if os.path.exists(db.db_file):
        status = db.drop_db()
        assert status == 0  # SUCCESS

    # create issues table (this creates a new db file)
    status = db.create_table()
    assert status == 0  # SUCCESS

    ga = GithubIssuesAPI()
    ga.set_seed_url(url_args)
    assert ga.status == 0  # SUCCESS

    gd = GithubIssuesData()
    assert gd.status == 0  # SUCCESS

    # request N pages and process data into a database
    for page in range(0, 1):
        ga.get_next_page()
        for issue in ga.response.json():
            db.insert_issue(issue)
    assert db.status == 0  # SUCCESS

    # initialise plotting data arrays (y axes)
    # and create an array of month labels (x-axis)
    gd.init_arrays(db.monthly_span)
    assert gd.status == 0  # SUCCESS
    assert len(gd.plot_points) == 3
    assert len(gd.opened_issue_counts) == 3
    assert len(gd.closed_issue_counts) == 3
    assert len(gd.total_open_issue_counts) == 3
    assert len(gd.issue_ages) == 0
    assert len(gd.month_labels) == 3
    assert gd.month_labels[0] == ''
    assert gd.month_labels[1] == 'Mar-11'
    assert gd.month_labels[2] == 'Apr-11'

    # start date prior to issues list date range
    [w_start, w_end] = gd.set_plot_window(args)
    assert gd.status == args['status']  # WARN
    assert w_start == args['start']
    assert w_end == args['end']


def test_GID_single_page_issues_list():
    """
    processing an issues list that only has a single page of issues
    (Link field in header is not present)
    NOT SURE HOW TO MOCK THIS SCENARIO
    """

    # db = 'test'
    # table = 'issues'
    # account = 'numpy'
    # repo = 'numpy'


def test_GID_no_issues():
    """
    processing an issues list that only has a single page of issues
    (the earliest page of the numpy issues list has zero open issues)
    """

    db_name = 'test'
    table_name = 'issues'
    args = {'account': 'numpy',
            'repo': 'numpy',
            'issue_type': 'issue',
            'first_page': 1,
            'page_count': 1}

    ga = GithubIssuesAPI()
    ga.set_seed_url(args)
    assert ga.status == 0  # SUCCESS

    # create the database handler object
    db = GithubIssuesDB(db_name, table_name, echo=False)
    assert db.status == 0  # SUCCESS

    # prepare test conditions - remove old db file if one exists
    if os.path.exists(db.db_file):
        status = db.drop_db()
        assert status == 0  # SUCCESS

    # create issues table (this creates a new db file)
    status = db.create_table()
    assert status == 0  # SUCCESS

    for page in range(0, args['page_count']):
        ga.get_next_page()
        for issue in ga.response.json():
            db.insert_issue(issue)
    assert db.status == 0  # SUCCESS
    assert db.count_records("issue") == 0


def test_simple():
    """
    very simple example test
    """
    assert 1 + 1 == 2
