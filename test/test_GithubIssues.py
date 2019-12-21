import os
import sys
sys.path.insert(0, "../src/")
from GithubIssues import GithubIssuesAPI, GithubIssuesDB, GithubIssuesData

# import re
# from pandas.tseries.offsets import MonthEnd
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import ListedColormap, to_rgba
# import matplotlib.patches as mpatches
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import numpy as np

ga = GithubIssuesAPI()
gd = GithubIssuesData()


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


def test_simple():
    """
    very simple example test
    """
    assert 1 + 1 == 2
