<h1>Running RepoDash</h1>


- cd <GIT FOLDER>
- git clone https://github.com/LaurenceMolloy/RepoDash.git
- cd RepoDash/src

To run the program in default (demo) mode, simply type 

    python3 RepoDash.py

This will read in the first 10 pages of the matplotlib repository issues, 1000 issues in all, 
and plot metrics for the latest 12 months of issues of type 'issue' found (i.e. not pull-requests, 
which are also reported along with issues in the Github API).

You can configure Repodash to run against any repository, requesting any number of pages starting 
from any specific page of issues as well as output any number of months of metrics up to any 
specified month.

If you specify a refernce date for metrics and/or timespan that falls outside of the date range of 
the issues data collected, RepoDash will do its best to adjust the dates and/or shorten the analysis 
timespan so that it maps to the data available.

Command line options:

- -u, --user         Github username or account (default: 'matplotlib')
- -r, --repo         Github repository name (default: 'matplotlib')
- -t, --type         Issue type, either 'issue' (default) or 'pull_request'
- -m, --months       plot metric analysis timespan in months (default: 12)
- -d, --refdate      plot metric reference end date (default: now)
- -f, --firstpage    first page number to request (default: 1)
- -c, --pagecount    number of pages of issues to request (default: 10)
- -p, --datapath     location of SQLite database (default: <REPO PATH>/data)

Example usage:

**EXAMPLE 1:** Request the first 6 pages of issues from the Numpy repository and plot issues list 
metrics for the period June 2012 to September 2012 inclusive (4 months).

> python3 RepoDash.py -u numpy -r numpy -m 3 -d '2012-09' -c 6


**EXAMPLE2:** Request pages 100 to 105 (12 pages) of issues from the Kubernetes repository and plot 
the last 6 months of metrics. The silent reference date defaults 'now'. If 'now' falls outside of the 
date range observed in the collected date range, RepoDash will map the analysis period to the latest N
months (where N is the timespan we have specified on the command line).

> python3 RepoDash.py -u kubernetes -r kubernetes -m 6 -f 100 -c 12


