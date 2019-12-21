import re
from pandas.tseries.offsets import MonthEnd
from GithubIssues import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba, ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


#################################################################
### ONLINE REFERENCES                                         ###
### PANDAS:     https://pandas.pydata.org/pandas-docs/stable/ ###
### SQLITE:     https://sqlite.org/docs.html                  ###
### NUMPY:      https://docs.scipy.org/doc/numpy/reference/   ###
### MATPLOTLIB: https://matplotlib.org/3.1.1/tutorials        ###
#################################################################

path_to_data = "../data"


ga  = GithubIssuesAPI()
db  = GithubIssuesDB(f'{path_to_data}/issues', 'issues', echo=False)
gd  = GithubIssuesData()

db.drop_table  ()
db.create_table()

account = "matplotlib"
repo = "matplotlib"

ga.next_page_url = f"https://api.github.com/repos/{account}/{repo}/issues?state=all&direction=asc&per_page=100&page=1"

for page in range(1,4):
    ga.get_next_page()
    for issue in ga.response.json():
        db.insert_issue(issue)

if db.count_records("issue") == 0:
    print ("NO DATA STORED")
    exit()

# get end of month dates for earliest created and latest issues 
start_date = pd.to_datetime(db.get_start_date('issue')) + MonthEnd(0)
end_date   = pd.to_datetime(db.get_end_date('issue'))   + MonthEnd(0)
date_range = pd.date_range(start=start_date, end=end_date, closed=None, freq='M')

gd.init_arrays(date_range)

i = 0
for idx in date_range:
    [gd.opened_issue_counts[i], gd.closed_issue_counts[i]] = db.count_issues('issue', idx.year, idx.month)
    if i == 0:
        gd.total_open_issue_counts[i] = gd.opened_issue_counts[i] - gd.closed_issue_counts[i]
    else:
        gd.total_open_issue_counts[i] = gd.total_open_issue_counts[(i-1)] + gd.opened_issue_counts[i] - gd.closed_issue_counts[i]
    npa = db.get_issue_ages('issue', idx.strftime('%Y-%m-%d'))
    gd.issue_ages_append(npa.values)
    i += 1

[w_start, w_end] = gd.set_plot_window(date_range[1],date_range[-1]) 
w_start -= 1
w_end += 1

#db.show_issues("issue")
#db.show_statistics(date_range)

#########################################################################
### TASK: calculate monthly rate of growth/reduction in issues list   ###
### (the percentage of open issues not offset by closed ones)         ###
### METHODS USED: add / negative / divide                             ###
### OUTPUT: an array with 12 numbers representing monthly growth rate ###
### REF: NUMPY >> Mathematical Functions >> Arithmetic Operations     ###
#########################################################################
monthly_diff         = np.add(gd.opened_issue_counts[w_start:w_end],
                              np.negative(gd.closed_issue_counts[w_start:w_end]))
monthly_diff_ratio   = np.divide(monthly_diff,
                                 gd.opened_issue_counts[w_start:w_end])
monthly_sum          = np.add(gd.opened_issue_counts[w_start:w_end],
                              gd.closed_issue_counts[w_start:w_end])
monthly_issues_progress = (2 * np.divide(gd.closed_issue_counts[w_start:w_end],
                                         monthly_sum)) - 1


########################################################################
### TASK: define heatmaps for displaying monthly_diff_ratio data     ###
### METHODS USED: get_cmap                                           ###
### OUTPUT: callable objects that return RGBA values when called     ###
###         with a value in the range [0-1] e.g. cm_greens(0.5)      ###
### NOTE: values outside this range gives default colours which      ###
###       can be re-defined with set_over and set_under methods      ### 
### REF: MATPLOTLIB >> Colors >> Creating Colormaps in Matplotlib    ###
###                                                                  ###
### A selection of named colormaps to choose from can be found here: ### 
### https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html     ###
########################################################################
cm_greens = cm.get_cmap('Greens', 15) # green = good (list reduction)
cm_reds   = cm.get_cmap('Reds', 15)   # red   = bad  (list growth)
cm_greens.set_over('#FFFF00')         # use yellow as default out of range green colour
cm_reds.set_over('#FF00FF')           # use purple as default out of range red colour

pos_cmap = cm.get_cmap('Greens_r', 15)
neg_cmap = cm.get_cmap('Reds', 15)
newcolors = np.vstack((pos_cmap(np.linspace(0, 0.7, 15)),
                       neg_cmap(np.linspace(0.3, 1, 15))))
combined_cmap = ListedColormap(newcolors, name='GreenRed')
neutral = np.array(to_rgba('skyblue'))
newcolors[14:16, :] = neutral

for color in newcolors:
    color[3] = 0.75

# monthly time block separation space
bar_spacing = 0.1

fig, ax = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=False)
title = f"Analysis of {db.get_repo_name()} Github issues for period {gd.month_labels[w_start+1]} to {gd.month_labels[w_end-1]}"
fig.suptitle(title, fontsize=14)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)

#######################################################################
### TOP SUBPLOT - count of issues opened or closed during the month ###
#######################################################################
plt.subplot(3,1,1)

opened = gd.opened_issue_counts[w_start:w_end]
closed = gd.closed_issue_counts[w_start:w_end]

max_height = np.concatenate([opened, closed]).max()

# create monthly fill areas/lines
for i in range(1,opened.size):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # time block shading
    plt.bar(start, max_height*1.1, width=1-(2*bar_spacing), align='edge', color='#EEEEEE',alpha=1.0)
    plt.bar(start, opened[i],      width=0.5-bar_spacing,   align='edge', color='#BB0000',alpha=0.5)
    plt.bar(i,     closed[i],      width=0.5-bar_spacing,   align='edge', color='#00BB00',alpha=0.5)

plt.axis([0, opened.size, 0, max_height*1.1])

plt.gca().invert_yaxis()

plt.ylabel("Newly Opened/Closed Issues", labelpad=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

red_patch   = mpatches.Patch(color='#BB0000', label='opened issues')
green_patch = mpatches.Patch(color='#00BB00', label='closed issues')
plt.legend(handles=[red_patch, green_patch], bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right',
           ncol=2, borderaxespad=0)


###########################################################################
### MIDDLE SUBPLOT - total issues open at the start & end of each month ###
###########################################################################
plt.subplot(3,1,2)

total_open = gd.total_open_issue_counts[w_start:w_end]

# create monthly fill areas/lines
for i in range(1,total_open.size):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # determine color map to use (increasing/decreasing numbers and deadband around zero)
    if monthly_issues_progress[i] < -0.05:
        col = cm_reds((-monthly_issues_progress[i]+0.5)/1.5)
    elif monthly_issues_progress[i] > 0.05:
        col = cm_greens((monthly_issues_progress[i]+0.5)/1.5)
    else:
        col = "skyblue"

    # time block shading
    plt.fill_between([start,finish],
                     [total_open.max()+10, total_open.max()+10],
                     color='#EEEEEE', alpha=1.0)
    # fill areas
    plt.fill_between([start,finish],
                     total_open[i-1:i+1],
                     color=col, alpha=0.75)
    # top lines
    plt.plot(        [start,finish],
                     total_open[i-1:i+1],
                     color="w", alpha=1, linewidth=2)

plt.axis([0, total_open.size, max(0,total_open.min()-10), total_open.max()+10])
plt.ylabel("Total Open Issues")

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


### create a color map bar to display on teh right side of the plot
cbar_ax = inset_axes(ax[1],
                     width="2%",     # width  = 2% of parent_bbox width
                     height="100%",  # height = 100% of parent_bbox height
                     loc='lower left',
                     bbox_to_anchor=(1.01, 0., 1, 1),
                     bbox_transform=ax[1].transAxes,
                     borderpad=0)
data = np.random.randn(1, 1) # dummy 2D array (allows me to create the pcolormesh below)
psm = ax[1].pcolormesh(data, cmap=combined_cmap, rasterized=True, vmin=-100, vmax=100)
cbar = fig.colorbar(psm, ax=ax[1], cax=cbar_ax)
label = 'mix of issue activity\n+100 = only opened issues\n-100 = only closed issues\n0 = equal amounts of both'
cbar.set_label(label)

#################################################################################
### BOTTOM SUBPLOT - age distribution of open issues at the end of each month ###
#################################################################################
plt.subplot(3,1,3)

ages = gd.issue_ages[w_start:w_end]
positions = np.arange(0,len(ages))
labels = gd.month_labels[w_start:w_end]

max_age = 0
for item in ages:
    if item.max() > max_age:
        max_age = item.max()

# create monthly fill areas/lines
for i in range(1,len(ages)):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # time block shading
    plt.fill_between([start,finish], [max_age,max_age], color='#EEAAAA', alpha=0.5)
    plt.fill_between([start,finish], [180,180], color='#EEEEAA', alpha=0.5)
    plt.fill_between([start,finish], [90,90],   color='#AAEEAA', alpha=0.5)

plt.boxplot(ages[1:], widths=0.5, positions=positions[1:], patch_artist=True)
plt.axis([0,len(ages),0,max_age])
plt.xticks(np.arange(labels.size), labels, rotation=60) 
plt.xlabel("Calendar Month")
plt.ylabel("Issue Age (days)", labelpad=10)

above_patch  = mpatches.Patch(color='#EEAAAA', label='above target (bad)')
target_patch = mpatches.Patch(color='#EEEEAA', label='target age range')
below_patch  = mpatches.Patch(color='#AAEEAA', label='below target (good)')
plt.legend(handles=[above_patch, target_patch, below_patch],
           bbox_to_anchor=(0, 1.02, 1, .102),
           loc='lower right',
           ncol=3,
           borderaxespad=0)

plt.show()
