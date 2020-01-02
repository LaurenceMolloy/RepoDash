import re
import sys, argparse

from GithubIssues import GithubIssuesUtils, GithubIssuesAPI, GithubIssuesDB, GithubIssuesData

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba, ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#################################################################
### ONLINE REFERENCES                                         ###
### PANDAS:     https://pandas.pydata.org/pandas-docs/stable/ ###
### SQLITE:     https://sqlite.org/docs.html                  ###
### NUMPY:      https://docs.scipy.org/doc/numpy/reference/   ###
### MATPLOTLIB: https://matplotlib.org/3.1.1/tutorials        ###
#################################################################

gu = GithubIssuesUtils()
args = gu.process_args()
args = gu.args

db = GithubIssuesDB(f'{gu.data_path}/issues', 'issues', echo=False)
# wipe and regenerate the issues database with every run.
# future versions may implement more extensive CRUD capabilities
db.drop_table()
db.create_table()

# collect issues data via Gtihub API (JSON)
ga = GithubIssuesAPI()
ga.set_seed_url(args)
for page in range(0, args['page_count']):
    ga.get_next_page(args)
    for issue in ga.response.json():
        db.insert_issue(issue)
    # stop when we've reached last page of issues
    if ga.status == 202:
        break

# if no issue activity is found, go no further
if db.count_records(f"{args['issue_type']}") == 0:
    print (f"ERROR: no {args['issue_type']}s found")
    exit()

gd = GithubIssuesData()
data_span = db.monthly_span
gd.init_arrays(data_span)



# populate numpy data arrays for plotting
# ([1] opened, [2] closed, [3] open & unlabelled [4] open [5] total open & [6] ages)
print("INFO Processing data...")
i = 0
for idx in data_span:
    # [1] & [2]
    [gd.opened_issue_counts[i],
     gd.closed_issue_counts[i]] = db.count_issues(args['issue_type'],
                                                  idx.year,
                                                  idx.month)
    # [3]
    gd.unlabelled_issue_counts[i] = db.count_monthly_unlabelled(args['issue_type'],
                                                                idx.year,
                                                                idx.month)
    # [4]
    gd.open_issue_counts[i] = db.count_monthly_open(args['issue_type'],
                                                    idx.year,
                                                    idx.month)
    # [5]
    if i == 0:
        gd.total_open_issue_counts[0] = (gd.open_issue_counts[0]
                                         if args['offset_month'] == 'opened'
                                         else gd.opened_issue_counts[0] - gd.closed_issue_counts[0])
        gd.total_unlabelled_issue_counts[0] = gd.unlabelled_issue_counts[0]
    else:
        gd.total_open_issue_counts[i] = (gd.total_open_issue_counts[(i-1)] + gd.open_issue_counts[i]
                                         if args['offset_month'] == 'opened'
                                         else gd.total_open_issue_counts[(i-1)] + gd.opened_issue_counts[i] - gd.closed_issue_counts[i])
        gd.total_unlabelled_issue_counts[i] = gd.total_unlabelled_issue_counts[i-1] + gd.unlabelled_issue_counts[i]       
    # [6]
    npa = db.get_issue_ages(args['issue_type'], idx.strftime('%Y-%m-%d'))
    gd.issue_ages.append(npa.values.flatten())
    i += 1

# identify start and end array indices for requested plotting time span
[w_start, w_end] = gd.set_plot_window(args)
w_start -= 1
w_end += 1

#db.show_issues(f"{args['issue_type']}")
#db.show_statistics(gd.plot_window)

# calculate monthly mix of opened/closed issues
# range = [-1, 1], -ve => closed > opened, +ve => opened > closed
w_opened = gd.opened_issue_counts[w_start:w_end].astype('float')
w_closed = gd.closed_issue_counts[w_start:w_end].astype('float')
monthly_sum          = np.add(w_opened, w_closed)
monthly_issues_mix = -1 + (2 * np.true_divide(w_closed, monthly_sum,
                                              out=np.zeros_like(w_closed),
                                              where=monthly_sum!=0))

# define heatmaps for displaying monthly issue mix data 
cm_greens = cm.get_cmap('Greens', 15) # green = good (list reduction)
cm_reds   = cm.get_cmap('Reds', 15)   # red   = bad  (list growth)
cm_greens.set_over('#FFFF00')         # use yellow as default out of range green colour
cm_reds.set_over('#FF00FF')           # use purple as default out of range red colour

# constrauct the colorbar from the heatmaps
pos_cmap = cm.get_cmap('Greens_r', 15)
neg_cmap = cm.get_cmap('Reds', 15)
newcolors = np.vstack((pos_cmap(np.linspace(0, 0.7, 15)),
                       neg_cmap(np.linspace(0.3, 1, 15))))
combined_cmap = ListedColormap(newcolors, name='GreenRed')
neutral = np.array(to_rgba('skyblue'))
newcolors[14:16, :] = neutral # define neutral colour for balanced mix of issues
for color in newcolors:       # set transparency level (same as for plots)
    color[3] = 0.75           

# set monthly time block separation space
bar_spacing = 0.1

# figure settings
#plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)
title = f"Analysis of {db.get_repo_name()} Github {args['issue_type']}s for period {gd.month_labels[w_start+1]} to {gd.month_labels[w_end-1]}"
fig.suptitle(title, fontsize=16)

#######################################################################
### TOP SUBPLOT - count of issues opened or closed during the month ###
#######################################################################
plt.subplot(3,1,1)

opened = gd.opened_issue_counts[w_start:w_end]
closed = gd.closed_issue_counts[w_start:w_end]
unlabelled = gd.unlabelled_issue_counts[w_start:w_end]
open_issue = gd.open_issue_counts[w_start:w_end]

max_height = np.concatenate([opened[1:], closed[1:]]).max()

ax_top = plt.gca()
label_offset = (0.5 - bar_spacing) / 2
    
# create monthly fill areas/lines
for i in range(1,opened.size):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # time block shading
    plt.bar(start, max_height*1.1, width=1-(2*bar_spacing), align='edge', color='#EEEEEE',alpha=1.0)

    plt.bar(start, opened[i], width=0.5-bar_spacing,   align='edge', color='#BB0000',alpha=0.5)    
    plt.bar(i, closed[i], width=0.5-bar_spacing,   align='edge', color='#00BB00',alpha=0.5)

    plt.bar(start, open_issue[i], width=0.5-bar_spacing,   align='edge', color='w',alpha=1)
    plt.bar(start, open_issue[i], width=0.5-bar_spacing,   align='edge', color='r',alpha=0.4)
    plt.plot([start,start+0.5-bar_spacing], [open_issue[i],open_issue[i]], color="w", alpha=1, linewidth=0.75)

    plt.bar(start, unlabelled[i], width=0.5-bar_spacing,   align='edge', color='w',alpha=1)
    plt.bar(start, unlabelled[i], width=0.5-bar_spacing,   align='edge', color='r',alpha=0.2)
    plt.plot([start,start+0.5-bar_spacing], [unlabelled[i],unlabelled[i]], color="w", alpha=1, linewidth=0.75)

    bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec=(0.75,0,0), lw=1.5)
    ax_top.text(i-label_offset, opened[i]+(max_height*0.05),
                str(opened[i]),
                ha="center", va="top", rotation=0,
                size=12, weight="bold",
                bbox=bbox_props)
    
    bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec=(0,0.75,0), lw=1.5)
    ax_top.text(i+label_offset, closed[i]+(max_height*0.05),
                str(closed[i]),
                ha="center", va="top", rotation=0,
                size=12, weight="bold",
                bbox=bbox_props)

plt.axis([0, opened.size, 0, max_height*1.1])

plt.gca().invert_yaxis()

plt.ylabel(f"Newly Opened/Closed {args['issue_type']}s", labelpad=10)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

light_red_patch   = mpatches.Patch(color='r', alpha=0.2, label=f"{args['issue_type']} requires triage")
mid_red_patch   = mpatches.Patch(color='r', alpha=0.4, label=f"{args['issue_type']} still open")
red_patch   = mpatches.Patch(color='#BB0000', alpha=0.5, label=f"opened {args['issue_type']}s")
green_patch = mpatches.Patch(color='#00BB00', alpha=0.5, label=f"closed {args['issue_type']}s")
plt.legend(handles=[light_red_patch, mid_red_patch, red_patch, green_patch],
           bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right',
           ncol=4, borderaxespad=0)

# Show the grid lines as dark grey lines
plt.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)

###########################################################################
### MIDDLE SUBPLOT - total issues open at the start & end of each month ###
###########################################################################
plt.subplot(3,1,2)

total_open = gd.total_open_issue_counts[w_start:w_end]
total_unlabelled = gd.total_unlabelled_issue_counts[w_start:w_end]

y_range = total_open.max() - total_open.min()
y_padding = y_range * 0.1 

ax_mid = plt.gca()
bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec="k", lw=1.5)

t = ax_mid.text(0.5, total_open[0]+(y_padding/2),
                str(total_open[0]),
                ha="center", va="bottom", rotation=0,
                size=12, weight="bold",
                bbox=bbox_props)

# create monthly fill areas/lines
for i in range(1,total_open.size):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # determine color map to use (increasing/decreasing numbers and deadband around zero)
    if monthly_issues_mix[i] < -0.05:
        col = cm_reds((-monthly_issues_mix[i]+0.5)/1.5)
    elif monthly_issues_mix[i] > 0.05:
        col = cm_greens((monthly_issues_mix[i]+0.5)/1.5)
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
    # fill areas (unlabelled)
    plt.fill_between([start,finish],
                     total_unlabelled[i-1:i+1],
                     color='w', alpha=1)
    plt.fill_between([start,finish],
                     total_unlabelled[i-1:i+1],
                     color='r', alpha=0.2)
    
    # open top lines
    plt.plot(        [start,finish],
                     total_open[i-1:i+1],
                     color="w", alpha=1, linewidth=3)

    # unlabelled top lines
    plt.plot(        [start,finish],
                     total_unlabelled[i-1:i+1],
                     color="w", alpha=1, linewidth=0.75)
    
    t = ax_mid.text(i+0.5, total_open[i]+(y_padding/2),
                    str(total_open[i]),
                    ha="center", va="bottom", rotation=0,
                    size=12, weight="bold",
                    bbox=bbox_props)

plt.axis([0, total_open.size, max(0,total_open.min()-y_padding), total_open.max()+y_padding])
plt.ylabel(f"Total Open {args['issue_type']}s", labelpad=10)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

# Show the grid lines as dark grey lines
plt.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)

light_red_patch   = mpatches.Patch(color='r', alpha=0.2, label=f"aggregate # of {args['issue_type']}s requiring triage")
plt.legend(handles=[light_red_patch],
           bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right',
           ncol=1, borderaxespad=0)

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
label = f"mix of {args['issue_type']} activity\n+100 = 100% opened {args['issue_type']}s\n-100 = 100% closed {args['issue_type']}s"
cbar.set_label(label)

#################################################################################
### BOTTOM SUBPLOT - age distribution of open issues at the end of each month ###
#################################################################################
plt.subplot(3,1,3)

ages = gd.issue_ages[w_start:w_end]
positions = np.arange(0,len(ages))
labels = gd.month_labels[w_start:w_end]

medianprops = dict(linewidth=2, color='w')

max_age = 0
for item in ages:
    if len(item) > 0 and item.max() > max_age:
        max_age = item.max()

# create monthly fill areas/lines
for i in range(1,len(ages)):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    # time block shading
    plt.fill_between([start,finish], [max_age,max_age], color='#EEAAAA', alpha=0.5)
    plt.fill_between([start,finish], [360,360], color='#EEEEAA', alpha=0.5)
    plt.fill_between([start,finish], [180,180],   color='#AAEEAA', alpha=0.5)

plt.boxplot(ages[1:], positions=positions[1:],
            widths=0.4, whis=[5, 95], notch=True, showfliers=False,
            patch_artist=True, medianprops=medianprops)
plt.axis([0,len(ages),0,max_age])
plt.xticks(np.arange(labels.size), labels, rotation=60) 
plt.xlabel("Calendar Month")
plt.ylabel(f"{args['issue_type']} Age (days)", labelpad=10)

# Show the grid lines as dark grey lines
plt.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)

above_patch  = mpatches.Patch(color='#EEAAAA', label='above target (bad)')
target_patch = mpatches.Patch(color='#EEEEAA', label='target age range')
below_patch  = mpatches.Patch(color='#AAEEAA', label='below target (good)')
plt.legend(handles=[above_patch, target_patch, below_patch],
           bbox_to_anchor=(0, 1.02, 1, .102),
           loc='lower right',
           ncol=3,
           borderaxespad=0)

plt.show()
