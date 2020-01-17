import re
import sys, argparse
import math

from GithubIssues import GithubIssuesUtils, GithubIssuesAPI, GithubIssuesDB, GithubIssuesData

import pandas as pd
import numpy as np

import matplotlib as mpl
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

# simply write debug info to a file and quit if -i/--info arg is supplied
if args['info'] == True:
    gu.write_debug_info()
    exit()

# wipe and create an empty issues database table with every run
# (future versions may implement incremental update capabilities)
db = GithubIssuesDB(f'{gu.data_path}/issues', 'issues', echo=False)
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

# initialise numpy arrays for plotting data
gd = GithubIssuesData()
data_span = db.monthly_span
gd.init_arrays(data_span)

# populate numpy data arrays for plotting
# ([1] opened, [2] closed, [3] open & unlabelled [4] open & unassigned [5] open [6] total open & [7] ages)
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
    gd.unassigned_issue_counts[i] = db.count_monthly_unassigned(args['issue_type'],
                                                                idx.year,
                                                                idx.month)
    # [5]
    gd.open_issue_counts[i] = db.count_monthly_open(args['issue_type'],
                                                    idx.year,
                                                    idx.month)
    # [6]
    if i == 0:
        gd.total_open_issue_counts[0] = (gd.open_issue_counts[0]
                                         if args['offset_month'] == 'opened'
                                         else gd.opened_issue_counts[0] - gd.closed_issue_counts[0])
        gd.total_unlabelled_issue_counts[0] = gd.unlabelled_issue_counts[0]
        gd.total_unassigned_issue_counts[0] = gd.unassigned_issue_counts[0]
    else:
        gd.total_open_issue_counts[i] = (gd.total_open_issue_counts[(i-1)] + gd.open_issue_counts[i]
                                         if args['offset_month'] == 'opened'
                                         else gd.total_open_issue_counts[(i-1)] + gd.opened_issue_counts[i] - gd.closed_issue_counts[i])
        gd.total_unlabelled_issue_counts[i] = gd.total_unlabelled_issue_counts[i-1] + gd.unlabelled_issue_counts[i]       
        gd.total_unassigned_issue_counts[i] = gd.total_unassigned_issue_counts[i-1] + gd.unassigned_issue_counts[i]       
    # [7]
    npa = db.get_issue_ages(args['issue_type'], idx.strftime('%Y-%m-%d'))
    gd.issue_ages.append(npa.values.flatten())
    i += 1

# identify start and end array indices for requested plotting time span
[w_start, w_end] = gd.set_plot_window(args)
w_start -= 1 # we want 1 month prior as well
w_end += 1   # account for open ended ranges

# optional debug functions
# db.show_issues(f"{args['issue_type']}")
# db.show_statistics(gd.plot_window)

# calculate monthly mix of opened/closed issues
# range = [-1, 1], -ve => closed > opened, +ve => opened > closed
w_opened = gd.opened_issue_counts[w_start:w_end].astype('float')
w_closed = gd.closed_issue_counts[w_start:w_end].astype('float')
monthly_sum          = np.add(w_opened, w_closed)
monthly_issues_mix = -1 + (2 * np.true_divide(w_closed, monthly_sum,
                                              out=np.full_like(w_closed, 0.5, dtype=np.float),
                                              where=monthly_sum!=0))

# define heatmaps for displaying monthly issue mix data 
cm_greens = cm.get_cmap('Greens', 15) # green = good (list reduction)
cm_reds   = cm.get_cmap('Reds', 15)   # red   = bad  (list growth)
cm_greens.set_over('#FFFF00')         # use yellow as default out of range green colour
cm_reds.set_over('#FF00FF')           # use purple as default out of range red colour

# construct the colorbar from the heatmaps
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
bar_spacing = 0.05

# figure settings
#plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)
title = f"Analysis of {db.get_repo_name()} Github {args['issue_type']}s for period {gd.month_labels[w_start+1]} to {gd.month_labels[w_end-1]}"
fig.suptitle(title, fontsize=16)

#######################################################################
### TOP SUBPLOT - count of issues opened or closed during the month ###
#######################################################################

def plot_monthly_bar(x, y, label_offset_y=None, ax=None, **kwargs):
    
    
    # provide default axis if axis is not specified
    if ax is None:
        ax = plt.gca()

    config = {'color': 'w', 'alpha': 1, 'zorder': 1, 'width': 1, 'align': 'edge',
              'line_color': 'w', 'line_alpha': 1, 'line_width': 1,
              **kwargs.get('config', {})}

    start_x = (x if config['align'] == 'edge' else x-(config['width']/2))

    if y > 0:
        ax.bar(x, y, width=config['width'], align=config['align'],
               color='w',
               alpha=1,
               zorder=config['zorder'])
        ax.bar(x, y, width=config['width'], align=config['align'],
               color=config['color'],
               alpha=config['alpha'],
               zorder=config['zorder'])
        ax.plot([start_x,start_x+config['width']], [y,y],
                linewidth=config['line_width'],
                color=config['line_color'],
                alpha=config['line_alpha'],
                zorder=config['zorder'])
    
    if label_offset_y is not None:
        label_offset_x = config['width'] / 2
        bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec=(0.75,0,0), lw=1.5)
        ax.text(start_x+label_offset_x, y+label_offset_y,
                str(y),
                ha="center", va="top", rotation=0,
                size=12, weight="bold",
                bbox=bbox_props)

def plot_monthly_counts(opened_counts, closed_counts , ax=None, **kwargs):
    
    # provide default axis if axis is not specified
    if ax is None:
        ax = plt.gca()
    
    # process configuration values, setting to default values where silent
    tbox_config = {'color': '#EEEEEE', 'alpha': 1, 'zorder': 0, 'align': 'edge', 
                   'line_color': 'w', 'line_alpha': 0, 'line_width': 0, 'spacing': 0.05,
                   **kwargs.get('tbox_config', {})}
    tbox_config.update({'width': 1 - (2 * tbox_config['spacing'])})
    
    opened_config = {'color': '#BB0000', 'alpha': 0.5, 'zorder': 1, 'align': 'center', 
                     'line_color': 'w', 'line_alpha': 1, 'line_width': 1, 'width_pct': 0.45,
                     **kwargs.get('opened_config', {})}
    opened_config.update({'width': opened_config['width_pct']*tbox_config['width']})

    closed_config = {'color': '#00BB00', 'alpha': 0.5, 'zorder': 2, 'align': 'center',
                     'line_color': 'w', 'line_alpha': 1, 'line_width': 1, 'width_pct': 0.45,
                     **kwargs.get('closed_config', {})}
    closed_config.update({'width': closed_config['width_pct']*tbox_config['width']})
    
    open_config = {'color': '#FF0000', 'alpha': 0.4, 'zorder': 3, 'align': 'center',
                   'line_color': 'w', 'line_alpha': 1, 'line_width': 1, 'width_pct': 0.45,
                   **kwargs.get('open_config', {})}
    open_config.update({'width': open_config['width_pct']*tbox_config['width']})
    
    unlabelled_config = {'color': '#FF0000', 'alpha': 0.2, 'zorder': 4, 'align': 'center',
                         'line_color': 'w', 'line_alpha': 1, 'line_width': 1, 'width_pct': 0.45,
                         **kwargs.get('unlabelled_config', {})}
    unlabelled_config.update({'width': unlabelled_config['width_pct']*tbox_config['width']})
    
    unassigned_config = {'color': '#BB0000', 'alpha': 0.8, 'zorder': 5, 'align': 'center',
                         'line_color': 'w', 'line_alpha': 0, 'line_width': 0, 'width_pct': 0.05,
                         **kwargs.get('unassigned_config', {})}
    unassigned_config.update({'width': unassigned_config['width_pct']*tbox_config['width']})
    
    # process other keyword arguments
    unlabelled_counts = kwargs.get('unlabelled_counts', [])
    unassigned_counts = kwargs.get('unassigned_counts', [])
    open_counts = kwargs.get('open_counts', [])    
    start_idx = kwargs.get('start_idx', 0)
    end_idx = kwargs.get('end_idx', None)

    # define data slices according to start_idx and end_idx, with sensible defaults
    if end_idx is None:
        opened = opened_counts[start_idx:]
        closed = closed_counts[start_idx:]
        unlabelled = (unlabelled_counts[start_idx:] if unlabelled_counts is not None else [])
        unassigned = (unassigned_counts[start_idx:] if unassigned_counts is not None else [])
        open_issue = (open_counts[start_idx:] if open_counts is not None else [])
    else:
        opened = opened_counts[start_idx:end_idx]
        closed = closed_counts[start_idx:end_idx]
        unlabelled = (unlabelled_counts[start_idx:end_idx] if unlabelled_counts is not None else [])
        unassigned = (unassigned_counts[start_idx:end_idx] if unassigned_counts is not None else [])
        open_issue = (open_counts[start_idx:end_idx] if open_counts is not None else [])

    # calculate required axis height, allowing for some headroom
    ax_height = math.ceil(np.concatenate([opened[1:], closed[1:]]).max() * 1.1)

    # create monthly fill areas/lines
    for i in range(1,opened.size):
        start  = i - (tbox_config['width']/2)
        finish = i + (tbox_config['width']/2)

        # time block shading
        plot_monthly_bar(start, ax_height, ax=ax, config=tbox_config)

        # 'opened this month' count bar
        x = (start if opened_config['align'] == 'edge' else (start+i)/2)
        plot_monthly_bar(x, opened[i], label_offset_y=(ax_height*0.05), ax=ax, config=opened_config)
        
        # 'closed this month' count bar
        x = (i if closed_config['align'] == 'edge' else (i+finish)/2)
        plot_monthly_bar(x, closed[i], label_offset_y=(ax_height*0.05), ax=ax, config=closed_config)

        # (optional) 'opened this month & currently still open' count bar
        if len(open_issue) > 0:
            x = (start if open_config['align'] == 'edge' else (start+i)/2)
            plot_monthly_bar(x, open_issue[i], ax=ax, config=open_config)

        # (optional) 'opened this month, currently still open & unlabelled' count bar
        if len(unlabelled) > 0:
            x = (start if unlabelled_config['align'] == 'edge' else (start+i)/2)
            plot_monthly_bar(x, unlabelled[i], ax=ax, config=unlabelled_config)
        
        # (optional) 'opened this month, currently still open & unassigned' count bar
        if len(unassigned) > 0:
            x = (start if unassigned_config['align'] == 'edge' else (start+i)/2)
            plot_monthly_bar(x, unassigned[i], ax=ax, config=unassigned_config)
    
    ax.axis([0, opened.size, 0, ax_height])
    ax.invert_yaxis()
    ax.set_ylabel(f"Newly Opened/Closed {args['issue_type']}s", labelpad=10)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
 
    legend = []
    legend.append(mpatches.Patch(color=opened_config['color'], alpha=opened_config['alpha'],
                                 label=f"opened {args['issue_type']}s"))
    legend.append(mpatches.Patch(color=closed_config['color'], alpha=closed_config['alpha'],
                                 label=f"closed {args['issue_type']}s"))
    if len(open_issue) > 0:
        legend.append(mpatches.Patch(color=open_config['color'], alpha=open_config['alpha'],
                                     label=f"{args['issue_type']}s still open"))
    if len(unlabelled) > 0:
        legend.append(mpatches.Patch(color=unlabelled_config['color'], alpha=unlabelled_config['alpha'],
                                     label=f"{args['issue_type']}s requires triage"))
    if len(unassigned) > 0:
        legend.append(mpatches.Patch(color=unassigned_config['color'], alpha=unassigned_config['alpha'],
                                     label=f"unassigned open {args['issue_type']}s"))
    ax.legend(handles=legend, ncol=len(legend), borderaxespad=0, bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right')

    # Show the grid lines as dark grey lines
    ax.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)


# calling code for topmost metrics dashboard (monthly counts)
ax_top = plt.subplot(3,1,1)
plot_monthly_counts(gd.opened_issue_counts, gd.closed_issue_counts, start_idx=w_start, end_idx=w_end, ax=ax_top,
                    open_counts=gd.open_issue_counts,
                    unlabelled_counts=gd.unlabelled_issue_counts,
                    unassigned_counts=gd.unassigned_issue_counts)

###########################################################################
### MIDDLE SUBPLOT - total issues open at the start & end of each month ###
###########################################################################
plt.subplot(3,1,2)

total_open = gd.total_open_issue_counts[w_start:w_end]
total_unlabelled = gd.total_unlabelled_issue_counts[w_start:w_end]


y_range = max(total_open.max() - total_open.min(), total_unlabelled.max() - total_unlabelled.min())
# y_range = total_open.max() - total_open.min()
y_padding = math.ceil(y_range * 0.1) 

# first axis created is plotted first, with axis and labels on the left
# we want 'requires triage' count to go behind 'open' count
ax_mid_r = plt.gca()
ax_mid_l = ax_mid_r.twinx()
ax_mid = ax_mid_l.twinx()

# swap left and right ticks and labels around
# we want 'open' count on left / 'requires triage' on right
ax_mid_l.yaxis.tick_left()
ax_mid_l.yaxis.set_label_position("left")
ax_mid_r.yaxis.tick_right()
ax_mid_r.yaxis.set_label_position("right")

ax_mid_l.set_ylabel(f"Total Open {args['issue_type']}s", labelpad=10)
ax_mid_r.set_ylabel(f"Total Open {args['issue_type']}s Requiring Triage", labelpad=10)

bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec="k", lw=1.5)

t = ax_mid_l.text(0.5, total_open[0]+(y_padding/2),
                  str(total_open[0]),
                  ha="center", va="bottom", rotation=0,
                  size=12, weight="bold",
                  bbox=bbox_props)

# create monthly fill areas/lines
for i in range(1,total_open.size):
    start  = i - (0.5-bar_spacing)
    finish = i + (0.5-bar_spacing)

    start_unl  = i - 0.5
    finish_unl = i + 0.5

    start_narrow  = i - ((0.5-bar_spacing)*0.7)
    finish_narrow = i + ((0.5-bar_spacing)*0.7)

    # time block shading
    ax_mid_r.fill_between([start,finish],
                          [total_open.max()+y_padding, total_open.max()+y_padding],
                          y2=0, color='#EEEEEE', alpha=1)
        
    # fill areas (unlabelled)
    ax_mid_r.fill_between([start,finish],
                          total_unlabelled[i-1:i+1],
                          y2=0, facecolor='w', alpha=1)
    
    # fill areas (unlabelled)
    ax_mid_r.fill_between([start,finish],
                          total_unlabelled[i-1:i+1],
                          y2=0, facecolor='r', alpha=0.2)

    # top lines (unlabelled)
    ax_mid_r.plot([start,finish],
                  total_unlabelled[i-1:i+1],
                  color="w", alpha=1, linewidth=3)

    # determine color map to use (increasing/decreasing numbers and deadband around zero)
    if monthly_issues_mix[i] < -0.05:
        col = cm_reds((-monthly_issues_mix[i]+0.5)/1.5)
    elif monthly_issues_mix[i] > 0.05:
        col = cm_greens((monthly_issues_mix[i]+0.5)/1.5)
    else:
        col = "skyblue"

    # fill areas (open)
    ax_mid_l.fill_between([start_narrow,finish_narrow],
                          total_open[i-1:i+1],
                          y2=0, color=col, alpha=1)
    
    # top lines (open)
    ax_mid_l.plot([start_narrow,finish_narrow],
                  total_open[i-1:i+1],
                  color="w", alpha=1, linewidth=3)
    # top lines (unlabelled)
    ax_mid.plot([start,finish],
                  total_unlabelled[i-1:i+1],
                  color="w", alpha=1, linewidth=0.3)
    
    t = ax_mid_l.text(i+0.5, total_open[i]+(y_padding/2),
                      str(total_open[i]),
                      ha="center", va="bottom", rotation=0,
                      size=12, weight="bold",
                      bbox=bbox_props)

# plot 'requires triage' using a linearly offset scale relative to 'open'
mid_l_range = y_range + y_padding
ax_mid_l.axis([0, total_open.size, total_open.min(), total_open.min() + mid_l_range])
ax_mid_r.axis([0, total_open.size, total_unlabelled.min(), total_unlabelled.min() + mid_l_range])
ax_mid.axis([0, total_open.size, total_unlabelled.min(), total_unlabelled.min() + mid_l_range])

ax_mid_r.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

# Show the grid lines as dark grey lines
ax_mid_l.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)

light_red_patch   = mpatches.Patch(fc='r', alpha=0.2, label=f"aggregate # of {args['issue_type']}s requiring triage")
ax_mid_r.legend(handles=[light_red_patch],
                bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right',
                ncol=1, borderaxespad=0)

### create a color map bar to display on the right side of the plot
cbar_ax = inset_axes(ax[1],
                     width="2%",     # width  = 2% of parent_bbox width
                     height="100%",  # height = 100% of parent_bbox height
                     loc='lower left',
                     bbox_to_anchor=(1.06, 0, 1, 1),
                     bbox_transform=ax[1].transAxes,
                     borderpad=0)

data = np.random.randn(1, 1) # dummy 2D array (allows me to create the pcolormesh below)
psm = ax[1].pcolormesh(data, cmap=combined_cmap, rasterized=True, vmin=-100, vmax=100)
cbar = fig.colorbar(psm, ax=ax[1], cax=cbar_ax)
cbar.ax.get_yaxis().set_ticks([])
cbar.ax.set_ylabel(f"<--more closed      more opened-->\nmix of {args['issue_type']} activity", rotation=90)

#################################################################################
### BOTTOM SUBPLOT - age distribution of open issues at the end of each month ###
#################################################################################
plt.subplot(3,1,3)

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

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

parts = plt.violinplot(ages[1:], positions=positions[1:],
                       showmeans=False, showmedians=False, showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
for month in ages:
    q1, m, q3 = np.percentile(month.tolist(), [25, 50, 75], axis=0)
    quartile1.append(q1)
    medians.append(m)
    quartile3.append(q3)

whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                     for sorted_array, q1, q3
                     in zip(ages, quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[1:, 0], whiskers[1:, 1]

inds = np.arange(0, len(medians))
plt.scatter(inds[1:], medians[1:], marker='o', color='white', s=30, zorder=3)
plt.vlines(inds[1:], quartile1[1:], quartile3[1:], color='k', linestyle='-', lw=5)
plt.vlines(inds[1:], whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

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

if args['save_file'] is None:
    plt.show()
else:
    print(f"{gu.stacktrace()} INFO writing metrics image to file ({args['save_file']}).") 
    plt.savefig(args['save_file'])