from GithubIssues import GithubIssuesUtils, GithubIssuesAPI, GithubIssuesDB, GithubIssuesData
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba, ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mplcursors

#################################################################
### ONLINE REFERENCES FOR MAJOR MODULE DEPENDENCIES           ###
### PANDAS:     https://pandas.pydata.org/pandas-docs/stable/ ###
### SQLITE:     https://sqlite.org/docs.html                  ###
### NUMPY:      https://docs.scipy.org/doc/numpy/reference/   ###
### MATPLOTLIB: https://matplotlib.org/3.1.1/tutorials        ###
#################################################################

def process_labels(ga, db, gd, args):
    '''
    Function: process_labels()
    Argument(s):
        ga      GithubIssuesAPI instance
        db      GithubIssuesDB instance
        gd      GithubIssuesData instance
        args    A dictionary of argument/value pairs
    Return Value(s): None
    Process the first N pages of labels used by the repository's issues 
    into the RepoDash database (page count determined by command line argument) 
    '''
    # read in labels list (multiple-pages)
    ga.set_seed_url(args, 'labels')
    for page in range(0, args['page_count']):
        ga.get_next_page(args)
        for label in ga.response.json():
            db.insert_label(label)
        # stop when we've reached last page of labels
        if ga.status == 202:
            break
    # read in a label grouping file if one is supplied
    # produces a stacked (grouped) bar chart of label counts for open issues
    if not args['label_file'] is None:
        gd.groups = args['label_file']
        db.update_labels(gd.groups)
    # in label file generation mode,
    # just write the labels processed out to a file and exit
    if args['out_label_file'] is not None:
        db.get_labels().to_csv(args['out_label_file'], 
                               columns=['label', 'label_group'], 
                               encoding='utf-8', 
                               index=False)
        exit()


def process_issues(ga, db, gd, args):
    '''
    Function: process_issues()
    
    Argument(s):
        ga      GithubIssuesAPI instance
        db      GithubIssuesDB instance
        gd      GithubIssuesData instance
        args    A dictionary of argument/value pairs
    
    Return Value(s): 
        data_span   DateTimeIndex (month-end dates, YYYY-MM-DD format)
    data_span represents months for which issues data has been processed
    
    Process N pages of issues logged against the repository into the RepoDash
    database (start page & page count determined by command line argument) 
    '''
    # read in issues list (multiple-pages)
    ga.set_seed_url(args, 'issues')
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
    data_span = db.monthly_span
    gd.init_arrays(data_span)
    return data_span


def calculate_monthly_stats(db, gd, data_span, args):
    '''
    Function: calculate_monthly_stats()
    Argument(s):
        db          GithubIssuesDB instance
        gd          GithubIssuesData instance
        date_span   DateTimeIndex (month-end dates, YYYY-MM-DD format)
        args        A dictionary of argument/value pairs
    Return Value(s): None
    Populate the following monthly numpy data arrays for plotting
    (arrays stored as list attributes within the db instance)
    [1]  issues opened during the given month
    [2]  issues closed during the given month
    [3]  issues opened during the given month, currently still open
    [4]  issues opened during the given month, currently still open & unlabelled
    [5]  issues opened during the given month, currently still open & unassigned
    [6]  total count of issues that remain open/unlabelled/unassigned at the end of the given month 
    [7]  ages of all issues that remain open at the end of the given month 
    [8]  monthly issues mix [-1,1]: -ve => closed>opened, +ve => opened>closed
    [9]  label counts for open issues at the end of the given month
    '''
    print("INFO Processing data...")
    i = 0
    for idx in data_span:
        # [1] & [2]
        [gd.opened_issue_counts[i],
        gd.closed_issue_counts[i]] = db.count_issues(args['issue_type'],
                                                    idx.year,
                                                    idx.month)
        # [3]
        gd.open_issue_counts[i] = db.count_monthly_open(args['issue_type'],
                                                        idx.year,
                                                        idx.month)
        # [4]
        gd.unlabelled_issue_counts[i] = db.count_monthly_unlabelled(args['issue_type'],
                                                                    idx.year,
                                                                    idx.month)
        # [5]
        gd.unassigned_issue_counts[i] = db.count_monthly_unassigned(args['issue_type'],
                                                                    idx.year,
                                                                    idx.month)
        # [6]
        # initialise total open/unlabelled/unassigned at end of month with first month's total
        if i == 0:
            gd.total_open_issue_counts[0] = (gd.open_issue_counts[0]
                                            if args['offset_month'] == 'opened'
                                            else gd.opened_issue_counts[0] - gd.closed_issue_counts[0])
            gd.total_unlabelled_issue_counts[0] = gd.unlabelled_issue_counts[0]
            gd.total_unassigned_issue_counts[0] = gd.unassigned_issue_counts[0]
        # for each subsequent month, add prior month's total to current month's total
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
    # [8]
    gd.calculate_monthly_issues_mix()
    # [9] 
    gd.labels = gd.labels.add(db.count_labels_point_in_time(args['issue_type'], data_span[-1].strftime('%Y-%m-%d')), fill_value=0)


###########################################################
### LABELS SUBPLOT - count of open issue types (groups) ###
###########################################################

def plot_label_counts(gd , db, ax=None, **kwargs):
    
    # provide default axis if axis is not specified
    if ax is None:
        ax = plt.gca()
    
    # process configuration values, setting to default values where silent
    bar_config = {'ec': 'w', 'lw': 1, **kwargs.get('bar_config', {})}
    highlight_config = {'ec': 'r', 'fc': '#00FF00', **kwargs.get('highlight_config', {})}
    
    # process other keyword arguments
    top = kwargs.get('top', 12)
    level = kwargs.get('level', 'group')
    cmap = kwargs.get('cmap', 'plasma')

    gd.group_labels(db)
    gd.labels.sort_values(by=['group_open_count','open_count'], inplace=True, ascending=False )
    group_open_count = gd.labels.groupby(['group'],sort=False)['open_count'].sum()
    max_label_count = int(gd.labels['open_count'].max())
    max_group_count = int(gd.labels['group_open_count'].max())
    colors = plt.cm.get_cmap(cmap)(np.linspace(0,1,max_label_count+1))

    # reduce count of items displayed if labels < top
    top = min(group_open_count.size, top)

    iter = 0
    for name,group in gd.labels.filter(items=['label','group','open_count']).groupby([level], sort=False):
        iter += 1
        i=0
        l=0
        for index, row in group.iterrows():
            
            c = (0 if max_label_count == 0 else int(row['open_count'])/max_label_count)
            ax.bar(name, int(row['open_count']), bottom=l, color=colors[int(row['open_count'])], edgecolor=bar_config['ec'], linewidth=bar_config['lw'],
                     label=f"{row['label']}: {int(row['open_count'])}")
            l += row['open_count']
            i += 1
        if iter == top:
            break

    # account for sensible default plot scales empty open label sets
    ymax = max(4,max_group_count)
    ax.set_ylim([0,ymax])
    ax.locator_params(axis='y', nbins=4)
    
    ax.set_xticks(np.arange(top))   # len(group_open_count)))
    ax.set_xticklabels(group_open_count.index[0:top], rotation=90)
    if level == 'group':
        ax.set_ylabel(f"Label Counts for Open Issues (Top {top}, grouped)", labelpad=10)
    else:
        ax.set_ylabel(f"Label Counts for Open Issues (Top {top})", labelpad=10)
    
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y, width, height = sel.artist[sel.target.index].get_bbox().bounds
        sel.annotation.set(text=sel.artist.get_label(), size=14, position=((top-1+(bar_config['lw']/2))*0.95, ymax*0.95)) # len(group_open_count)
        sel.annotation.xy = (x + width, y + height)
        sel.annotation.set_horizontalalignment('right')
        sel.annotation.set_verticalalignment('top')
        sel.artist[sel.target.index].set_edgecolor(highlight_config['ec'])
        sel.artist[sel.target.index].set_linewidth(5)
        sel.artist[sel.target.index].set_facecolor(highlight_config['fc'])

    @cursor.connect("remove")
    def on_remove(sel):
        x, y, width, height = sel.artist[sel.target.index].get_bbox().bounds
        sel.artist[sel.target.index].set_edgecolor(bar_config['ec'])
        sel.artist[sel.target.index].set_linewidth(bar_config['lw'])
        sel.artist[sel.target.index].set_facecolor(colors[int(height)])
        for sel in cursor.selections:
            sel.artist[sel.target.index].set_edgecolor(highlight_config['ec'])
            sel.artist[sel.target.index].set_linewidth(5)
            sel.artist[sel.target.index].set_facecolor(highlight_config['fc'])

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
                size=8, weight="bold",
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
    issue_type = kwargs.get('issue_type', None)

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
    ax_height = math.ceil(np.concatenate([opened[1:], closed[1:]]).max() * 1.2)

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
    ax.set_ylabel(f"Newly Opened/Closed {issue_type}s", labelpad=10)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
 
    legend = []
    legend.append(mpatches.Patch(color=opened_config['color'], alpha=opened_config['alpha'],
                                 label=f"opened {issue_type}s"))
    legend.append(mpatches.Patch(color=closed_config['color'], alpha=closed_config['alpha'],
                                 label=f"closed {issue_type}s"))
    if len(open_issue) > 0:
        legend.append(mpatches.Patch(color=open_config['color'], alpha=open_config['alpha'],
                                     label=f"{issue_type}s still open"))
    if len(unlabelled) > 0:
        legend.append(mpatches.Patch(color=unlabelled_config['color'], alpha=unlabelled_config['alpha'],
                                     label=f"{issue_type}s requires triage"))
    if len(unassigned) > 0:
        legend.append(mpatches.Patch(color=unassigned_config['color'], alpha=unassigned_config['alpha'],
                                     label=f"unassigned open {issue_type}s"))
    ax.legend(handles=legend, ncol=len(legend), borderaxespad=0, bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right')

    # Show the grid lines as dark grey lines
    ax.grid(axis='y', b=True, which='major', color='#666666', linestyle='-', alpha=0.25)

###########################################################################
### MIDDLE SUBPLOT - total issues open at the start & end of each month ###
###########################################################################

def issue_mix_colormap():
    '''
    Function: issue_mix_colormap()
    Argument(s):        None0
    Return Value(s):    Matplotlib ListedColormap object

    Construct and return a user-defined color map for heat-mapping the panel
    displaying the total issues open at the start and end of each month, 
    according to the mix of issues opened and closed that month
    '''
    positive_colormap = cm.get_cmap('Greens_r', 15)
    negative_colormap = cm.get_cmap('Reds', 15)
    combined_colormap = np.vstack((positive_colormap(np.linspace(0, 0.7, 15)),
                                   negative_colormap(np.linspace(0.3, 1, 15))))
    neutral = np.array(to_rgba('skyblue'))
    combined_colormap[14:16, :] = neutral   # define neutral colour for balanced mix of issues
    for color in combined_colormap:         # set transparency level (same as for plots)
        color[3] = 0.75           
    return ListedColormap(combined_colormap, name='GreenRed')


def plot_total_counts(total_open_counts, total_unlabelled_counts, issues_mix,
                      fig=None, ax=None, **kwargs):
    '''
    Function: plot_total_counts()
    Argument(s):        
        total_open_counts           numpy array (int)
        total_unlabelled_counts     numpy array (int)
        issues_mix                  numpy array (int)
        ax                          matplotlib axes
        start_index                 int - start month in the array
        end_index                   int - end month in the array
        bar_spacing                 float (matplotlib configuration value)
    Return Value(s): None
    Plot the middle panel of the dataviz, including total open issues at start and 
    end of each month as well as total untriaged (unlabelled) issues.
    '''
    # first axis created is plotted first, with axis and labels on the left
    # we want 'requires triage' count to go behind 'open' count
    # provide default axis if axis is not specified
    ax_mid_r = ax if not ax is None else plt.gca()
    ax_mid_l = ax_mid_r.twinx()
    ax_mid = ax_mid_l.twinx()

    # process other keyword arguments
    start_idx = kwargs.get('start_idx', 0)
    end_idx = kwargs.get('end_idx', None)
    bar_spacing = kwargs.get('bar_spacing', 0)
    issue_type = kwargs.get('issue_type', None)

    # define data slices according to start_idx and end_idx, with sensible defaults
    if end_idx is None:
        total_open = total_open_counts[start_idx:]
        total_unlabelled = (total_unlabelled_counts[start_idx:] if total_unlabelled_counts is not None else [])
        monthly_issues_mix = issues_mix[start_idx:]
    else:
        total_open = total_open_counts[start_idx:end_idx]
        total_unlabelled = (total_unlabelled_counts[start_idx:end_idx] if total_unlabelled_counts is not None else [])
        monthly_issues_mix = issues_mix[start_idx:end_idx]

    y_range = max(total_open.max() - total_open.min(), total_unlabelled.max() - total_unlabelled.min())
    y_padding = math.ceil(y_range * 0.2)

    # swap left and right ticks and labels around
    # we want 'open' count on left / 'requires triage' on right
    ax_mid_l.yaxis.tick_left()
    ax_mid_l.yaxis.set_label_position("left")
    ax_mid_r.yaxis.tick_right()
    ax_mid_r.yaxis.set_label_position("right")

    ax_mid_l.set_ylabel(f"Total Open {issue_type}s")
    ax_mid_r.set_ylabel(f"Total Open {issue_type}s Requiring Triage")

    bbox_props = dict(boxstyle="round,pad=0.2", fc=(1,1,0.9), ec="k", lw=1.5)

    # the first total count box
    t = ax_mid_l.text(0.5, total_open[0]+(y_padding/2),
                      str(total_open[0]),
                      ha="center", va="bottom", rotation=0,
                      size=8, weight="bold",
                      bbox=bbox_props)

    # create monthly fill areas/lines
    for i in range(1,total_open.size):
        start  = i - (0.5-bar_spacing)
        finish = i + (0.5-bar_spacing)

        #start_unl  = i - 0.5
        #finish_unl = i + 0.5

        start_narrow  = i - ((0.5-bar_spacing)*0.8)
        finish_narrow = i + ((0.5-bar_spacing)*0.8)

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

        # define heatmaps for displaying monthly issue mix data 
        # determine color map to use (increasing/decreasing numbers and deadband around zero)
        # list growth (red, bad)
        if monthly_issues_mix[i] < -0.05:
            cm_reds   = cm.get_cmap('Reds', 15)
            cm_reds.set_over('#FF00FF')     # use purple as default out of range red colour
            col = cm_reds((-monthly_issues_mix[i]+0.5)/1.5)
        # list reduction (green, good)
        elif monthly_issues_mix[i] > 0.05:
            cm_greens = cm.get_cmap('Greens', 15)
            cm_greens.set_over('#FFFF00')   # use yellow as default out of range green colour
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
    
        # ith total count boxes
        t = ax_mid_l.text(i+0.5, total_open[i]+(y_padding/4),
                          str(total_open[i]),
                          ha="center", va="bottom", rotation=0,
                          size=8, weight="bold",
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

    light_red_patch   = mpatches.Patch(fc='r', alpha=0.2, label=f"aggregate # of {issue_type}s requiring triage")
    ax_mid_r.legend(handles=[light_red_patch],
                    bbox_to_anchor=(0, 1.02, 1, .102), loc='lower right',
                    ncol=1, borderaxespad=0)

    ### create a color map bar to display on the right side of the plot
    cbar_ax = inset_axes(ax_mid,
                         width="2%",     # width  = 2% of parent_bbox width
                         height="100%",  # height = 100% of parent_bbox height
                         loc='lower left',
                         bbox_to_anchor=(1.1, 0, 1, 1),
                         bbox_transform=ax_mid.transAxes,
                         borderpad=0)

    data = np.random.randn(1, 1) # dummy 2D array (allows me to create the pcolormesh below)
    psm = ax_mid.pcolormesh(data, cmap=issue_mix_colormap(), rasterized=True, vmin=-100, vmax=100)
    cbar = fig.colorbar(psm, ax=ax_mid, cax=cbar_ax)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.set_ylabel(f"<--more closed      more opened-->\nmix of {issue_type} activity", labelpad=5, rotation=90)

#################################################################################
### BOTTOM SUBPLOT - age distribution of open issues at the end of each month ###
#################################################################################
plt.subplot(3,3,(8,9))


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_age_distributions(issue_ages, month_labels, ax=None, **kwargs):
    '''
    Function: plot_age_distributions()
    Argument(s):        
        issue_ages                  numpy array (int)
        month_labels                numpy array ('Mon-YY') e.g. 'Jan-21'
        ax                          matplotlib axes
        start_index                 int - start month in the array
        end_index                   int - end month in the array
        bar_spacing                 float (matplotlib configuration value)
    Return Value(s): None
    Plot the middle panel of the dataviz, including total open issues at start and 
    end of each month as well as total untriaged (unlabelled) issues.
    '''
    # first axis created is plotted first, with axis and labels on the left
    # we want 'requires triage' count to go behind 'open' count
    # provide default axis if axis is not specified
    #ax_mid_r = ax if not ax is None else plt.gca()
    #ax_mid_l = ax_mid_r.twinx()
    #ax_mid = ax_mid_l.twinx()

    # process other keyword arguments
    start_idx = kwargs.get('start_idx', 0)
    end_idx = kwargs.get('end_idx', None)
    bar_spacing = kwargs.get('bar_spacing', 0)
    issue_type = kwargs.get('issue_type', 0)

    # define data slices according to start_idx and end_idx, with sensible defaults
    if end_idx is None:
        ages = issue_ages[start_idx:]
        labels = month_labels[start_idx:]
    else:
        ages = issue_ages[start_idx:end_idx]
        labels = month_labels[start_idx:end_idx]

    positions = np.arange(0,len(ages))

    #ages = gd.issue_ages[w_start:w_end]
    #positions = np.arange(0,len(ages))
    #labels = gd.month_labels[w_start:w_end]

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

    parts = plt.violinplot( ages[1:], positions=positions[1:],
                            showmeans=False, showmedians=False, showextrema=False )

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
    plt.ylabel(f"{issue_type} Age (days)", labelpad=10)

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


def output(args):
    if args['save_file'] is None:
        plt.show()
    else:
        print(f"{gu.stacktrace()} INFO writing metrics image to file ({args['save_file']}).") 
        plt.savefig(args['save_file'])


def main() -> None:

    # initialise the utility functions object and process the command-line arguments
    # to produce a dictionary of argument/value pairs
    gu = GithubIssuesUtils()
    args = gu.process_args()

    # initialise object for interfacing with GtihubAPI
    ga = GithubIssuesAPI()

    # initialise object for storing issues data collected via GithubAPI
    db = GithubIssuesDB(f'{gu.data_path}/issues', 'issues', echo=False)

    # initialise data structures for plotting data
    gd = GithubIssuesData()

    process_labels(ga, db, gd, args)
    data_span = process_issues(ga, db, gd, args)
    calculate_monthly_stats(db, gd, data_span, args)

    # identify start and end array indices for requested plotting time span
    [w_start, w_end] = gd.set_plot_window(args)

    # optional debug functions
    # where needed, it's best to run these after data ingestion & analysis, but before plotting
    if args['verbose_stats']: 
        db.show_issues(f"{args['issue_type']}")

    if args['summary_stats']:
        db.show_statistics(gd.plot_window)

    ## figure settings
    fig, ax = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.20)
    title = f"Analysis of {db.get_repo_name()} Github {args['issue_type']}s for period {gd.month_labels[w_start+1]} to {gd.month_labels[w_end-1]}"
    fig.suptitle(title, fontsize=16)
    # set monthly time block separation space
    bar_spacing = 0.05

    # calling code for topmost metrics dashboard (monthly counts)
    ax_lab = plt.subplot(3,3,(1,4))
    plot_label_counts(gd, db, ax=ax_lab, top=args['num_labels'])

    # calling code for topmost metrics dashboard (monthly counts)
    ax_top = plt.subplot(3,3,(2,3))
    plot_monthly_counts(gd.opened_issue_counts, gd.closed_issue_counts, 
                        ax=ax_top, start_idx=w_start, end_idx=w_end,
                        issue_type=args['issue_type'],
                        open_counts=gd.open_issue_counts,
                        unlabelled_counts=gd.unlabelled_issue_counts,
                        unassigned_counts=gd.unassigned_issue_counts)

    # calling code for middle metrics dashboard (total counts)
    ax_mid = plt.subplot(3,3,(5,6))
    plot_total_counts(gd.total_open_issue_counts, gd.total_unlabelled_issue_counts, gd.monthly_issues_mix,
                      fig=fig, ax=ax_mid, start_idx=w_start, end_idx=w_end, 
                      issue_type=args['issue_type'],
                      bar_spacing=bar_spacing)

    # calling code for middle metrics dashboard (total counts)
    ax_bottom = plt.subplot(3,3,(8,9))
    plot_age_distributions(gd.issue_ages, gd.month_labels,
                           ax=ax_bottom, start_idx=w_start, end_idx=w_end, 
                           issue_type=args['issue_type'],
                           bar_spacing=bar_spacing)

    # hide final panel (bottom left)
    plt.subplot(3,3,7).axis('off')

    # throw up a data viz or write out to a file
    output(args)

if __name__ == "__main__":
    main()
