import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from src.per_trips import fitGMM

etho_label={0:'border',1:'turn', 2: 'run',  3:'feeding', 4:'other'}
seg_label={0:'border',1:'visit', 2: 'loop',  3:'finding', 4:'leaving',5:'missing'}

# color code:
etho_color={0:'grey',1:'teal', 2: 'lightcoral',  3:'darkorange', 4:'darkorchid'}
seg_color={0:'grey',1:'cornflowerblue', 2: 'indianred',  3:'forestgreen',
           4:'peru',5:'yellowgreen'}

# colors for conditions
condition_palette = {
    "CantonSMH\n24hr":'orangered',
    "CantonSMH\n40hr":'brown',
    "CantonSMH\n0-125M_24hr":'orangered',
    "CantonSMH\n0-125M_40hr":'brown',
    "EPG-TNT\n24hr":'gold',
    "EPG-Kir\n24hr": 'goldenrod',
    "EPG2-Kir\n24hr": 'yellow',
    "SS96xCSMH-TNT\n0-125M_24hr":'gold',
    "SS96xCS-Kir\n0-125M_24hr": 'goldenrod',
    "SS90_x_Kir\n0-125M_24hr": 'yellow',
    "Or83b\n24hr":'palevioletred',
    "Tmem63\n24hr":'cornflowerblue',
    "Tmem63b\n24hr":'lightskyblue',
    'SS41970_Kir\n24hr':'slateblue',
    'SS41970_TNT\n24hr':'mediumpurple',
    'SS42302_Kir\n24hr':'purple',
    'SS45222_Kir\n24hr': 'plum',
    'SS45222_TNT\n24hr': 'orchid',
    'SS51315_Kir\n24hr': 'pink',
    'SS51315_TNT\n24hr': 'hotpink',
    'Gr43a\n40hr': 'fuchsia',
    'Gr5a\n40hr': 'm'}

cond_color = {
    '0-125M_24hr': 'orangered',
    '0-125M_24hr_ctrl': 'lightsalmon',
    '0-125M_40hr': 'brown',
    '0M_24hr': 'lightseagreen',
    '0M_40hr': 'teal',
    'sorbitol': 'steelblue',
    'sucralose': 'y',
    'sucrose': 'tomato',
    'optoGr43a_40hr': 'fuchsia',
    'optoGr5a_40hr': 'm',
    'Gr43a-40hr': 'deeppink',
    'Gr5a-40hr': 'violet',
    'Orco-40hr': 'coral',
    'Or42b-40hr': 'c',
    'Or42b_7d_starved-7d': 'teal',
    'Or42b_HCS-40hr': 'grey',
    '24hr': 'orangered', 
    '40hr': 'brown'
}

trip_color = {
    'short': 'mediumpurple',
    'long': 'teal'
}

trip_color_vs = {
    'very short': 'purple',
    'short': 'mediumpurple',
    'long': 'teal'
}
    
pal = {"allJR_WT":'grey',
       "CantonSMH":'grey',
       "allJR_WT_ctrl":'grey',
       "EPG-TNT":'deepskyblue',
       "EPG-Kir": 'royalblue',
       "EPG2-Kir": 'turquoise',
       "SS96xCSMH-TNT":'deepskyblue',
       "SS96xCS-Kir": 'royalblue',
       "SS90_x_Kir": 'turquoise',
       "Or83b":'palevioletred',
       "Tmem63":'green',#'cornflowerblue',
       "Tmem63b":'yellowgreen',#lightskyblue',
       "Tmem63bxCSMH":'tan',
       "Tmem63b_x_CSMH":'tan',
       'SS41970_Kir':'slateblue',
       'SS41970_TNT':'mediumpurple',
       'SS42302_Kir':'purple',
       'SS45222_Kir': 'plum',
       'SS45222_TNT': 'orchid',
       'SS51315_Kir': 'pink',
       'SS51315_TNT': 'hotpink',
       'Gr43a': 'fuchsia',
       'Gr5a': 'blueviolet',
       'Or42b': 'teal',
       "Orco":'palevioletred',
      }

def pval2star(pval):
    if pval <= 0.001: return '***'
    elif pval <= 0.01: return '**'
    elif pval <= 0.05: return '*'
    else: return 'ns'

# axis related
def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)

def axisTheme(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def beautifyAxes(ax,xlab,ylab,titlestr):
    sns.despine(ax=ax, offset=5, trim=False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(titlestr)

# specialized plots
def plotArenaAndFood(ax, foodRad, arenaRad, foodCol='r'):
    foodpatch = plt.Circle((0,0),foodRad,color=foodCol, alpha=0.1)
    ax.add_artist(foodpatch)
    arena = plt.Circle((0,0),arenaRad, color='grey', fill=False)
    ax.add_artist(arena)
    ax.set_xlim(-arenaRad-5,arenaRad+5)
    ax.set_ylim(-arenaRad-5,arenaRad+5)
    ax.set_aspect('equal')
    ax.axis('off')

def plotTrajectory(data, metadata, x='body_x', y='body_y', hue='segment', palette=[v for v in seg_color.values()], size=1, alpha=0.2, ax=None):
    if ax is None:
        ax = plt.gca()

    xvec, yvec, huevec = data[x].values, data[y].values, data[hue].values
    for e, c in zip(data[hue].unique(), palette):
        ax.scatter(xvec[huevec==e], yvec[huevec==e], color=c, s=size, alpha=alpha)
    plotArenaAndFood(ax, metadata['spot_radius'], metadata['arena_radius']/metadata['px_per_mm'])

    return ax


def stripWithBoxplot(ax, datadf, xvals, yvals, colpalette, ylabeltext, order=None, ylimvals=None, minvals=0, dotsize=6, jitter=0.2, boxwidth=0.7, clip=False):
    #filter if desired to have minvals number of observations in each shown bin
    tmp = dict(datadf[xvals].value_counts())
    for j in np.sort(datadf[xvals].dropna().unique()):
        if tmp[j] < minvals: datadf[datadf[xvals] == j] = np.nan;

    if order is None:
        order = datadf[xvals].sort_values().unique()
    if clip:
        ylims = ax.get_ylim()
        sns.stripplot(data = datadf.loc[(datadf[yvals]>=ylims[0])&(datadf[yvals]<ylims[1])], x=xvals, y=yvals, order=order,
                      ax=ax,palette=colpalette, hue=xvals, size=dotsize, jitter=jitter, legend = False, clip_on=False)
    else:
        sns.stripplot(data = datadf, x=xvals, y=yvals, order=order, ax=ax,palette=colpalette, hue=xvals, size=dotsize, jitter=jitter,
                      legend = False, clip_on=False)

    sns.boxplot(data = datadf, x=xvals, y=yvals, order=order, ax=ax,color='white', whis=False, fliersize=0, width=boxwidth)

    meanVals = datadf.groupby([xvals],observed=False).mean(numeric_only=True).reset_index()
    if clip:
        sns.stripplot(x=xvals, y=yvals,linewidth=3,dodge=False,data=meanVals.loc[(meanVals[yvals]>=ylims[0])&(meanVals[yvals]<ylims[1])],
                      palette=colpalette,hue=xvals, ax=ax, legend = False)
    else:
        sns.stripplot(x=xvals, y=yvals,linewidth=3,dodge=False,data=meanVals, palette=colpalette,hue=xvals, ax=ax, legend = False)

    ax.set_ylabel(ylabeltext)
    if ylimvals is not None:
        ax.set_ylim(ylimvals)
    else:
        if clip:
            ax.set_ylim(ylims)

    stats_df = []
    if len(datadf.condition.unique())==2:
        groups = []
        groups = [datadf.loc[datadf.condition==cond, yvals].values for cond in datadf.condition.unique()]
        stat, pval = stats.ranksums(groups[0][~np.isnan(groups[0])], groups[1][~np.isnan(groups[1])])
        print(f"pval = {pval:.3e}")
        this_stats = {  'group1': datadf.condition.unique()[0], 
                        'group2': datadf.condition.unique()[1],
                        'test': 'ranksums',
                        'statistic': stat,
                        'p-value': pval}
        stats_df.append(this_stats)
    else:
        groups = [datadf.loc[datadf.condition==cond, yvals].values for cond in datadf.condition.unique()]
        labels = [cond for cond in datadf.condition.unique()]
        for i, a in enumerate(groups):
            for j in range(i,len(groups)):
                stat, pval = stats.ranksums(groups[i][~np.isnan(groups[i])], groups[j][~np.isnan(groups[j])])
                print(f"{labels[i]} vs {labels[j]}: pval = {pval:.3e}")
                this_stats = {  'group1': labels[i],
                                'group2': labels[j],
                                'test': 'ranksums',
                                'statistic': stat,
                                'p-value': pval}
                stats_df.append(this_stats)
    stats_df = pd.DataFrame(stats_df)

    myAxisTheme(ax)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

    return ax, stats_df

def stripWithBoxplot_color(ax, datadf, xvals, yvals, colpalette, ylabeltext, ylimvals, minvals=0, dotsize=6, jitter=0.2, boxwidth=0.7):
    #filter if desired to have minvals number of observations in each shown bin
    tmp = dict(datadf[xvals].value_counts())
    for j in np.sort(datadf[xvals].dropna().unique()):
        if tmp[j] < minvals: datadf[datadf[xvals] == j] = np.nan;
    try:
        sns.stripplot(data = datadf, x=xvals, y=yvals, ax=ax,palette=colpalette, hue=xvals, size=dotsize, jitter=jitter)
    except ValueError:
        sns.stripplot(data = datadf, x=xvals, y=yvals, ax=ax,color=colpalette, size=dotsize, jitter=jitter)

    sns.boxplot(data = datadf, x=xvals, y=yvals, ax=ax,color='white', whis=False, fliersize=0, width=boxwidth)

    meanVals = datadf.groupby([xvals]).mean(numeric_only=True).reset_index()
    sns.stripplot(x=xvals, y=yvals,linewidth=2, color='k',dodge=False,data=meanVals, ax=ax)

    ax.set_ylabel(ylabeltext)
    ax.set_ylim(ylimvals)
    ax.legend('', frameon=False)
    myAxisTheme(ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    return ax

def stripWithBoxplot_oldSeaborn(ax, datadf, xvals, yvals, colpalette, ylabeltext, huevals=None, order=None, ylimvals=None, dotsize=6, jitter=0.2, boxwidth=0.7):
    if order is None:
        order = datadf[xvals].unique()
    if huevals is None:
        huevals = xvals
    sns.stripplot(data = datadf, x=xvals, y=yvals, order=order, ax=ax,palette=colpalette, hue=xvals, size=dotsize, jitter=jitter)
    sns.boxplot(data = datadf, x=xvals, y=yvals, order=order, ax=ax,color='white', whis=False, fliersize=0, width=boxwidth)

    meanVals = datadf.groupby([xvals]).mean(numeric_only=True).reset_index()
    sns.stripplot(x=xvals, y=yvals,linewidth=3,dodge=False,data=meanVals, palette=colpalette,hue=xvals, ax=ax)

    ax.set_ylabel(ylabeltext)
    ax.legend('', frameon=False)
    if ylimvals is not None:
        ax.set_ylim(ylimvals)

    myAxisTheme(ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    return ax

def makeColormap(minval, maxval, usecmap):
    import matplotlib.colors as colors

    cNorm = colors.Normalize(vmin=minval,vmax=maxval)
    myCmap = plt.cm.ScalarMappable(norm=cNorm, cmap=usecmap)
    return myCmap


def slicedDistributionPlot(duration, velo, seg, cond, ax1, ax2, mycmap1, mycmap2, histrange1, histrange2, nbins1, nbins2, minpts=100):
    bins1 = np.linspace(histrange1[0], histrange1[1], nbins1)
    bins2 = np.linspace(histrange2[0], histrange2[1], nbins2)

    # 2D histogram
    H, xed, yed = np.histogram2d(duration, velo, bins=[nbins1, nbins2],
                     range=[[histrange1[0],histrange1[1]],[histrange2[0],histrange2[1]]], density=False)
    X, Y = np.meshgrid(xed[:-1]+np.diff(xed[:2])/2, yed[:-1]+np.diff(yed[:2])/2)

    for k,sp in enumerate(np.arange(0,nbins2)):
        if np.sum(H[:,sp]) >= minpts:
            ax1.plot(H[:,sp]/np.sum(H[:,sp]),color=mycmap2.to_rgba(k),label='>={}mm/s'.format(round(bins2[sp],3)))

    for k,le in enumerate(np.arange(0,nbins1)):
        if np.sum(H[le,:]) >= minpts:
            ax2.plot(H[le,:]/np.sum(H[le,:]),color=mycmap1.to_rgba(k),label='>={}mm'.format(round(bins1[le],3)))

    return bins1,bins2

def slicedDistributionAxes(axs, nbins, bins, xlab, ylimval, skipn):
    for ax in axs:
        ax.set_xlabel(xlab)
        ax.set_ylabel('frequency')
        ax.set_ylim(ylimval)
        ax.set_xlim(0,nbins)
        plt.sca(ax)
        plt.xticks(np.arange(0,nbins,skipn),bins[np.arange(0,nbins,skipn)])
        labels = [item.get_text() for item in ax.get_xticklabels()]
        # Beat them into submission and set them back again
        ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
        sns.despine(ax=ax, offset=5, trim=True)
    axs[0].legend(frameon=False)


# Per-trip viz
# function for plotting
def plotGMM(data, bestModel, infoOI, ax, xname, yname, title, alphavals=0.5, numbins = 100, linecolors=['c','b']):
    nvec, binvec, patches = ax.hist(data,numbins, density = True, alpha = alphavals, histtype = 'bar', color='grey')
    xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 1000)
    best_fit_line = np.exp(bestModel.score_samples(xscan.reshape(-1, 1)))
    pdf_individual = bestModel.predict_proba(xscan.reshape(-1, 1)) * best_fit_line[:, np.newaxis]
    ax.plot(xscan, best_fit_line,linewidth = 1, linestyle='dashed',color='dimgrey')
    order = np.argsort(np.squeeze(bestModel.means_))
    for i, o in enumerate(order):
        ax.plot(xscan, pdf_individual[:,o], color=linecolors[i], linewidth = 1)
    ax.text(np.min(data),np.max(nvec)-0.2,infoOI)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(title)
    return ax

# function for comparing distribution of a quantity of interest
def plot_trip_distribution_fit(df_all, qOI, condnames_all, otherparams, numbins = 100, iflog = False, 
                               colwidth = 5, rowwidth = 5, alphaval = 0.5, linecols=['c','b']):
    xname = qOI
    if iflog == True: xname = "log10(" + xname +")"
    yname = 'P'

    fig, ax = plt.subplots(1,len(condnames_all), figsize=(len(condnames_all)*colwidth, 1*rowwidth))

    for condIndx, cond in enumerate(condnames_all):
            rel_df = df_all.loc[df_all.condition == cond]
            if len(rel_df)<=0: continue
            
            qoIvec = rel_df[qOI].values
            qoIvec = qoIvec[~np.isnan(qoIvec)]
            data = qoIvec
            
            if iflog == True: data = np.log10(data[data>0])
    
            M_best, infoOI = fitGMM(data.reshape(-1,1),otherparams['Nmin'],otherparams['Nmax'])

            if len(condnames_all) > 1:
                currax = plotGMM(data, M_best, infoOI, ax[condIndx], xname, yname, cond, alphaval, numbins, linecolors=linecols)
            else:
                currax = plotGMM(data, M_best, infoOI, ax, xname, yname, cond, alphaval, numbins, linecolors=linecols)
            myAxisTheme(currax)
    return fig, M_best  
