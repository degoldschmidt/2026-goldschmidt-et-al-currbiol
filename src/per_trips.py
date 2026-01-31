import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.helper import rle

def get_trip_df(data, metadata):
    """
    Returns a DataFrame with values of every trip (every segment after a feeding visit, i.e., loops and leaving).

    args:
    - data: per-frame data of a given experiment
    - metadata: corresponding metadata from Yaml file.
    """

    ### Output DataFrame column definitions
    per_trip_df = {'fly': [],
                   'flyid': [],
                   'condition': [],
                   'trip_index': [], # trip index (since encountering patch)
                   'time': [], # time at the start of trip
                   'cumul_feeding': [], # cumulative feeding at start of trip
                   'norm_cumul_feeding': [], # normalized cumulative feeding at start of trip
                   #'cumul_feeding_frac': [],  # cumulative feeding fraction at start of trip (fraction of time feeding since start of expt)
                   'duration': [], # duration of trip
                   'length': [], # path length of trip
                   'runturnsegs': [], # number of etho segments in the trip that are runs or turns
                   #'diff_cff': [], # change in CFF during trip
                   #'diff_rel_cff': [], # change in CFF during trip  (relative to CFF at start of trip)
                   #'max_distance': [], # maximum displacement from center of food patch
                   #'max_disp_prevtrips': [],
                   #'max_disp_sinceboundary': [], # maximum displacement from center of food patch across previous trips (since last encounter with arena boundary)
                   #'numtrips_sinceboundary': [], # number of trips made previously since last encounter with arena boundary
                   'previsit_duration': [], # duration of previous visit
                   #'previsit_cff': [], # CFF at the start of previous visit
                   #'previsit_diff_cff': [], # increase in CFF during previous visit
                   #'previsit_diff_rel_cff': [], # increase in CFF (relative to max possible increase) during previous visit
                   #'previsit_diff_cf': [], # increase in CF during previous visit
                   'isloop': [], # if this is a loop
                   #'ratio_dist2disp':[], # trip distance over maximum displacement
                   #'ratio_dist2duration':[], # trip distance over trip duration (average speed during trip)
                   'postvisit_duration': [], # if trip is loop, how long is next visit

                   'scaled_duration':[], # trip duration divided by mean trip duration (per fly)
                   'scaled_length':[], # trip distance divided by mean trip distance (per fly)
                   'scaled_duration_loop':[], # trip duration divided by mean loop duration (per fly), for loop-only analysis
                   'scaled_length_loop':[], # trip distance divided by mean loop distance (per fly), for loop-only analysis
    }
    per_trip_df = pd.DataFrame(per_trip_df)
    ### loop through conditions
    for condition in data.condition.unique():

        ### select flies of condition
        cdf = data.loc[data.condition==condition]
        flyidlookup = {(list(cdf.fly.unique()))[i]: 'fly{}'.format(str(i+1).zfill(2)) for i in range(len(list(cdf.fly.unique())))}

        """
        Below are calculations per condition
        """
        # relevant df vectors
        #segment = cdf['segment'].values
        #dt  = cdf['dt'].values
        # run length encoding of segments vector
        #runlen, pos, state, durations = rle(segment, dt=dt)
        # select all trips
        #loop_durs = durations[state==2]
        #loop_thr = extractGaussianIntersects(np.log10(loop_durs[loop_durs>0]).reshape(-1,1))

        ### loop through flies
        for f, fly in enumerate(tqdm(cdf.fly.unique())):
            flydf = cdf.loc[cdf.fly==fly]
            per_trip_fly_df = pd.DataFrame()

            """
            Some extra per-frame calculations
            """
            dt = flydf['dt'].values
            time = np.cumsum(dt)
            # segments vector (0 = border, 1 = feeding visit, 2 = loop [visit -> visit], 3 = finding [border -> visit],
            #                  4 = leaving [visit -> border], 5 = missing [border -> border])
            segment = flydf['segment'].values
            # ethogram vector (0 = border, 1 = turn, 2 = run, 3 = feeding, 4 = other)
            etho = flydf['ethogram'].values
            # centroid trajectory to calculate displacements
            body_x, body_y = flydf['body_x'].values, flydf['body_y'].values
            displacements = np.nancumsum(np.append(0,np.diff(body_x)**2 + np.diff(body_y)**2))
            # feeding mask
            feeding = (etho==3)
            # cumulative feeding
            cumul_feeding = np.cumsum(dt*feeding)
            # normalized feeding
            norm_cumul_feeding = np.cumsum(dt*feeding)/np.sum(dt*feeding)
            # trip mask vector (trip := non-feeding visit segment after a visit with excluding the final segment)
            trip_mask = (segment!=1)&(cumul_feeding>0)&(cumul_feeding<cumul_feeding[-1])

            # run length encoding of trips vector
            _, segpos, segstate, _ = rle(segment, dt=dt)
            runlen, pos, state, duration = rle(trip_mask, dt=dt)

            per_fly_seg_df = {'trip_index': [], 'time': [], 'cumul_feeding': [], 'norm_cumul_feeding': [],
                              'duration': [], 'length': [], 'runturnsegs': [],
                              'previsit_duration': [], 'isloop': [], 'postvisit_duration': []}

            mask = (state==1)
            for i, (j, start, rlen, dur) in enumerate(zip(np.arange(len(pos))[mask], pos[mask], runlen[mask], duration[mask])):
                if start+rlen==len(displacements): ## Don't take a trip that ends assay
                    continue
                per_fly_seg_df['trip_index'].append(i)
                per_fly_seg_df['time'].append(time[start-1])
                per_fly_seg_df['isloop'].append(segment[start]==2)
                per_fly_seg_df['duration'].append(dur)
                _, _, trip_etho_states, _ = rle(etho[start:start+rlen], dt=dt)
                numrunturns = ((trip_etho_states[:-1] == 1) & (trip_etho_states[1:] == 2)).sum()
                per_fly_seg_df['runturnsegs'].append(numrunturns)
                per_fly_seg_df['length'].append(displacements[start+rlen] - displacements[start])
                per_fly_seg_df['previsit_duration'].append(duration[j-1])
                if segment[start]==2:
                    per_fly_seg_df['postvisit_duration'].append(duration[j+1])
                else:
                    per_fly_seg_df['postvisit_duration'].append(np.nan)
                per_fly_seg_df['cumul_feeding'].append(cumul_feeding[start])
                per_fly_seg_df['norm_cumul_feeding'].append(norm_cumul_feeding[start])

            per_fly_seg_df = pd.DataFrame(per_fly_seg_df)
            per_fly_seg_df['fly'] = fly
            per_fly_seg_df['flyid'] = flyidlookup[fly]

            per_fly_seg_df['condition'] = condition
            per_trip_fly_df = pd.concat((per_trip_fly_df,per_fly_seg_df))

            # add scaled trip duration and distance (normalization per fly):
            per_trip_fly_df['scaled_duration'] = per_trip_fly_df['duration']/np.mean(per_trip_fly_df['duration'])
            per_trip_fly_df['scaled_length'] = per_trip_fly_df['length']/np.mean(per_trip_fly_df['length'])

            per_trip_fly_df['scaled_previsit_duration'] = per_trip_fly_df['previsit_duration']/np.mean(per_trip_fly_df['previsit_duration'])
            per_trip_fly_df['scaled_postvisit_duration'] = per_trip_fly_df['postvisit_duration']/np.mean(per_trip_fly_df['postvisit_duration'])

            per_trip_fly_df['scaled_duration_loop'] = per_trip_fly_df['duration']/np.mean(per_trip_fly_df.query('isloop==True')['duration'])
            per_trip_fly_df['scaled_length_loop'] = per_trip_fly_df['length']/np.mean(per_trip_fly_df.query('isloop==True')['length'])

            per_trip_df = pd.concat((per_trip_df,per_trip_fly_df))

    return per_trip_df



# Function to extract intersects between 2 individual Gaussians in GMM
def extractGaussianIntersects(data):
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(2).fit(data)
    xscan = np.linspace(np.amin(data)-0.2, np.amax(data)+0.2, 2000)
    logprob = model.score_samples(xscan.reshape(-1, 1))
    responsibilities = model.predict_proba(xscan.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    pdf_diff = pdf_individual[:,0] - pdf_individual[:,1]
    signdiff = np.sign(pdf_diff)
    transitionIndx = np.where(signdiff[0:-1] != signdiff[1:])[0]
    xthres = (xscan[transitionIndx] + xscan[transitionIndx+1])/2
    return xthres


# function for fitting data to a Gaussian mixture model
def fitGMM(data, Nmin, Nmax):
    from sklearn.mixture import GaussianMixture

    # fit models with 1-5 components
    N = np.arange(Nmin, Nmax+1)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(data)

    # compute the AIC and the BIC
    AIC = [m.aic(data) for m in models]
    BIC = [m.bic(data) for m in models]

    # extract best model
    M_best = models[np.min([np.argmin(AIC),np.argmin(BIC)])]

    Nopt = len(M_best.weights_)

    if Nopt == 1:
        infoOI = 'mu = ' + str(round(M_best.means_[0][0],2)) + ', sd = ' + str(round(np.sqrt(M_best.covariances_[0][0][0]),2))
    else:
        infoOI = ''
        order = np.argsort(np.squeeze(M_best.means_))
        for k in range(Nopt):
            infoOI = infoOI + ('mu = ' + str(round(M_best.means_[k][0],2)) +
                                ', sd = ' + str(round(np.sqrt(M_best.covariances_[k][0][0]),2)) +
                                ', w = ' + str(round(M_best.weights_[k],2)) + '\n')
    return M_best, infoOI
