import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.helper import rle

def get_fly_df(data, metadata):
    """
    Returns a DataFrame with values of every fly.
    
    args:
    - data: per-frame data of a given experiment
    - metadata: corresponding metadata from Yaml file.
    """
    ### Output DataFrame column definitions
    per_fly_df = {'fly': [], 
                  'flyid': [],
                  'condition': [],
                  'genotype': [],
                  ### feeding and visits
                  'total_feeding': [], ### total duration of feeding segments in seconds,
                  'total_visit': [], ### total feeding visit time in seconds,
                  'mean_visit': [], ### mean feeding visit time in seconds,
                  'number_visit': [], ### number of feeding visits,
                  ### encounters
                  'total_encounter': [], ### total feeding encounter time in seconds,
                  'mean_encounter': [], ### mean feeding encounter time in seconds,
                  'number_encounter': [], ### number of feeding encounters,
                  'rate_encounter': [], ### rate of feeding encounters in seconds (number of encounters divided by time off food),
                  ### loops
                  'total_loop': [], ### total duration of loop segments in seconds,
                  'mean_loop': [], ### mean duration of loop segments in seconds,
                  'number_loop': [], ### number of loops,
                  'max_distance_loop': [], ### mean maximum distance from food during loops in mm,
                  ### border
                  'total_border': [], ### total duration of border segments in seconds,
                  'mean_border': [], ### mean duration of border segments in seconds,
                  'number_border': [], ### number of border segments, 
                  ### search (bouts of consecutive alternating feeding-loop segments)
                  'total_search': [], ### total duration of search segments in seconds,
                  'mean_search': [], ### mean duration of search segments in seconds,
                  'number_search': [], ### number of search segments,
                  'mean_search_no_feed': [], ### mean duration of search segments without feeding in seconds,
                  'distance_search': [], ### sum of all loop distances during search in mm,
    }

    ### loop through conditions
    for condition in tqdm(data.condition.unique()):
        
        ### select flies of condition
        cdf = data.loc[data.condition==condition]
        flyidlookup = {(list(cdf.fly.unique()))[i]: 'fly{}'.format(str(i+1).zfill(2)) for i in range(len(list(cdf.fly.unique())))}
        
        ### loop through flies
        for f, fly in enumerate(tqdm(cdf.fly.unique(), leave=False)):
            flydf = cdf.loc[cdf.fly==fly]
            flyid = flyidlookup[fly]
            genotype = cdf.loc[cdf.fly==fly].genotype.unique()[0]

            spots = metadata[fly]['arena']['spots']
            if type(spots) is list:
                spot_pos = (spots[0]['x'], spots[0]['y'])
            else:
                spot_pos = (spots['x'], spots['y'])
                
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
            displacements = np.append(0,np.diff(body_x)**2 + np.diff(body_y)**2)
            # distance between head and food for encounters
            head_x, head_y = flydf['head_x'].values, flydf['head_y'].values
            dx, dy = head_x-spot_pos[0], head_y-spot_pos[1]
            distance_from_food = np.sqrt(dx*dx+dy*dy)
            encounter = (distance_from_food<3.).astype(int)
            # Schmitt trigger
            rlen_e, pos_e, st_e, dur_e = rle(encounter, dt=dt)
            for p,rl in zip(pos_e[st_e==0], rlen_e[st_e==0]):
                if p > 0 and np.all(distance_from_food[p:min(p+rl,len(encounter))]<5.):
                    encounter[p:min(p+rl,len(encounter))] = 1
            
            # feeding mask
            feeding = (etho==3) 
            # cumulative feeding
            cumul_feeding = np.cumsum(dt*feeding)
            # normalized feeding
            if cumul_feeding[-1] == 0:
                norm_cumul_feeding = cumul_feeding
            else:
                norm_cumul_feeding = np.cumsum(dt*feeding)/np.sum(dt*feeding) 
            # trip mask vector (trip := non-feeding visit segment after a visit with excluding the final segment)
            trip_mask = (segment!=1)&(cumul_feeding>0)&(cumul_feeding<cumul_feeding[-1])

            # search mask vector (search := consecutive alternating visit and loop segments after a visit with excluding the final segment)
            search_mask = np.zeros(trip_mask.shape) #(segment==1)|(segment==2)
            
            # run length encoding of segments vector
            rlen, pos, st, dur = rle(segment, dt=dt)
            for k, p in zip(np.arange(len(pos))[st==3], pos[st==3]): ### first array enumerates
                if k+2<len(pos) and k+1<len(pos):
                    if st[k+1] == 1 and st[k+2] == 2:
                        if np.any(st[k+3:]==4): ## depart
                            search_mask[pos[k+2]] = 1
                            kk = np.where(st[k+3:]==4)[0][0]
                            search_mask[pos[k+3+kk]] = -1
            search_mask = np.cumsum(search_mask)
            # run length encoding of segments vector
            rlen_search, pos_search, st_search, dur_search = rle(search_mask, dt=dt)
            
            maxdists = []
            for p,rl in zip(pos[st==2],rlen[st==2]):
                maxdists.append(np.amax(distance_from_food[p:p+rl]))

            """
            Append data to DataFrame
            """
            per_fly_df['fly'].append(fly)
            per_fly_df['flyid'].append(flyid)
            per_fly_df['condition'].append(condition)
            per_fly_df['genotype'].append(genotype)
            
            per_fly_df['total_feeding'].append(cumul_feeding[-1])

            if np.sum(st_e==1) == 0:
                per_fly_df['total_encounter'].append(0.0)
                per_fly_df['mean_encounter'].append(np.nan)
                per_fly_df['number_encounter'].append(0)
                per_fly_df['rate_encounter'].append(0.0)
            else:
                per_fly_df['total_encounter'].append(np.sum(dur_e[st_e==1]))
                per_fly_df['mean_encounter'].append(np.mean(dur_e[st_e==1]))
                per_fly_df['number_encounter'].append(np.sum(st_e==1))
                per_fly_df['rate_encounter'].append(np.sum(st_e==1)/np.nansum(dt[~feeding]))
            
            if np.sum(st==1) == 0:
                per_fly_df['total_visit'].append(0.0)
                per_fly_df['mean_visit'].append(np.nan)
                per_fly_df['number_visit'].append(0)
            else:
                per_fly_df['total_visit'].append(np.sum(dur[st==1]))
                per_fly_df['mean_visit'].append(np.mean(dur[st==1]))
                per_fly_df['number_visit'].append(np.sum(st==1))
            
            if np.sum(st==2) == 0: ### no loops
                per_fly_df['total_loop'].append(0.0)
                per_fly_df['mean_loop'].append(np.nan)
                per_fly_df['number_loop'].append(0)
                per_fly_df['max_distance_loop'].append(np.nan)
            else:
                per_fly_df['total_loop'].append(np.sum(dur[st==2]))
                per_fly_df['mean_loop'].append(np.mean(dur[st==2]))
                per_fly_df['number_loop'].append(np.sum(st==2))
                per_fly_df['max_distance_loop'].append(np.nanmax(maxdists))
            
            if np.sum(st==0) == 0:
                per_fly_df['total_border'].append(0.0)
                per_fly_df['mean_border'].append(np.nan)
                per_fly_df['number_border'].append(0)
            else:
                per_fly_df['total_border'].append(np.sum(dur[st==0]))
                per_fly_df['mean_border'].append(np.mean(dur[st==0]))
                per_fly_df['number_border'].append(np.sum(st==0))    

            if np.sum(st_search==1) == 0:
                per_fly_df['total_search'].append(0.0)
                per_fly_df['mean_search'].append(np.nan)
                per_fly_df['number_search'].append(0)
                per_fly_df['mean_search_no_feed'].append(np.nan)
                per_fly_df['distance_search'].append(0.0)
            else:
                per_fly_df['total_search'].append(np.sum(dur_search[st_search==1]))
                per_fly_df['mean_search'].append(np.mean(dur_search[st_search==1]))
                per_fly_df['number_search'].append(np.sum(st_search==1))
                summed_duration_search_no_feed = [np.sum(dt[p:p+rl][segment[p:p+rl]==2]) for p, rl in zip(pos_search[st_search==1], rlen_search[st_search==1])]
                per_fly_df['mean_search_no_feed'].append(np.mean(summed_duration_search_no_feed))
                per_fly_df['distance_search'].append(np.nansum(displacements[segment==2]))
    return pd.DataFrame(per_fly_df)