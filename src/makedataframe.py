import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from os.path import isfile, sep
from src.helper import rle, read_yaml, computeTurnAng

# useful definitions
from src.viz import seg_label, etho_label

### define ethogram states
state_border = 0
state_turn = 1
state_run = 2
state_feeding = 3
state_other = 4

### define segment states
state_border = 0 ## same as in ethogram
state_visit = 1
state_loop = 2
state_finding = 3
state_leaving = 4
state_missing = 5

nseg = len(seg_label)
netho = len(etho_label)

# utilities
def getDist(x,y, start, length):
    dist = np.hypot(x[start+length]-x[start], y[start+length]-y[start])
    return dist

def makeconddf(file, rootDir):
    condDf = pd.read_feather(rootDir + sep + file + '.feather', columns=None)
    condDf['place'] = file.split('_')[0]
    condDf['food'] = file.split('_')[-2]
    condDf['light'] = file.split('_')[2]
    condDf['starvation'] = file.split('_')[-1]
    return condDf

def addExpInfo(df, flyid, condition, place, light, food):
    df['fly'] = flyid
    df['condition'] = condition
    df['place'] = place
    df['light'] = light
    df['food'] = food
    return df

def getMovementParams(df,segstart, segend):
    posX = df.body_x.values[segstart:segend]
    posY = df.body_y.values[segstart:segend]
    segType = df.segment.values[segstart:segend]
    cspath = np.cumsum( np.hypot( np.hstack((0, np.diff(posX) )),  np.hstack((0, np.diff(posY) )) ) )
    velo = ( np.hypot( np.hstack((0, np.diff(posX) )), np.hstack((0, np.diff(posY) )) ) )/df.dt[segstart:segend]
    return posX,posY,segType,cspath,velo


# get all per-frame data
def getDataPerGroup(rootDir, files):
    datDf = pd.DataFrame()
    metadata = {}
    for i, file in enumerate(tqdm(files)):
        tmp = pd.read_feather(rootDir + sep + file + '.feather', columns=None)
        flyidlookup = {(list(np.unique(tmp.fly.values)))[i]: 'fly{}'.format(str(i+1).zfill(2)) for i in range(len(list(np.unique(tmp.fly.values))))}
        tmp['place'] = file.split('_')[0]
        tmp['light'] = file.split('_')[2]
        tmp['food'] = file.split('_')[4]
        tmp['starvation'] = file.split('_')[5]
        metadat = read_yaml(rootDir + sep + file + '.yaml')
        metadata.update(metadat)
        for f, fly in enumerate(tqdm(list(np.unique(tmp.fly.values)), leave=True)):
            tmpfly = tmp.query('fly == "{}"'.format(fly))
            arenaRad = metadat[fly]['arena']['radius']/metadat[fly]['px_per_mm']
            disp = np.hypot(np.diff(tmpfly.body_x.values, prepend=0),np.diff(tmpfly.body_y.values,prepend=0))
            tmpfly['displacement'] = disp
            tmpfly['arenaRad'] = arenaRad
            try:
                tmpfly['food_x'] = metadat[fly]['arena']['spots']['x']/metadat[fly]['px_per_mm']
                tmpfly['food_y'] = metadat[fly]['arena']['spots']['y']/metadat[fly]['px_per_mm']
                tmpfly['food_r'] = metadat[fly]['arena']['spots']['radius']/metadat[fly]['px_per_mm']
            except TypeError: # deal with list format
                tmpfly['food_x'] = metadat[fly]['arena']['spots'][0]['x']/metadat[fly]['px_per_mm']
                tmpfly['food_y'] = metadat[fly]['arena']['spots'][0]['y']/metadat[fly]['px_per_mm']
                tmpfly['food_r'] = metadat[fly]['arena']['spots'][0]['radius']/metadat[fly]['px_per_mm']
            tmpfly['distance_patch_0'] = np.hypot( (tmpfly.body_x.values - tmpfly.food_x.values), (tmpfly.body_y.values - tmpfly.food_y.values) )
            tmpfly['flyid'] = flyidlookup[fly]
            datDf = pd.concat([datDf,tmpfly])
            datDf = datDf.reset_index(drop=True)
    datDf = datDf.reset_index(drop=True)

    datDf = datDf.assign(is_feeding = lambda dataframe: dataframe['ethogram'].map(lambda ethogram: 1 if ethogram == 3 else 0) )
    datDf['fed'] = datDf.groupby(['condition','fly'],group_keys=False)['is_feeding'].apply(lambda x: np.cumsum(x)).reset_index().rename(columns={'is_feeding':'fed'})['fed']
    datDf['fed'] = datDf.fed * datDf.dt.values
    with np.errstate(divide='ignore'):
        datDf['cff'] = np.divide(datDf.fed.values, datDf.time.values)
    datDf.cff = datDf.cff.replace(np.nan, 0)

    return datDf, metadata

# Loop dataframes

def makeLoopDf(datDf):
    loopDf1 = datDf.copy()[['condition','flyid','fly','body_x','body_y','ethogram','segment','frame','distance_patch_0','arenaRad','food_r',
                            'displacement','dt', 'cff','time']]
    # mark loops
    loopDf1 = loopDf1.assign(is_loop = lambda dataframe: dataframe['segment'].map(lambda segment: 1 if segment == 2 else 0) )
    tmp1 = loopDf1.is_loop.values
    loopDf1['loop_start'] = (np.diff(tmp1, prepend=0) > 0).astype(int)
    loopDf1['loopID'] = loopDf1.loop_start.cumsum()

    # mark feeding
    loopDf1 = loopDf1.assign(is_feeding = lambda dataframe: dataframe['ethogram'].map(lambda ethogram: 1 if ethogram == 3 else 0) ).reset_index(drop=True)
    #loopDf1.is_feeding[np.isnan(loopDf1.is_feeding)] = 0
    loopDf1.loc[np.isnan(loopDf1.is_feeding), "is_feeding"] = 0

    # mark visits
    loopDf1 = loopDf1.assign(is_visit = lambda dataframe: dataframe['segment'].map(lambda segment: 1 if segment == 1 else 0) )
    loopDf1['visit_start'] = (np.diff(loopDf1.is_visit.values, prepend=0) > 0).astype(int)
    loopDf1['visit_end'] = (np.diff(loopDf1.is_visit.values, prepend=0) < 0).astype(int)
    loopDf1['loopID'] = loopDf1.loop_start.cumsum()

    # loop over conditions and flies
    loopDf = pd.DataFrame()
    loopDepDf = pd.DataFrame()
    conditions = list(np.unique(datDf.condition.values))
    flyids = list(np.unique(datDf.flyid.values))
    for c, cond in enumerate(conditions):
        for f, flyid in enumerate(flyids):
            tmp = loopDf1.query(f'flyid == "{flyid}" & condition == "{cond}"').reset_index(drop=True)

            if len(tmp) == 0: continue
            tmp['loopN'] = tmp.loop_start.values.cumsum()
            try:
                startTime = tmp.query('loop_start>0').time.values[0]
            except IndexError:
                continue
            #feeding variabls
            tmp['fed'] = (tmp.is_feeding.values*np.median(tmp.dt.values)).cumsum()
            tmp['normFed'] = (tmp.fed.values/max(tmp.fed.values)).round(3)

            #compute change in cummulative feeding fraction during visits
            vistmp = tmp.query('visit_end == 1')[['condition','flyid','frame']].reset_index(drop=True)
            if len(vistmp) > 0:
                try: vistmp['deltaCFF'] = (tmp.loc[tmp['visit_end'] == 1, 'cff'].values - tmp.loc[tmp['visit_start'] == 1, 'cff'].values)
                except ValueError:
                    if len(tmp.loc[tmp['visit_end'] == 1, 'cff'].values) == (len(tmp.loc[tmp['visit_start'] == 1, 'cff'].values) - 1):
                        #last visit not ending in trial
                        vistmp['deltaCFF'] = (tmp.loc[tmp['visit_end'] == 1, 'cff'].values - tmp.loc[tmp['visit_start'] == 1, 'cff'].values[:-1])

                    elif len(tmp.loc[tmp['visit_end'] == 1, 'cff'].values) == (len(tmp.loc[tmp['visit_start'] == 1, 'cff'].values) + 1):
                        #first visit started before trial
                        dcff = np.append(np.nan, (tmp.loc[tmp['visit_end'] == 1, 'cff'].values[1:] - tmp.loc[tmp['visit_start'] == 1, 'cff'].values))
                        vistmp['deltaCFF'] = dcff
                    else:
                        # not sure what's going on, we will not attempt to compute ∆CFF for this fly
                        print(f'something strange is happening here: {cond}, {flyid}')
                        vistmp['deltaCFF'] = np.nan

                vistmp = vistmp.merge(tmp.query('visit_end == 1').reset_index(drop=True),
                    left_on=['condition','flyid','frame'],right_on=['condition','flyid','frame'], how='left')

                #keep one dataframe with departures included
                loopDepDf = pd.concat([loopDepDf, vistmp.query('segment in [2,4]').reset_index(drop=True)])

            # now focus on loopsx
            tmp = tmp.query('segment==2').reset_index(drop=True)
            if len(tmp) == 0:
                print('No loops in this experiment: {}, {}'.format(cond, flyid))
                loopDf = pd.concat([loopDf, tmp])
                continue

            for l in range(tmp.loopN.values[-1]+1):
                loop = tmp.query(f'loopN == {l}')
                if len(loop) == 0: continue

                loop['loop_pathL'] = np.cumsum( loop.displacement )
                loop['loop_frame'] = loop.frame - loop.frame.values[0]
                loop['loop_time'] = loop.time - startTime
                loop['loop_normT'] = loop.loop_frame/max(loop.loop_frame)
                loop['loop_normP'] = loop.loop_pathL/max(loop.loop_pathL)
                loop['loop_length'] = np.max(loop.loop_pathL.values)
                loop['deltaCFF'] = float(vistmp.query(f'loopN == {l} & segment==2').iloc[0].deltaCFF) #asssuming this is always single element series

                loopDf = pd.concat([loopDf, loop])
    return loopDf, loopDepDf


def makeLoopDepartureStatsDf(datDf):
    # Construct dataframe that also contains departure segments
    loopdepStats1 = datDf.copy()[['condition','fly','flyid','segment','ethogram','distance_patch_0','displacement','dt']].reset_index(drop=True)
    loopdepStats1 = loopdepStats1.assign(is_feeding = lambda dataframe: dataframe['ethogram'].map(lambda ethogram: 1 if ethogram == 3 else 0) )
    loopdepStats = pd.DataFrame()
    conditions = list(np.unique(datDf.condition.values))
    flyids = list(np.unique(datDf.flyid.values))
    for c, cond in enumerate(conditions):
        for f, flyid in enumerate(flyids):
            tmp = loopdepStats1.query('flyid == "{}" & condition == "{}"'.format(flyid, cond))
            if len(tmp) == 0: continue

            tmp['fed'] = (tmp.is_feeding*tmp.dt).cumsum()
            #add start time...
            tmp['normFed'] = (tmp.fed/max(tmp.fed)).round(3)

            tmp['changeSeg'] = (np.diff(tmp.segment, prepend=0) != 0).astype(int)
            tmp = tmp.query('segment in [2,4]').reset_index(drop=True)

            if len(tmp) == 0: continue

            tmp['segID'] = np.cumsum(tmp.changeSeg)

            for s, seg in enumerate(list(np.unique(tmp.segID.values))):
                segdf = tmp.query('segID == {}'.format(seg))
                if len(segdf) == 0: continue

                segdf['cumpathlength'] = segdf.displacement.cumsum().values
                segdf['maxDist'] = max(segdf.distance_patch_0)
                segdf['pathLength'] = max(segdf.cumpathlength.values)
                segdf['pathLengthDistanceRatio'] = segdf.pathLength/segdf.maxDist

                loopdepStats = pd.concat([loopdepStats, segdf])

    loopdepStats = loopdepStats.query('changeSeg == True').reset_index(drop=True)
    return loopdepStats


# Make dataframe for run statistics
def makerundf(files,rootDir, addTurnDuration=True):
    runDf = pd.DataFrame()

    for file in files:
        condDf = makeconddf(file, rootDir)
        cond = np.unique(condDf.condition.values)[0]

        for fly, flyID in enumerate(list(np.unique(condDf.fly.values))):
            flyDf = condDf.query('fly == "{}"'.format(flyID))
            segL, segSt, segType = rle(flyDf.segment)

            # compute cummulated feeding
            cumfeeding = np.cumsum( (flyDf.ethogram.values == state_feeding).astype('int')*flyDf.dt.values )
            if cumfeeding[-1] > 0: cumfeeding = cumfeeding/cumfeeding[-1]

            # initialize run dataframe per fly
            runDf_fly = pd.DataFrame()

            # itterate over segments
            for i, ss in enumerate(segSt):
                # initialize run dataframe per segment
                runDf_seg = pd.DataFrame()

                se = min(ss+segL[i]+1, len(flyDf.body_x.values)) #find segment end
                ethoL, ethoSt, ethoType, ethoDuration = rle(flyDf.ethogram[ss:se], flyDf.dt[ss:se])

                # get movement related parameters
                posX,posY,segType,cspath,velo = getMovementParams(flyDf,ss, se)

                # find run segments
                runSegs = np.where(ethoType==state_run)[0]
                if len(runSegs) == 0: continue
                # start indices
                runDf_seg['start'] = ss + ethoSt[runSegs]

                # get mean velocity during run
                runDf_seg['velo'] = [np.mean(velo[ethoSt[rs]: ethoSt[rs] + ethoL[rs]]) for rs in runSegs]
                runDf_seg['stdvelo'] = [np.std(velo[ethoSt[rs]: ethoSt[rs] + ethoL[rs]]) for rs in runSegs]

                # position at the start of the run
                runDf_seg['body_x'] = posX[ethoSt[runSegs]]
                runDf_seg['body_y'] = posY[ethoSt[runSegs]]
                runDf_seg['segment'] = segType[ethoSt[runSegs]]

                # following segment
                runDf_seg['ethoAfter'] = ethoType[runSegs] + 1
                # add feeding
                runDf_seg['feeding'] = cumfeeding[ss + ethoSt[runSegs]]


                ends = ss + ethoSt[runSegs] + ethoL[runSegs]
                lengths = cspath[ethoSt[runSegs] + ethoL[runSegs]-1]-cspath[ethoSt[runSegs]]
                distances = getDist(posX, posY, ethoSt[runSegs], ethoL[runSegs]-1)
                durations = ethoDuration[runSegs]

                runDf_seg['end'] = ends
                runDf_seg['length'] = lengths
                runDf_seg['distance'] = distances
                runDf_seg['duration'] = durations

                # check if the following segment is a turn, if yes, then add duration of the turn
                if addTurnDuration:
                    for k, rs in enumerate(runSegs):
                        if rs+1 < len(ethoL):
                            if ethoType[rs+1] == state_turn:
                                ends[k] = ss + ethoSt[rs] + ethoL[rs] + ethoL[rs+1]
                                lengths[k] = cspath[ ethoSt[rs] + ethoL[rs]-1 + ethoL[rs+1]-1 ]  -  cspath[ ethoSt[rs] ]
                                distances[k] = getDist(posX, posY, ethoSt[rs], ethoL[rs]+ethoL[rs+1]-1)
                                durations[k] = ethoDuration[rs] + ethoDuration[rs+1]

                runDf_seg['end_wt'] = ends
                runDf_seg['length_wt'] = lengths
                runDf_seg['distance_wt'] = distances
                runDf_seg['duration_wt'] = durations

                # Add info about the experiment
                runDf_seg = addExpInfo(runDf_seg, 'fly'+str(fly+1).zfill(2), cond, flyDf['place'].values[0], flyDf['light'].values[0], flyDf['food'].values[0])
                runDf_fly = pd.concat([runDf_fly,runDf_seg])

            runDf = pd.concat([runDf,runDf_fly])

    return runDf


# Make dataframe for turn and statistics
## TODO:
## * add turn duration & distance to run duration & distance

def maketurndf(files,rootDir):
    turnDf = pd.DataFrame()

    for file in files:
        condDf = makeconddf(file, rootDir)
        cond = np.unique(condDf.condition.values)[0]
        metadat = read_yaml(rootDir + sep + file + '.yaml')

        for fly, flyID in enumerate(list(np.unique(condDf.fly.values))):
            flyDf = condDf.query('fly == "{}" '.format(flyID))

            totalPathLen = np.nancumsum(np.hypot(np.diff(flyDf.body_x),np.diff(flyDf.body_y)))

            foodx = metadat[flyID]['arena']['spots']['x']/metadat[flyID]['px_per_mm']
            foody = metadat[flyID]['arena']['spots']['y']/metadat[flyID]['px_per_mm']
            dist2food = np.hypot( (flyDf.body_x.values - foodx), (flyDf.body_y.values - foody) )

            # compute cummulated feeding
            cumfeeding = np.cumsum( (flyDf.ethogram.values == state_feeding).astype('int')*flyDf.dt.values )
            if cumfeeding[-1] > 0: cumfeeding = cumfeeding/cumfeeding[-1]

            segL, segSt, segType = rle(flyDf.segment)
            turnDf_fly = pd.DataFrame()
            for i, ss in enumerate(segSt):
                turnDf_seg = pd.DataFrame()
                se = min(ss+segL[i]+1, len(flyDf.body_x.values))
                angleSeg = flyDf.angle.values[ss:se]

                ethoL, ethoSt, ethoType, ethoDuration = rle(flyDf.ethogram.iloc[ss:se], flyDf.dt.iloc[ss:se])
                turnSegs = np.where(ethoType==state_turn)[0]

                if len(turnSegs) == 0: continue

                turnDf_seg['start'] = ss + ethoSt[turnSegs]
                turnDf_seg['end'] = ss + ethoSt[turnSegs] + ethoL[turnSegs]
                turnDf_seg['turnframes'] = ethoL[turnSegs]
                turnDf_seg['turnpathdist'] = totalPathLen[ss + ethoSt[turnSegs] + ethoL[turnSegs]] - totalPathLen[ss + ethoSt[turnSegs]]
                turnDf_seg['turnduration'] = ethoDuration[turnSegs]

                turnDf_seg['turnfooddist'] =dist2food[ss + ethoSt[turnSegs]]

                turnDf_seg['segment'] = segType[i]
                turnDf_seg['condition'] = cond

                # add feeding
                turnDf_seg['feeding'] = cumfeeding[ss + ethoSt[turnSegs]]

                # get turn size and check surrounding segments
                turnSize =  np.nan*np.ones(len(turnSegs))
                prevturnSize =  np.nan*np.ones(len(turnSegs))
                prevRunLen =  np.nan*np.ones(len(turnSegs))
                prevRunDur =  np.nan*np.ones(len(turnSegs))
                nextRunLen =  np.nan*np.ones(len(turnSegs))
                nextRunDur =  np.nan*np.ones(len(turnSegs))
                for t, ts in enumerate(turnSegs):
                    tss = ethoSt[ts]
                    tse = min((ethoSt[ts]+ethoL[ts])+1, len(angleSeg)-1)

                    # compute turn size
                    turnSize[t] = computeTurnAng( (np.cos(angleSeg[tss]),np.sin(angleSeg[tss])) ,
                                                  (np.cos(angleSeg[tse]),np.sin(angleSeg[tse])) )

                    #look for previous run
                    if (ts>0):
                        if (ethoType[ts-1] == state_run):
                            prevRunLen[t] = totalPathLen[ss + ethoSt[ts-1] + ethoL[ts-1]] - totalPathLen[ss + ethoSt[ts-1]]
                            prevRunDur[t] = ethoDuration[ts-1]
                        elif (ethoType[ts-1] == state_other) & (ts>1):
                            if (ethoType[ts-2] == state_run):
                                prevRunLen[t] = totalPathLen[ss + ethoSt[ts-2] + ethoL[ts-2]] - totalPathLen[ss + ethoSt[ts-2]]
                                prevRunDur[t] = ethoDuration[ts-2]

                    #look for next run
                    if (ts < (len(ethoType)-1)):
                        if(ethoType[ts+1] == state_run):
                            rse = min(ss + ethoSt[ts+1] + ethoL[ts+1], len(totalPathLen)-1)
                            nextRunLen[t] = totalPathLen[rse] - totalPathLen[ss + ethoSt[ts+1]]
                            nextRunDur[t] = ethoDuration[ts+1]
                        elif (ethoType[ts+1] == state_other) & (ts < (len(ethoType)-3)):
                            if(ethoType[ts+2] == state_run):
                                rse = min(ss + ethoSt[ts+2] + ethoL[ts+2], len(totalPathLen)-1)
                                nextRunLen[t] = totalPathLen[rse] - totalPathLen[ss + ethoSt[ts+2]]
                                nextRunDur[t] = ethoDuration[ts+2]


                    #look for previous turn
                    if (ts>1):
                        if (ethoType[ts-2] == state_turn):
                            ptss = ethoSt[ts-2]
                            ptse = ethoSt[ts-2] + ethoL[ts-2] + 1
                            prevturnSize[t] = computeTurnAng( (np.cos(angleSeg[ptss]),np.sin(angleSeg[ptss])) ,
                                                  (np.cos(angleSeg[ptse]),np.sin(angleSeg[ptse])) )
                        elif (ethoType[ts-2] == state_other) & (ts>2):
                            if (ethoType[ts-3] == state_turn):
                                ptss = ethoSt[ts-3]
                                ptse = ethoSt[ts-3] + ethoL[ts-3] + 1
                                prevturnSize[t] = computeTurnAng( (np.cos(angleSeg[ptss]),np.sin(angleSeg[ptss])) ,
                                                      (np.cos(angleSeg[ptse]),np.sin(angleSeg[ptse])) )

                turnDf_seg['turnsize'] = turnSize*180/np.pi
                turnDf_seg['turndirection'] = np.sign(turnSize*180/np.pi)
                turnDf_seg['turnabssize'] = abs(turnSize*180/np.pi)

                turnDf_seg['prevturnsize'] = prevturnSize*180/np.pi
                turnDf_seg['prevturndirection'] = np.sign(prevturnSize*180/np.pi)
                turnDf_seg['prevturnabssize'] = abs(prevturnSize*180/np.pi)

                turnDf_seg['prevrunlen'] = prevRunLen
                turnDf_seg['prevrundur'] = prevRunDur
                turnDf_seg['nextrunlen'] = nextRunLen
                turnDf_seg['nextrundur'] = nextRunDur

                turnDf_fly = pd.concat([turnDf_fly,turnDf_seg])

            if len(turnDf_fly) == 0: continue
            turnDf_fly['fly'] = 'fly'+str(fly+1).zfill(2)
            turnDf = pd.concat([turnDf,turnDf_fly])

    return turnDf

# function for computing turn angle between two angles
def computeTurnAngBetween2angles(startangle,endangle):
    turnAngle = endangle - startangle
    if hasattr(turnAngle, "__len__") == False:
        if turnAngle > np.pi:
            turnAngle =  turnAngle - 2*np.pi
        elif turnAngle <= -np.pi:
            turnAngle = turnAngle + 2*np.pi
    else:
        turnAngle[turnAngle>np.pi] = turnAngle[turnAngle>np.pi] - 2*np.pi
        turnAngle[turnAngle<=-np.pi] = turnAngle[turnAngle<=-np.pi] + 2*np.pi
    return turnAngle

def makePerMoveSegmentDF(df,ethoStatsOfInterest):
    perMoveSegDf = pd.DataFrame()
    for genotype in df.genotype.unique():
        for condition in df.condition.unique():
            for flyid in df.flyid.unique():
                flyDf = df.query(f'flyid=="{flyid}" & condition=="{condition}" & genotype=="{genotype}"')
                if len(flyDf) == 0: continue

                segL, segSt, segType = rle(flyDf.segment)

                prev_visit = 0
                cumdist_curr = 0
                cumRunTime_curr = 0
                cumtime_curr = 0

                moveSegDf_fly = {
                    'seg_state': [],
                    'etho_state': [],
                    'after_which_visit': [],
                    'dist_since_visit': [], # total distance travelled during runs since last food spot visit
                    'time_since_visit': [], # time since last food spot visit
                    'cumRunTime_since_visit': [], # total run time since last food spot visit
                    'seg_duration': [],
                    'seg_length': [], # total distance travelled by fly during this segment
                    'headturnangle': [], # net heading turn angle during movement segment (body angle at end - body angle at start)
                    'absheadturnangle': [], # absolute value of the turn angle
                    'ifCW': [], # whether the turn is in CW direction
                }

                for ii, ss in enumerate(segSt):
                    segtype_curr = segType[ii]
                    if segtype_curr == 1:
                        prev_visit = prev_visit + 1
                    if (segtype_curr == 1) or (segtype_curr == 2) or (segtype_curr == 4):
                        cumdist_curr = 0
                        cumRunTime_curr = 0
                        cumtime_curr = 0 # reset time since leaving food spot

                    se = min(ss+segL[ii], len(flyDf.body_x.values))

                    ethoL, ethoSt, ethoType, ethoDuration = rle(flyDf.ethogram[ss:se], flyDf.dt[ss:se])
                    moveSegs = np.where(np.isin(ethoType,ethoStatsOfInterest))[0]
                    etho_StartTime = np.cumsum(np.insert(ethoDuration,0,0))
                    moveSegs_starttime = etho_StartTime[moveSegs]
                    moveSegs_ethoType = ethoType[moveSegs]

                    if len(moveSegs) > 0:
                        for moveIndx in range(len(moveSegs)):
                            moveSegDf_fly['seg_state'].append(segtype_curr)
                            moveSegDf_fly['etho_state'].append(moveSegs_ethoType[moveIndx])
                            moveSegDf_fly['after_which_visit'].append(prev_visit)

                            movedur = ethoDuration[moveSegs[moveIndx]]
                            moveSegDf_fly['seg_duration'].append(movedur)

                            moveSegDf_fly['cumRunTime_since_visit'].append(cumRunTime_curr)
                            if segtype_curr != 1: cumRunTime_curr = cumRunTime_curr + movedur

                            moveSegDf_fly['time_since_visit'].append(cumtime_curr + moveSegs_starttime[moveIndx])

                            # segment length (distance)
                            startframe = max(ss + ethoSt[moveSegs[moveIndx]],0)
                            endframe = min(ss + ethoSt[moveSegs[moveIndx]] + ethoL[moveSegs[moveIndx]] + 1, len(flyDf.body_x.values)) # one frame after last frame
                            xpos_all = flyDf.body_x.values[startframe:endframe]
                            ypos_all = flyDf.body_y.values[startframe:endframe]
                            dist_all = np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2)
                            totdist = np.nansum(dist_all)
                            moveSegDf_fly['seg_length'].append(totdist)
                            # store total running distance since last visit
                            moveSegDf_fly['dist_since_visit'].append(cumdist_curr)
                            if segtype_curr != 1:
                                cumdist_curr = cumdist_curr + totdist

                            # starting displacement from center of spot
                            dispVec_fromcenter = np.array([xpos_all[0]-flyDf.food_x.values[0],ypos_all[0]-flyDf.food_y.values[0]])

                            # heading angle (vector from body to head of fly):
                            if endframe - startframe > 1:
                                headingAngle_all = flyDf.angle.values[startframe:endframe]
                                headturnangle_all = computeTurnAngBetween2angles(headingAngle_all[0],headingAngle_all[1:])
                                turnAngleDir_all = np.sign(headturnangle_all)
                                changedirFrameInds = np.where(turnAngleDir_all[0:-1] != turnAngleDir_all[1:])[0]
                                numchanges = len(changedirFrameInds)
                                if numchanges == 0:
                                    headturnangle_currseg = headturnangle_all[-1]
                                else:
                                    # investigate the first direction change
                                    changeFrameIndx = changedirFrameInds[-1] # frame right before change in direction
                                    if (np.abs(headturnangle_all[changeFrameIndx]) > np.pi/2) and (np.abs(headturnangle_all[changeFrameIndx+1]) > np.pi/2):
                                        headturnangle_last = headturnangle_all[-1]
                                        headturnangle_currseg = -np.sign(headturnangle_last)*2*np.pi + headturnangle_last
                                    else:
                                        headturnangle_currseg = headturnangle_all[-1]
                                moveSegDf_fly['headturnangle'].append(headturnangle_currseg)
                                moveSegDf_fly['absheadturnangle'].append(np.abs(headturnangle_currseg))
                                ifCW_currseg = headturnangle_currseg<0
                                moveSegDf_fly['ifCW'].append(ifCW_currseg)

                            else:
                                moveSegDf_fly['headturnangle'].append(0)
                                moveSegDf_fly['absheadturnangle'].append(0)
                                moveSegDf_fly['ifCW'].append(np.nan)

                    if (segtype_curr == 4) or (segtype_curr == 5) or (segtype_curr == 0):
                        cumtime_curr = cumtime_curr + etho_StartTime[-1]

                moveSegDf_fly = pd.DataFrame(moveSegDf_fly)
                moveSegDf_fly['genotype'] = genotype
                moveSegDf_fly = addExpInfo(moveSegDf_fly, flyid, condition, 'na', 'na', 'na')
                perMoveSegDf = pd.concat([perMoveSegDf, moveSegDf_fly], sort=False)

    # Augment per-segment dataframe with other properties
    # radius of curvature of segments
    perMoveSegDf['effArcRadius'] = perMoveSegDf['seg_length']/perMoveSegDf['absheadturnangle']
    return perMoveSegDf
