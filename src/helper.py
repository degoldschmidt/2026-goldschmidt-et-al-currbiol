import numpy as np

# General purpose
def rle(inarray, dt=None):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray, dtype=np.int32)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions

        if dt is None:
            return(z, p, ia[i]) # simply return array runlengths
        else:
            try:
                dt = np.array(dt)   # force numpy
                l = np.zeros(z.shape) ## real time durations
                for j,_ in enumerate(p[:-1]):
                    l[j] = np.sum(dt[p[j]:p[j+1]])
                l[-1] = np.sum(dt[p[-1]:]) ## length of last segment
                return(z, p, ia[i], l) # return array runlengths & real time durations
            except TypeError:
                print('Your array is invalid')

# File IO
def read_yaml(_file):
    import io, yaml
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out


# Related to turn analysis
def computeTurnAng(vec1,vec2):
    turnSize = np.arctan2(vec2[1],vec2[0]) - np.arctan2(vec1[1],vec1[0])
    if turnSize > np.pi:
        turnSize =  -(2*np.pi - turnSize)
    elif turnSize <= -np.pi:
        turnSize = turnSize + 2*np.pi
    return turnSize


def computeReorientationAng(vBefore,vAfter,vObjBefore,vObjAfter):
    relOriBefore = computeTurnAng(vBefore, vObjBefore) 
    relOriAfter = computeTurnAng(vAfter, vObjAfter)
    return relOriBefore, relOriAfter


def gethiscounts(vals, nbins, histrange, densityFlag=True, returnBincenters=False):
    h, ed = np.histogram(vals, bins=nbins, range=histrange, density=densityFlag)
    if returnBincenters:
        bincent = ed[:-1]+np.mean(np.diff(ed))/2.
        return h, bincent
    else:
        return h