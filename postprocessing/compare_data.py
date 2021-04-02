from   __future__ import print_function
import numpy, plots, math, operator, copy

COPY = copy.deepcopy
# define a post-processing operation that always remembers the last data set and 
# computes statistical differences between that one and the next data set.
# After that, the new data set becomes the last one.

# Purpose: to verify that two chunks of data extraced from measurement sets that *should*
# be equal, are, in fact, equal.

# If a new type of data set is detected (new plot type) the last data set is erased,
# this data set becomes the stored one, and no output is generated

last_dataset = None
tolerance    = 1e-7

def plotar2unidict(plotar):
    rv = plots.Dict()
    rv.plotType = plotar.plotType
    # loop over all plots and datasets-within-plot
    for k in plotar.keys():
        for d in plotar[k].keys():
            # get the full data set label - we have access to all the data set's properties (FQ, SB, POL etc)
            n     = plots.join_label(k, d)
            # and make a copy of the dataset
            rv[n] = COPY( plotar[k][d] )
    return rv

# this is the function to pass to `postprocess ...`
def compare_data(plotar, ms2mappings):
    global last_dataset
    # Whatever we need to do - this can be done unconditionally
    new_dataset = plotar2unidict( plotar )
    error       = ""
    # Check if we need to do anything at all
    if last_dataset is not None and last_dataset.plotType == new_dataset.plotType:
        # OK check all common keys
        old_keys = set(last_dataset.keys())
        new_keys = set(new_dataset.keys())
        # if the sets are not equal, the data will also not compare equal!
        if old_keys == new_keys:
            # inspect x and y separately, add up all the diffs
            dx, dy = 0, 0
            for k in old_keys:
                ods = last_dataset[ k ]
                nds = new_dataset[ k ]
                dx  = numpy.add(numpy.abs( ods.xval - nds.xval ), dx)
                dy  = numpy.add(numpy.abs( ods.yval - nds.yval ), dy)
            if numpy.any( dx>abs(tolerance) ):
                print(">>> compare_data: total diffs in x exceed tolerance")
                print("    tolerance=", tolerance)
                print("    accum. dx=", dx)
                error += "Datasets mismatch in X according to tolerance. "
            if numpy.any( dy>abs(tolerance) ):
                print(">>> compare_data: total diffs in y exceed tolerance")
                print("    tolerance=", tolerance)
                print("    accum. dy=", dy)
                error = "Datasets mismatch in Y according to tolerance. "
        else:
            print(">>> compare_data: The datasets to compare have different data content?!")
            common = old_keys & new_keys
            only_o = old_keys - common
            only_n = new_keys - common
            print("    Common keys:", len(common))
            print("    Uniq in Old:", len(only_o))
            print("    Uniq in New:", len(only_n))
            error += "Datasets mismatch in content. "
            #with open('/tmp/oldkeys.txt', 'w') as f:
            #    list( map(lambda s: f.write(str(s) + '\n'), sorted(only_o)) )
            #with open('/tmp/newkeys.txt', 'w') as f:
            #    list( map(lambda s: f.write(str(s) + '\n'), sorted(only_n)) )
    # Install new dataset as new last dataset
    last_dataset = new_dataset
    if error:
        raise RuntimeError(error)
