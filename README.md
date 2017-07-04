# jplotter / jiveplot
Python based visualization tool for AIPS++/CASA MeasurementSet data

The jplotter command line tool allows the user to quickly visualize the
radio-astronomical data contained in a MeasurementSet (`ms`).

## What can be visualized?

Quantities that can be visualized are, e.g., amplitude-versus-time,
phase-versus-frequency, amplitude-versus-uv-distance, weight-versus-time, to
name but a few.

Some key features:
- the package focuses on powerful selection syntax
- the ability to difference whole sets of plots, e.g. to visualize before-after changes or to
compare output of different correlators
- time- or frequency averaging of the data before plotting
- plots can be saved to file (postscript).
- the current selection can be written out as a new *reference* `ms`; data
  is not copied but the newly created `ms` references rows of data in the
parent `ms`. It can be treated as a real `ms`.
- plots/data sets can be organized at will
- the data can be indexed (`> indexr`) to create a scan list, after which powerful
  scan-based selection can be used
- plotting can be scripted/play back stored commands from text file
- open/visualize multiple data sets at the same time or the same data set
  from different 'angles'

## Data selection
`ms`'s can contain several GBs of binary data. Therefore, data selection is
desirable, preferably in a fairly natural way, even without knowing the
exact details of the experiment's data.

The jplotter selection commands take a stab at suiting the needs of a radio
astronomer:

```sh
# select a time range near the end of the experiment
> time $end-1h to +2m20s

# select IF 0,1,2 with parallel hand polarizations
> fq  0-2/p
# equivalent, but would not work for XX, YY whereas the former would
> fq  0-2/rr,ll

# select sources whose name matches this
> src j(19|30)*

# select all cross baselines, remove those to stations xx and yy, but add xx-ef
> bl cross -(xx|yy)* +xx(ef)

# select 80% of the band, irrespective of how many channels
# the correlator produced
> ch 0.1*last:0.9*last

# after running indexr, extract a bit of data (trimming 1 minute from either
# end) from scans on sources matching 042* and who are longer than three minutes
> scan start+1m to end-1m where length>3m and field ~ '042*'
```

# Dependencies

The package uses the [https://github.com/casacore/python-casacore](pyrap)
Python binding to access data.

It uses pgplot to visualize (it was faster and easier than matplotlib):
[http://www.jive.eu/~verkout/ppgplot-1.4.tar.gz](Python) binding to pgplot
