## From: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
## (Heavily) adapted by HV to work with Python 2.6 and up
## Simplified to generate HSV (PGPLOT understands HSV as Hue-Saturation-Lightness)
## Also: I wanted the getfracs() to divide the colors nicer over the colourspace
##       so there's an assumption about how many variations of each colour are created
##       such that we can divide the "H" colour space in roughly even bits.
##       If you requested a low number of colours from the original code, they would be
##       rather close to each other
import itertools, math, colorsys
from fractions import Fraction

def getfracs(n):
    # we generate four variations / colour (=h)
    # so we want the step size to step through the colour wheel in ~ n/4 steps
    # This stepsize seems to avoid the 120 and 240 degrees H value reasonably well
    #d  = (float(n)/4) * 9.2*math.e/360.0
    ngen = math.ceil(float(n)/4)*4
    s  = 105.0  # start of colour wheel - just below red
    e  = 263.0  # end of colour wheel - just pas green
    d  = (4*(e - s))/ngen
    f0 = s
    while f0<e:
        yield f0
        f0 += d

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5) 

## Iterate the fastest over the actual colour, then by pureness and finally
## by lightness
def genhsv(n):
    for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
        for s in [Fraction(x,13) for x in range(3,9,3)]: # optionally use range
            for h in getfracs(n):
                yield (h, s, bias(v))

getflt  = lambda x: map(float, x)
gethsvs = lambda n: map(getflt, genhsv(n))
getrgb  = lambda x: colorsys.hsv_to_rgb(*x)
getncol_hsv = lambda n: list(itertools.islice(gethsvs(n), n))
getncol_rgb = lambda n: map(getrgb, getncol_hsv(n))

if __name__ == "__main__":
    l = getncol(13)
    print "Generated ",len(l)," colours"
    print map(getrgb, l)
