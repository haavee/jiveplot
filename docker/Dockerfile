FROM kernsuite/base:4

RUN apt-get update && apt-get install --fix-missing -y git
RUN docker-apt-install python-casacore ppgplot
RUN cd /usr/local/ && git clone https://github.com/haavee/jiveplot.git && cd jiveplot && git checkout master

ENTRYPOINT ["/usr/local/jiveplot/jplotter"]
