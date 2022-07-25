# -*- coding: utf-8 -*-
"""
@author: Sumin Han
"""

from bs4 import BeautifulSoup            # HTML parsing library
from lxml import html

nf= open('MapData.txt','w')              # open text file to write output

with open('kaistmap-refined.osm.xml','r') as f:      # open html file to read data
    page = f.read()


#soup = BeautifulSoup(page)
soup = BeautifulSoup(page, "lxml")
nodes = soup('node', {})      # save the name of tracks only from whole html file data
ways = soup('way', {})

print "nodes", len(nodes)
print "ways", len(ways)

node_list = []
node_reindex = []
buildings = []
forests = []

idx = 0;
usedidx = 0;
for n in nodes:
   node_list += [(idx, n['id'], n['lat'], n['lon'])]
   idx += 1

def search(mid):
   global node_reindex, usedidx
   for (idx, nid, nlat, nlon) in node_list:
      if(mid == nid):
         if len(node_reindex) > 0:
            for (ridx, pidx, mlat, mlon) in node_reindex:
               if(idx == pidx):
                  return ridx
         node_reindex += [(usedidx, idx, nlat, nlon)];
         usedidx += 1
         return usedidx-1
   return -1



bnum = 0
fnum = 0
for w in ways:
   isBuild = False
   isForest = False
   bname = '_'
   for t in w.select("tag"):
      k = t['k']
      v = t['v']
      if k == 'building':
         isBuild = True
         bnum += 1
      if k == 'natural' and v == 'wood':
         fnum += 1
         isForest = True
      
   if isBuild:
      if not w.find("nd"): continue
      bname = t['v']
      tmpstr = "b " + bname.encode('utf8')
      for nd in w.select("nd"):
         kk = search(nd['ref'])
         tmpstr = tmpstr + "\t" + str(kk);   
      tmpstr = tmpstr + "\n"
      buildings += [tmpstr]

   if isForest:
      if not w.find("nd"): continue
      tmpstr = "f " + bname.encode('utf8')
      for nd in w.select("nd"):
         kk = search(nd['ref'])
         tmpstr = tmpstr + "\t" + str(kk);   
      tmpstr = tmpstr + "\n"
      forests += [tmpstr]
      

print "There are ", len(node_reindex), "/", len(node_list), " ", bnum, "buildings and ", fnum, "forests."
nf.write("i\t" + str(len(node_reindex)) + "\t" + str(len(buildings)) + "\t" + str(len(forests)) + "\n")
nf.write("# nodes\n")
for (ridx, idx, mlat, mlon) in node_reindex:
   nf.write("n\t" + str(mlat) + "\t" + str(mlon) + "\n")
nf.write("# buildings\n")
for btext in buildings:
   nf.write(btext)
nf.write("# forests\n")
for ftext in forests:
   nf.write(ftext)
nf.close()
