from skimage.morphology import thin,skeletonize, remove_small_objects, remove_small_holes
import numpy as np
from matplotlib.pylab import plt
import cv2
import os
import pandas as pd
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage as ndi
from collections import defaultdict, OrderedDict
import sys
from osgeo import ogr, osr, gdal
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
from utils import sknw

from multiprocessing import Pool

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_sequential_duplicates(seq):
    #todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_angle(p0, p1=np.array([0,0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def array2raster(newRasterfn, originRasterfn, array):

    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)

    originDS = gdal.Open(originRasterfn,0)
    geoTrans = originDS.GetGeoTransform()
    outRaster.SetGeoTransform(geoTrans)
    outRaster.SetProjection(originDS.GetProjection())
    outband.FlushCache()

def preprocess(img, thresh):
    img = (img > (255 * thresh)).astype(np.bool)
    remove_small_objects(img, 30, in_place=True)
    remove_small_holes(img, 30, in_place=True)
    #img = cv2.dilate(img.astype(np.uint8), np.ones((5, 5)))
    return img

def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # draw node by o
    node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

def remove_small_terminal(G):
    deg = G.degree()
    terminal_points = [i[0] for i in deg if i[1]==1]
    #for i in deg:
    #    if i[1] == 1:
    #        terminal_points.append(i[0])
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(e)
    return


linestring = "LINESTRING {}"
def make_skeleton(root,ske_path,fn, debug, fix_borders):
    replicate = 5
    clip = 2
    rec = replicate + clip
    # open and skeletonize
    #以灰度图打开
    print(os.path.join(root,fn))
    img = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
    #assert img.shape == (1300, 1300)
    if fix_borders:
        #边缘填充
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
    thresh = 0.4
    #去掉小的区域块
    img = preprocess(img, thresh)
    #return img_copy, img
    if not np.any(img):
        return None, None
    ske = skeletonize(img).astype(np.uint16)
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)

    save_path=os.path.join(ske_path,fn)
    # save_path=save_path.replace("prediction","ske")
    # save_path=save_path.replace(".tif",".png")
    # cv2.imwrite(save_path,ske)
    # save_path=save_path.replace(".tif","_ske.tif")
    array2raster(save_path, os.path.join(root, fn), ske)

    return img_copy, ske


def add_small_segments(G, terminal_points, terminal_lines):
    node = G.node
    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < 20))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > 20) & (dists < 100))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > 20:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if -20 < angle < 20 or angle < -160 or angle > 160:
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.node[s]['o'], G.node[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.node[s]['o'], G.node[e]['o']
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
    return wkt


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def build_graph(root,ske_path, fn, debug=False, add_small=True, fix_borders=True):
    #ske = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)

    img_copy, ske = make_skeleton(root, ske_path,fn, debug, fix_borders=False)
    if ske is None:
        return [linestring.format("EMPTY")]
    G = sknw.build_sknw(ske, multi=True)
    remove_small_terminal(G)
    node_lines = graph2lines(G)
    if not node_lines:
        return [linestring.format("EMPTY")]
    node = G.node
    deg = G.degree()
    wkt = []
    #terminal_points = [i for i, d in deg.items() if d == 1]
    terminal_points = [i[0] for i in deg if i[1]==1]

    terminal_lines = {}
    vertices = []
    for w in node_lines:
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        wkt.extend(add_small_segments(G, terminal_points, terminal_lines))

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    return wkt

#from spatialfunclib import projection_onto_line
#def douglas_peucker(segment, epsilon):
#    dmax = 0
#    index = 0

#    for i in range(1, len(segment) - 1):
#        (_, _, d) = projection_onto_line(segment[0].latitude, segment[0].longitude, segment[-1].latitude, segment[-1].longitude, segment[i].latitude, segment[i].longitude)

#        if (d > dmax):
#            index = i
#            dmax = d

#    if (dmax >= epsilon):
#        rec_results1 = douglas_peucker(segment[0:index], epsilon)
#        rec_results2 = douglas_peucker(segment[index:], epsilon)

#        smoothed_segment = rec_results1
#        smoothed_segment.extend(rec_results2)
#    else:
#        smoothed_segment = [segment[0], segment[-1]]

#    return smoothed_segment


def watershed():
    image=r"D:\Data\nanjing\JS_qixiaqu_edge.tif"
    label=image[0:image.rfind('.')]+'_lab.tif'
    label3=image[0:image.rfind('.')]+'_lab3.tif'

    datasetname = gdal.Open( image, gdal.GA_ReadOnly )
    if datasetname is None:
        print('Could not open %s'% label)  #mask改成了label
    img_width = datasetname.RasterXSize
    img_height = datasetname.RasterYSize
    nBand = datasetname.RasterCount
    if nBand != 1:
        print('process first Band!')
    imageData = np.array(datasetname.GetRasterBand(1).ReadAsArray())

    #imgData = filters.rank.median(imageData, morphology.disk(1)) #过滤噪声
    gradient = filters.rank.gradient(imageData, morphology.disk(2)) #计算梯度
    array2raster(label,image,gradient)
    line = imageData.astype(np.int) - gradient
    line[line < 0]=0
    array2raster(label3,image,line.astype(np.uint8))
    return label3
    #gradient = filters.rank.gradient(line.astype(np.uint8), morphology.disk(1)) #计算梯度
    #line = line-gradient
    #line[line < 0]=0

    #markers = np.zeros(img.shape)
    #for i in range(img_height):
    #    for j in range(img_width):
    #        if (img[i,j] < 100 and gradient[i,j] < 15):
    #            markers[i,j] = 1

    #remove_small_objects(img, 300, in_place=True)
    #remove_small_holes(img, 300, in_place=True)
    #skeleton =morphology.skeletonize(img)
    #将梯度值低于10的作为开始标记点
    #markers = filters.rank.gradient(denoised, morphology.disk(2)) <10
    #markers = ndi.label(markers)[0]

    #labels =morphology.watershed(gradient, markers, mask=imageData) #基于梯度的分水岭算法

    #array2raster(label,image,labels)

def skeleton():

    results_root = r"F:\acm_building_800\polygon2"
    # fn = '13_clip2_edge31.tif'
    # fn=r"‪F:\acm_building_800\edge\0000000002_road.tif"
    fn="5-0000000003_1.png"
    wkt=build_graph(results_root, fn)
    #print(wkt)
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")   
    gdal.SetConfigOption("SHAPE_ENCODING","")  
    refds=gdal.Open(os.path.join(results_root,fn))
    proj=refds.GetProjectionRef()  #获取投影
    spatial=osr.SpatialReference(proj)
    adfGeoTransform = refds.GetGeoTransform()

    # file = os.path.join(results_root, 'res.shp')
    file=os.path.join(results_root,fn)
    file=file.replace(".tif","_res.shp")
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if(os.path.exists(file)):
        driver.DeleteDataSource(file)
    dataSource = driver.CreateDataSource(file)
    layer = dataSource.CreateLayer('line', geom_type=ogr.wkbLineString, srs=spatial)
    fieldDefn = ogr.FieldDefn('id', ogr.OFTString)
    fieldDefn.SetWidth(8)
    featureDefn = layer.GetLayerDefn()
    layer.CreateField(fieldDefn)
    feature = ogr.Feature(featureDefn)

    for line in wkt:
        geom = ogr.CreateGeometryFromWkt( line )
        line0 = ogr.Geometry(ogr.wkbLineString)
        for i in range(0, geom.GetPointCount()):
            pt = geom.GetPoint(i)
            x = adfGeoTransform[0] + pt[0] * adfGeoTransform[1]
            y = adfGeoTransform[3] + pt[1] * adfGeoTransform[5]
            line0.AddPoint(x,y)
        feature.SetGeometry(line0)
        layer.CreateFeature(feature)
    dataSource.Destroy()

def skeleton2(edge_path,ske_path,ske_temp_path):

    fns=os.listdir(edge_path)
    for fn in fns:
        wkt=build_graph(edge_path,ske_path, fn)
        #print(wkt)
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")
        gdal.SetConfigOption("SHAPE_ENCODING","")
        refds=gdal.Open(os.path.join(edge_path,fn))
        proj=refds.GetProjectionRef()  #获取投影
        spatial=osr.SpatialReference(proj)
        adfGeoTransform = refds.GetGeoTransform()

        # file = os.path.join(results_root, 'res.shp')
        file=os.path.join(ske_temp_path,fn)
        file=file.replace(".tif","_res.shp")
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if(os.path.exists(file)):
            driver.DeleteDataSource(file)
        dataSource = driver.CreateDataSource(file)
        layer = dataSource.CreateLayer('line', geom_type=ogr.wkbLineString, srs=spatial)
        fieldDefn = ogr.FieldDefn('id', ogr.OFTString)
        fieldDefn.SetWidth(8)
        featureDefn = layer.GetLayerDefn()
        layer.CreateField(fieldDefn)
        feature = ogr.Feature(featureDefn)

        for line in wkt:
            geom = ogr.CreateGeometryFromWkt( line )
            line0 = ogr.Geometry(ogr.wkbLineString)
            for i in range(0, geom.GetPointCount()):
                pt = geom.GetPoint(i)
                x = adfGeoTransform[0] + pt[0] * adfGeoTransform[1]
                y = adfGeoTransform[3] + pt[1] * adfGeoTransform[5]
                line0.AddPoint(x,y)
            feature.SetGeometry(line0)
            layer.CreateFeature(feature)
        dataSource.Destroy()
        ske_name=os.path.join(ske_path,fn)
        ske=cv2.imread(ske_name,0)
        ske=np.where(ske==0,0,255)
        cv2.imwrite(ske_name,ske)

if __name__ == "__main__":
    skeleton()
    #watershed()
