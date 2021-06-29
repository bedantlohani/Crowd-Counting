import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import savemat

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_points = {}


    for annPoints in root.iter('image'):

        filename = annPoints.get('name')
        y, x = None, None
        list_with_single_points = []

        for point in annPoints.iter('points'):
            x, y= point.get('points').split(',')
            list_with_single_points.append([float(x), float(y)])

        if filename in list_with_all_points:
            list_with_all_points[filename].append(list_with_single_points)
        else:
            list_with_all_points[filename.split('.')[0]] = list_with_single_points

    return list_with_all_points

if __name__ == '__main__':
    annPoints = read_content("/home/bedant/crowdcount-mcnn/annotations.xml")
    #annPoints = np.array(annPoints)
    savemat("/home/bedant/crowdcount-mcnn/output/gt/ann.mat", annPoints)
