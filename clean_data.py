import os

from datetime import date, timedelta
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import networkx as nx

ROAD_FILE = 'road_network_sub-dataset/road_network_sub-dataset'
GPS_FILE = 'road_network_sub-dataset/link_gps'
TRAFFIC_FILE = 'traffic_speed_sub-dataset'

CLEAN_FOLDER = './data_clean/'

FIRST_DATE = date(2017, 4, 1)

def clean_graph():
    road_netw = pd.read_csv(ROAD_FILE, sep='\t')
    link_loc = pd.read_csv(GPS_FILE, sep='\t', header=None)
    link_loc.columns = ['link_id', 'lon', 'lat']
    
    for c in ['lat', 'lon']:
        road_netw[c] = link_loc[c]
        
    good_links = road_netw['snodeid'].isin(road_netw['enodeid'])
    G = nx.Graph()
    G.add_edges_from(road_netw[good_links][['snodeid', 'enodeid']].apply(tuple, axis=1))
    
    max_len = 0
    max_comp = None
    for i, cc in enumerate(nx.connected_components(G)):
        if max_len < len(cc):
            max_len = len(cc)
            max_comp = cc
            
    biggest_connected = road_netw['snodeid'].isin(max_comp) & road_netw['enodeid'].isin(max_comp)
    
    link_loc['node_id'] = road_netw['enodeid']
    return road_netw[biggest_connected].drop(['lat', 'lon'], axis=1), link_loc[['node_id', 'lat', 'lon']].drop_duplicates()


def parse_traffic(link_set):
    folder_name = CLEAN_FOLDER + 'traffic/'
    os.makedirs(folder_name)
    
    open_fd = {}
    
    traffic_file = open(TRAFFIC_FILE)
    for line in tqdm(traffic_file, total=264386688):
        time_tag = int(line.split()[1].strip(', '))
        link_id = int(line.split()[0].strip(', '))
        if link_id not in link_set:
            continue

        line_date = FIRST_DATE + timedelta(days=time_tag // (24 * 4))
        
        date_str = str(line_date)
        if date_str not in open_fd:
            open_fd[date_str] = open(folder_name + '/' + date_str, 'w')
        
        open_fd[date_str].write(line)
            
    traffic_file.close()
        
    for _, f in open_fd.items():
        f.close()
        

if __name__ == '__main__':
    
    assert os.path.isfile(ROAD_FILE)
    assert os.path.isfile(GPS_FILE)
    assert os.path.isfile(TRAFFIC_FILE)
    
    os.makedirs(CLEAN_FOLDER)
    
    road_clean, loc_clean = clean_graph()
    road_clean.to_csv(CLEAN_FOLDER + 'roads.csv', index=False)
    loc_clean.to_csv(CLEAN_FOLDER + 'gps.csv', index=False)
    
    parse_traffic(set(road_clean['link_id'].unique()))
    
    
    
    