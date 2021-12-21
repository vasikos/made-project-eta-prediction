from flask import Flask, request, send_from_directory
import dgl
import torch
import networkx as nx
import pandas as pd
import numpy as np
import json
import osmnx as ox
from model import load_model
import pickle
from path import Dijkstra

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'device={device}')


# MODEL_FILE_NAME =  'model.pth'

def give_nodes_coord(G):
    nodes_id = []
    nodes_coords = []
    for node in G.nodes:
        nodes_id.append(node)
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        nodes_coords.append([y, x])
    return np.array(nodes_id), np.array(nodes_coords).astype(np.float32)


G = nx.read_gml('graph')
with open('mini_traffic_dict', 'rb') as pickle_file:
    mini_traffic_dict = pickle.load(pickle_file)

data = pd.read_csv('20170420_processed.zip')

ids, coords = give_nodes_coord(G)

data = data[(data['s_dist'] <= 0.0007) & (data['e_dist'] <= 0.0007)][['snode', 'enode', '3']]

s = []
e = []
for i in range(data.shape[0]):
    s.append(ids[data.iloc[i]['snode']])
    e.append(ids[data.iloc[i]['enode']])
data['snode'] = s
data['enode'] = e

G = nx.DiGraph(G)
for node in G.nodes:
    G.nodes[node]['data'] = np.array([G.nodes[node]['x'], G.nodes[node]['y']])
    G.nodes[node]['idx'] = np.array([int(node)])

dg = dgl.from_networkx(G, node_attrs=['data', 'idx'])

tmp = np.array(dg.ndata['idx'])

g_map = dict()
for i in range(tmp.shape[0]):
    g_map[str(tmp[i][0])] = i

data['snode'] = data['snode'].map(g_map)

data['enode'] = data['enode'].map(g_map)

G2 = nx.relabel.relabel_nodes(G, g_map)

data = data.reset_index()[['snode', 'enode', '3']]

model = load_model()
model.eval()
print(model)

dg.ndata['data'] = (dg.ndata['data'] - torch.tensor([116., 39.])).float()

dg = dgl.add_self_loop(dg)

app = Flask(__name__)


@app.route("/")
def print_help():
    return send_from_directory(".", "index.html")


@app.route("/eta")
def base_eta():
    coordinates_str = request.args.get('cs')
    coordinates = [float(i) for i in coordinates_str.split(',')]

    point_1 = ox.nearest_nodes(G, coordinates[1], coordinates[0], return_dist=True)
    point_2 = ox.nearest_nodes(G, coordinates[3], coordinates[2], return_dist=True)

    if point_1[1] > 1000 or point_2[1] > 1000:
        return '1000'

    return str(round(
        model(dg, 'data', [[g_map[str(point_1[0])]], [g_map[str(point_2[0])]]]).cpu().detach().numpy().tolist()[0][0]))


@app.route("/path")
def short_path():
    coordinates_str = request.args.get('cs')
    param = coordinates_str.split(',')
    coordinates = [float(param[i]) for i in range(4)]
    time_list = param[4].split(':')
    start_time = int(time_list[0]) * 60 + int(time_list[1])
    point_1 = ox.nearest_nodes(G, coordinates[1], coordinates[0], return_dist=True)
    point_2 = ox.nearest_nodes(G, coordinates[3], coordinates[2], return_dist=True)
    print(f"time_list {time_list}")
    print(f"start_time = {start_time}")
    if point_1[1] > 1000 or point_2[1] > 1000:
        return ''

    # path = nx.algorithms.shortest_paths.generic.shortest_path(G,source=point_1[0],target=point_2[0], weight='length')
    dijkstra = Dijkstra(G, mini_traffic_dict, start_time)
    time, path = dijkstra.path(point_1[0], point_2[0])
    print(f"time {time}")
    # print(point_1[1], point_2[1])
    path_coord = [{'lat': G.nodes[node]['y'], 'lng': G.nodes[node]['x']} for node in path]
    path_coord.append(time)
    return json.dumps(path_coord)


@app.route("/etadebug")
def debug_eta():
    coordinates_str = request.args.get('cs')
    coordinates = [float(i) for i in coordinates_str.split(',')]

    # est_time_arr = model(dg, 'data', [[8089],[8088]])
    point_1 = ox.nearest_nodes(G, coordinates[1], coordinates[0], return_dist=True)
    point_2 = ox.nearest_nodes(G, coordinates[3], coordinates[2], return_dist=True)
    r_str = f'<p>удаленность найденных точек в графе - {point_1[1]},\t {point_2[1]}</p>'
    if point_1[1] > 1000 or point_2[1] > 1000:
        r_str += '<p>попробуйте другие координаты, эти слишком далеко</p>'
        return r_str

    r_str += f'<p>{g_map[str(point_1[0])], g_map[str(point_2[0])]}</p>'

    r_str += f"<p> {model(dg, 'data', [[g_map[str(point_1[0])]], [g_map[str(point_2[0])]]])} </p>"

    # r_str += f'<p>{g_map[point_1[0]], g_map[point_2[0]]}</p>'
    return r_str


@app.route("/static/<path>")
def static_path(path):
    return send_from_directory(".", path)


app.run()
