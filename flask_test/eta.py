from flask import Flask, request
import dgl
import torch
import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
from model import load_model


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'device={device}')

#MODEL_FILE_NAME =  'model.pth'

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

data = pd.read_csv('20170420_processed.zip')

ids, coords = give_nodes_coord(G)

data = data[(data['s_dist']<=0.0007)&(data['e_dist']<=0.0007)][['snode','enode', '3']]


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
    G.nodes[node]['idx'] =np.array([int(node)])


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
    return "<p>Для использования отправьте координаты череза запятую</p>"+ \
            "<p>пример http://localhost:5000/eta?cs=39.934155,116.433318,39.932,116.386536</p>"+ \
            "<p>или 9fe7-176-213-136-166.ngrok.io/eta?cs=39.934155,116.433318,39.932,116.386536</p>"+ \
            "<p>или 9fe7-176-213-136-166.ngrok.io/etadebug?cs=39.934155,116.433318,39.932,116.386536</p>"+ \
            f"<p>тест модели {model(dg, 'data', [[8089],[8088]])}</p>"


@app.route("/eta")
def base_eta():
    coordinates_str= request.args.get('cs')
    coordinates = [float(i) for i in coordinates_str.split(',')]

    point_1 = ox.nearest_nodes(G, coordinates[1], coordinates[0], return_dist=True)
    point_2 = ox.nearest_nodes(G, coordinates[3], coordinates[2], return_dist=True)

    if point_1[1] > 1000 or  point_2[1] > 1000:
        return '1000'   

    return str(model(dg, 'data', [[g_map[str(point_1[0])]], [g_map[str(point_2[0])]]]).cpu().detach().numpy().tolist()[0][0])


@app.route("/etadebug")
def debug_eta():
    coordinates_str= request.args.get('cs')
    coordinates = [float(i) for i in coordinates_str.split(',')]

    #est_time_arr = model(dg, 'data', [[8089],[8088]])
    point_1 = ox.nearest_nodes(G, coordinates[1], coordinates[0], return_dist=True)
    point_2 = ox.nearest_nodes(G, coordinates[3], coordinates[2], return_dist=True)
    r_str = f'<p>удаленность найденных точек в графе - {point_1[1]},\t {point_2[1]}</p>'
    if point_1[1] > 1000 or  point_2[1] > 1000:
        r_str += '<p>попробуйте другие координаты, эти слишком далеко</p>'
        return r_str    
    
    r_str += f'<p>{g_map[str(point_1[0])], g_map[str(point_2[0])]}</p>'

    r_str += f"<p> {model(dg, 'data', [[g_map[str(point_1[0])]], [g_map[str(point_2[0])]]])} </p>"

    # r_str += f'<p>{g_map[point_1[0]], g_map[point_2[0]]}</p>'
    return r_str


