
from collections import deque, defaultdict
from heapq import heappush, heappop
import networkx as nx
import numpy as np
from itertools import count


class Dijkstra(object):
    def __init__(self, G, traffic_dict, start_time=0, time_step=15, use_dinamic=True):
        self.G = G
        self.traffic_dict = traffic_dict
        self.time_step = time_step
        self.start_time = start_time
        self.use_dinamic = use_dinamic

    def path(self, snodeid, enodeid):
        paths = defaultdict(list)
        times = self._dijkstra_multisource(self.G, [snodeid], self.weight, None, paths, None, enodeid)
        # print(f'times {times}')
        # print(f'paths {paths}')
        return times.get(enodeid, 0), paths.get(enodeid, [])

    def _dijkstra_multisource(
            self, G, sources, weight, pred=None, paths=None, cutoff=None, target=None
    ):
        """Uses Dijkstra's algorithm to find shortest weighted paths

        Parameters
        ----------
        G : NetworkX graph

        sources : non-empty iterable of nodes
            Starting nodes for paths. If this is just an iterable containing
            a single node, then all paths computed by this function will
            start from that node. If there are two or more nodes in this
            iterable, the computed paths may begin from any one of the start
            nodes.

        weight: function
            Function with (u, v, data) input that returns that edges weight

        pred: dict of lists, optional(default=None)
            dict to store a list of predecessors keyed by that node
            If None, predecessors are not stored.

        paths: dict, optional (default=None)
            dict to store the path list from source to each node, keyed by node.
            If None, paths are not stored.

        target : node label, optional
            Ending node for path. Search is halted when target is found.

        cutoff : integer or float, optional
            Length (sum of edge weights) at which the search is stopped.
            If cutoff is provided, only return paths with summed weight <= cutoff.

        Returns
        -------
        distance : dictionary
            A mapping from node to shortest distance to that node from one
            of the source nodes.

        Raises
        ------
        NodeNotFound
            If any of `sources` is not in `G`.

        Notes
        -----
        The optional predecessor and path dictionaries can be accessed by
        the caller through the original pred and paths objects passed
        as arguments. No need to explicitly return pred or paths.

        """
        G_succ = G._succ if G.is_directed() else G._adj

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        for source in sources:
            if source not in G:
                raise nx.NodeNotFound(f"Source {source} not in G")
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                break
            for u, e in G_succ[v].items():
                cost = weight(v, u, e, d)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)

        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.
        return dist

    def weight(self, snodeid, enodeid, date, cur_time):
        # Пробуем с текущим временем
        # road = self.road_df.loc[(self.road_df['snodeid']==snodeid) & (self.road_df['enodeid']==enodeid)]
        # print(f'snodeid, enodeid = {snodeid},{enodeid}')
        timestamp = (
                                cur_time + self.start_time) / self.time_step if self.use_dinamic else self.start_time / self.time_step
        # if road.shape[0] == 0:
        #   raise Exception(f"Road not found snodeid = {snodeid} enodeid = {enodeid}")
        #   return int(1e6)
        # else:
        link_id = int(date.get('link_id', 0))
        length = date.get('length', 50)
        # print(f'link_id {link_id} length {length}')
        dict_traffic = self.traffic_dict.get(link_id)
        speed = dict_traffic.get(timestamp)
        # df = self.traffic_df.loc[(self.traffic_df['link_id']==link_id) & (self.traffic_df['time_code'] == timestamp)]
        # if df.shape[0] == 0:
        if speed is None:
            # Если не вышло берем +- 15 мин и усредняем
            s = dict_traffic.get(round(timestamp))
            speed_list = []
            if s is not None:
                speed_list.append(s)
            s = dict_traffic.get(round(timestamp) + 1)
            if s is not None:
                speed_list.append(s)
            speed = None if len(speed_list) == 0 else np.mean(speed_list)
            # df = self.traffic_df.loc[(self.traffic_df['link_id']==link_id) & (self.traffic_df['time_code'] < timestamp+1) & (self.traffic_df['time_code'] > timestamp-1)]
        if speed is None:
            # if df.shape[0] == 0:
            if dict_traffic:
                speed = np.mean(list(dict_traffic.values()))
            # df = self.traffic_df.loc[self.traffic_df['link_id']==link_id]
            # print(f'time_code {timestamp} snodeid = {snodeid} enodeid = {enodeid}')
            # Если не вышло берем за все время

        # print(f'snodeid, enodeid = {snodeid},{enodeid}  link_id = {link_id},   df =  {df}')
        # if df.shape[0] == 0:
        if speed is None or speed == 0:
            # Если иданных нет совсем берем скорость поменьше, чтобы туда не вел
            raise Exception(f"Traffic not found snodeid = {snodeid} enodeid = {enodeid} link_id = {link_id}")
        else:
            return (60 * length) / (speed * 1000)
