import random
import networkx as nx
import torch
import math
import os.path
import numpy as np
import matplotlib.pyplot as plt
from cargonet.utils.geo import dist_m_v2
from datetime import timedelta, datetime
from cargonet.dataset.dataset import RailDataset
from cargonet.dataset.activeroutesv1 import RoadsAsEdgesV1

random.seed(123)

class Merklin:

    def __init__(self, path, start_time, base_speed, net, pause_p, pause_minmax, 
        net_node_delay_params, net_mapping, transport_id):
        self.path = path
        self.base_speed = base_speed
        self.start_time = start_time
        self.net = net
        self.pause_p = pause_p
        self.pause_minmax = pause_minmax
        self.net_node_delay_params = net_node_delay_params
        self.net_mapping = net_mapping
        self.transport_id = transport_id

        # Assign random properties to this specific transport
        self.graph = nx.DiGraph()

    def periodic_delay_func(self, x_h, x_d, x_m, c_h, c_d, c_m, s_h, s_d, s_m):
        hourly = s_h * np.sin(c_h + math.pi / 24 * 2 * x_h)
        daily = s_d * np.sin(c_d + math.pi / (24 * 7) * 2 * x_d)
        monthly = s_m * np.sin(c_m + math.pi / (24 * 31) * 2 * x_m)
        return hourly + daily # + monthly

    def cantor_pairing(self, a, b):
        return int((((a + b) * (a + b + 1)) / 2) + b)

    def random_cont_func(self, x):
        r = np.random.RandomState(2*x)
        return r.uniform(0, 1, 1)[0]

    def incident_delay_func(self, station_id, t, duration=timedelta(hours=3), p=0.01, max_delay=120): # 3 hour duration
        t_i = (t - datetime(2019, 1, 1)).total_seconds()
        reduced = int(math.floor(t_i / duration.total_seconds()))
        seed = self.cantor_pairing(reduced, station_id)
        if self.random_cont_func(seed) <= p:
            print("Incident at station", station_id)
            return timedelta(minutes=(random.random() * max_delay) + 1)
        return timedelta(minutes=0)

    def get_current_node_delay_p(self, station, planned_arrival):
        return self.periodic_delay_func(
            x_h=planned_arrival.hour,
            x_d=planned_arrival.day * 24,
            c_h=self.net_node_delay_params["net_station_delay_shift"][station, 0],
            c_d=self.net_node_delay_params["net_station_delay_shift"][station, 1],
            s_h=self.net_node_delay_params["net_station_delay_scale"][station, 0],
            s_d=self.net_node_delay_params["net_station_delay_scale"][station, 1],
            x_m=0, c_m=0, s_m=0, # Not used atm
        )

    def build(self):
        # Interpolate the arrival times, pauses and the delays
        for i, station in enumerate(self.path):
            mapped_station = self.net_mapping[station]
            if i > 0:
                # Interpolate
                edge = (i-1, i)
                net_edge = (self.path[i-1], self.path[i])
                stations = (self.net.nodes[net_edge[0]], self.net.nodes[net_edge[1]])
                positions = [s["pos"] for s in stations]
                
                # Add a random pause
                pause = timedelta(minutes=0)
                if random.random() <= self.pause_p:
                    pause = timedelta(minutes=random.randint(*self.pause_minmax))

                # Add edge
                distance = dist_m_v2(*positions)
                planned_duration = timedelta(hours=(distance / 1_000) / self.base_speed)
                planned_arrival = self.graph.nodes[i-1]["plannedDepartureTime"] + planned_duration
                planned_departure = planned_arrival + pause

                # Initial delay depends on the stations but must be updated in case trains collide
                node_delay_p = self.get_current_node_delay_p(mapped_station, planned_arrival)
                
                # Plot the delay function
                if False:
                    plt.plot(
                        range(10000),
                        [self.get_current_node_delay_p(mapped_station, planned_arrival + t * timedelta(minutes=10)) for t in range(10000)]
                    )
                    plt.show()
                rel_delay = node_delay_p.item() * planned_duration

                # Add delay caused by incidents
                rel_delay += self.incident_delay_func(mapped_station, planned_arrival)
                duration = planned_duration + rel_delay

                # Edges are done
                self.graph.add_edge(*edge, **dict(
                    distance=distance,
                    plannedDuration=planned_duration,
                    duration=duration,
                ))

                delay = timedelta(minutes=self.graph.nodes[i-1]["delay"]) + rel_delay

                # Add node
                self.graph.add_node(i, **dict(
                    pos=self.net.nodes[station]["pos"],
                    pause=pause,
                    plannedArrivalTime=planned_arrival,
                    plannedDepartureTime=planned_departure,
                    arrivalTime=planned_arrival + rel_delay,
                    departureTime=planned_departure + rel_delay,
                    transportId=self.transport_id,
                    stationId=mapped_station,
                    delay=int(delay.total_seconds() / 60),
                ))
            else:
                # Initialize
                planned_departure = planned_arrival = self.start_time
                node_delay_p = self.get_current_node_delay_p(mapped_station, planned_departure)
                rel_delay = node_delay_p.item() * timedelta(minutes=10)
                delay = rel_delay

                self.graph.add_node(i, **dict(
                    pos=self.net.nodes[station]["pos"],
                    pause=0,
                    plannedArrivalTime=planned_arrival,
                    plannedDepartureTime=planned_departure,
                    arrivalTime=planned_arrival,
                    departureTime=planned_departure + rel_delay,
                    transportId=self.transport_id,
                    stationId=mapped_station,
                    delay=int(delay.total_seconds() / 60),
                ))
        return self.graph



class Simulation(RailDataset):

    node_feature_mapping = ["stationId", "imId", "country", "lat", "lon"]
    edge_feature_mapping = ["distance", "popularity"]

    def __init__(
        self,
        root,
        name=None,
        transform=None,
        pre_transform=None,
        limit=1,
        plot_download=False,
        plot_processing=False,
        force_reprocess=False,
        verbose=True,
        interval=timedelta(minutes=10),
        seq_len=3,
        pred_seq_len=3,
    ):
        self.root = root

        self.undirected = False
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        self.p_new_route = 0.8
        self.edge_speed_variance = 0.1
        self.node_delay_variance = torch.FloatTensor([0.2, 0.2]) 

        self.num_stations = 10_000
        self.target_edges_total = 40_000
        self.p_edge = 2 * self.target_edges_total / (10_000 * 10_000)
        self.pause_p = 0.2

        self.snet, self.net_node_delay_params, self.net_edge_delay_params, self.snet_mapping = self.gen_network(
            num_stations=self.num_stations,
            p_edge=self.p_edge,
            edge_speed_variance=self.edge_speed_variance,
            node_delay_variance=self.node_delay_variance
        )

        self.net, self.mapping = self.process_full_graph(
            self.snet, self.snet_mapping, reprocess=force_reprocess
        )
        print("Simulation network:", self.net)
        self.last_transport_id = 0

        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            limit=limit,
            plot_download=plot_download,
            plot_processing=plot_processing,
            force_reprocess=force_reprocess,
            verbose=verbose,
            interval=interval,
        )

    @property
    def timerange(self):
        start, end = self.total_timerange
        t, c = start, 0
        timerange = []
        while t <= end and c < self.limit:
            timerange.append(t)
            t = t + self.interval
            c += 1
        return timerange

    def download(self):
        # No need for download
        pass
    
    def process(self):
        states_count = len(self.raw_paths)
        sl, pl = (
            self.seq_len,
            self.pred_seq_len,
        )
        transports = []
        for i in range(states_count):
            out = os.path.join(
                self.processed_dir, self.processed_file_names[i],
            )

            t = self.timerange[i]
            t_next = t + self.interval
            self.vlog(
                "Processing t[%d] %s - %s (%d/%d, %d+%d states) active=%d" % (i, t, t_next, i, states_count, sl, pl, len(transports))
            )

            (data, _), transports = self.aggregate(t, transports)
            
            # Apply filters and transformations
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data,
                out,
            )

    
    @staticmethod
    def gen_network(num_stations, p_edge, edge_speed_variance, node_delay_variance):
        print("Generating network")
        sqrt_nodes = int(np.sqrt(num_stations))
        net = nx.grid_graph(dim=[sqrt_nodes, sqrt_nodes])
        pos = dict(zip(net.nodes(), net.nodes()))

        # Remove most of the edges
        if False:
            for u, v in net.edges:
                if random.random() > p_edge:
                    net.remove_edge(u,v)

        # Map the positions to lat and lon
        min_lat, max_lat = 47, 55
        min_lon, max_lon = 5.8, 14.5
        pos = {n: (min_lat + (x/sqrt_nodes * (max_lat - min_lat)), 
                    min_lon + (y/sqrt_nodes * (max_lon - min_lon))
                    ) for n, (x,y) in pos.items()}

        # Apply some random translation to the nodes
        variation_p = 0.1
        pos = {n: (
            x + ((random.random() * 2 -1) * (max_lat-min_lat) * variation_p),
            y + ((random.random() * 2 -1) * (max_lon-min_lon) * variation_p))
            for n, (x,y) in pos.items()}

        # Create a station mapping
        mapping = dict(zip(sorted(net.nodes), range(len(net.nodes))))

        # Save the position back
        for n, data in net.nodes(data=True):
            data["pos"] = pos[n]
            data["stationId"] = mapping[n]
            data["imId"] = 0
            data["country"] = 0

        for u, v, data in net.edges(data=True):
            positions = (pos[u], pos[v])
            data["distance"] = dist_m_v2(*positions)
            data["popularity"] = 0
        
        print("Generating random weights for nodes and edges")
        net_edge_weights = (torch.rand((net.number_of_nodes(), net.number_of_nodes())) * 2 - 1) * edge_speed_variance
        net_station_delay_shift = torch.randint(low=0, high=10000, size=(net.number_of_nodes(), 2)) # c_d and c_y 
        net_station_delay_scale = torch.FloatTensor([0, 2 * node_delay_variance[1]]) + node_delay_variance * torch.rand((net.number_of_nodes(), 2)) # s_d and s_y

        net_station_weights = dict(
            net_station_delay_shift=net_station_delay_shift,
            net_station_delay_scale=net_station_delay_scale,
        )

        print("Generated random network with %d nodes and %d edges" % (net.number_of_nodes(), net.number_of_edges()))
        
        return net, net_station_weights, net_edge_weights, mapping

    def get_random_route(self, t):
        while True: 
            start, end = random.choice(list(self.snet.nodes)), random.choice(list(self.snet.nodes))
            paths = []

            gen = nx.all_simple_paths(self.snet, source=start, target=end)
            def dist(a, b):
                (x1, y1) = a
                (x2, y2) = b
                # Pythagoras
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            paths.append(nx.astar_path(self.snet, source=start, target=end, heuristic=dist))
            path = random.choice(paths)
            base_speed = random.randint(70, 150) # kmph
            transport = Merklin(
                path=path,
                net=self.snet,
                start_time=t,
                pause_p=self.pause_p,
                net_node_delay_params=self.net_node_delay_params,
                base_speed=base_speed,
                pause_minmax=(0, 45),
                net_mapping=self.snet_mapping,
                transport_id=self.last_transport_id
            )
            self.last_transport_id += 1
            return transport

    @property
    def encoder(self):
        return RoadsAsEdgesV1

    def aggregate(self, t, transports):
        # Spawn new transports with some possibility
        collisions = []
        if random.random() <= self.p_new_route:
            route = self.get_random_route(t).build()
            
            # Find conflicting trains and add delays to them
            for _ in range(1):
                for transport in transports:
                    for n_t in nx.topological_sort(transport):
                        for n_new in nx.topological_sort(route):
                            d_t = transport.nodes[n_t]
                            d_new = route.nodes[n_new]
                            if d_t["stationId"] == d_new["stationId"]:
                                influence = abs(d_t["arrivalTime"] - d_new["arrivalTime"])
                                if influence < timedelta(minutes=15):
                                    collisions.append((t, ))
                                    collision = (route, n_new) if d_new["arrivalTime"] >= d_t["arrivalTime"] else (transport, n_t)
                                    descendants = nx.nodes(nx.dfs_tree(*collision))
                                    for descendant in descendants:
                                        collision[0].nodes[descendant]["delay"] += int((influence + timedelta(minutes=20)).total_seconds() / 60)
                                        pass
            transports.append(route)
            print("added new route of length", len(route.nodes))

        def get_segment(_transport, _t):
            # Find segment of current timestep t
            current_segment = None
            for u, v, data in _transport.edges(data=True):
                src = _transport.nodes[u].get("arrivalTime")
                dest = _transport.nodes[v].get("arrivalTime")
                if src <= _t < dest:
                    current_segment = (u, v)
            return current_segment

        # Filter only transports that fit current timeframe
        first_time = None
        # Filter all transports that completed
        for tp in transports:
            if get_segment(tp, t) is None:
                transports.remove(tp)
        
        for transport in transports:
            for _, data in transport.nodes(data=True):
                candidates = [
                    data.get("plannedArrivalTime"),
                    data.get("arrivalTime"),
                    data.get("plannedDepartureTime"),
                    data.get("departureTime"),
                ]
                candidate = min(candidates)
                if first_time is None:
                    first_time = candidate
                first_time = min(candidate, first_time)

        id_mapping = dict(zip(self.snet_mapping.values(), self.snet_mapping.values()))
        encoder = self.encoder(
            first_time=first_time,
            seq_len=self.seq_len,
            pred_seq_len=self.pred_seq_len,
            max_transports=len(transports),
            net=(self.net, id_mapping),
        )

        for transport in transports:
            # Plot the transport
            if False:
                delays = [data["delay"] for _, data in transport.nodes(data=True)]
                print(delays)
                plt.plot(delays)
                plt.show()

            current_segment = get_segment(transport, t)
            if current_segment is None:
                # Does not fit current timeframe, should never happen with filtering
                continue
            cu, cv = current_segment
            # Include some ancestors and descendants
            desc = list(nx.nodes(nx.dfs_tree(transport, cv)))[: self.pred_seq_len]
            anc = list(nx.nodes(nx.dfs_tree(transport.reverse(copy=True), cu)))[
                : self.seq_len
            ]
            assert len(set(desc + anc)) <= self.pred_seq_len + self.seq_len + 1

            encoder.add(t, transport, center_edge=(cu, cv), anc=anc, desc=desc)

        return encoder.encode(t), transports

    def simulate(self, start, steps, interval=timedelta(minutes=10)):
        transports = []
        for i, t in enumerate([start + n * interval for n in range(steps)]):
            data, transports = self.aggregate(t, transports)
            print("t=%s active=%d" % (t, len(transports)))

        print("Done")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset_base_path = os.path.join(base_path, "../../datasets")
    models_base_path = os.path.join(base_path, "../../trained")
    assert os.path.exists(dataset_base_path)
    assert os.path.exists(models_base_path)

    dataset_name = "simulation-v1"
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    ds_options = dict(seq_len=10, pred_seq_len=10,)

    sim = Simulation(
        root=dataset_path,
        name=dataset_name,
        limit=10,
        force_reprocess=True,
        **ds_options
    )