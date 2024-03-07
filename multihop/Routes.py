from .config import settings
from tabulate import tabulate
import sys
import math
class Route:
    def __init__(self):
        self.neighbour_list = []
        self.fixed = False

    def set_fixed(self, uid, hops):
        self.fixed = True

        self.neighbour_list.append({'uid': uid,
                                    'snr': 0,
                                    'cumulative_lqi': sys.maxsize,
                                    'hops': hops,
                                    'best': True})

    def update(self, uid, snr, cumulative_lqi, hops):
        # TODO temp fix
        if not self.fixed:
            neighbour = self.find_node(uid)
            if neighbour is None:
                if len(self.neighbour_list) >= settings.MAX_ROUTE_SIZE:
                    self.neighbour_list.remove(self.find_worst())

                self.neighbour_list.append({'uid': uid,
                                            'snr': snr,
                                            'cumulative_lqi': cumulative_lqi,
                                            'hops': hops,
                                            'best': False})
            else:
                neighbour["uid"] = uid
                neighbour["snr"] = snr
                neighbour["cumulative_lqi"] = cumulative_lqi
                neighbour["hops"] = hops
            self.find_route()

    def find_node(self, _uid):
        #print("Search for uid", _uid)
        #print(self.neighbour_list)
        for neighbour in self.neighbour_list:
            #a = (_uid == neighbour["uid"])
            #print(a)
            if _uid == neighbour["uid"]:
                return neighbour
        return None
    
    def min_hop(self):
        print("Search min hop")
        min_hop = settings["MAX_ROUTE_SIZE"]
        for i, neighbour in enumerate(self.neighbour_list):
            if int(neighbour["hops"]) < min_hop:
                min_hop = neighbour["hops"]
        print("Found min hop", min_hop)
        return min_hop + 1
    
    
    def find_closer(self, hops):
        uids = []
        for neig in self.neighbour_list:
            if neig["hops"] < hops:
                uids.append(neig["uid"])
        return uids

              

    def find_worst(self):
        worst_i = 0
        for i, neighbour in enumerate(self.neighbour_list):
            if neighbour["cumulative_lqi"] > self.neighbour_list[worst_i]["cumulative_lqi"]:
                # cumulative LQI of this neighbour is worse than the previous one
                # -> save index of this neighbour
                worst_i = i
            elif neighbour["cumulative_lqi"] == self.neighbour_list[worst_i]["cumulative_lqi"]:
                # See if the LQI is equal -> worst route is the highest number of hops
                if neighbour["hops"] > self.neighbour_list[worst_i]["hops"]:
                    worst_i = i
                elif neighbour["hops"] == self.neighbour_list[worst_i]["hops"]:
                    # See if the nr of hops is equal -> worst route is the lowest snr to neighbour
                    if neighbour["snr"] < self.neighbour_list[worst_i]["snr"]:
                        worst_i = i

        return self.neighbour_list[worst_i]

    """     def find_best(self, exclude=[], mode = 0):
        if len(self.neighbour_list) > 0:
            if mode == 0:
                best_i = 0
                for i, neighbour in enumerate(self.neighbour_list):
                    neighbour["best"] = False
                    if neighbour["uid"] not in exclude:
                        if neighbour["cumulative_lqi"] < self.neighbour_list[best_i]["cumulative_lqi"]:
                            # cumulative LQI of this neighbour is better than the previous one
                            # -> save index of this neighbour
                            best_i = i
                        elif neighbour["cumulative_lqi"] == self.neighbour_list[best_i]["cumulative_lqi"]:
                            # See if the LQI is equal -> best route is the lowest number of hops
                            if neighbour["hops"] < self.neighbour_list[best_i]["hops"]:
                                best_i = i
                            elif neighbour["hops"] == self.neighbour_list[best_i]["hops"]:
                                # See if the nr of hops is equal -> best route is the lowest snr to neighbour
                                if neighbour["snr"] > self.neighbour_list[best_i]["snr"]:
                                    best_i = i

                self.neighbour_list[best_i]["best"] = True
            elif mode == 1:
                for i, neighbour in enumerate(self.neighbour_list):
                    sels = 1
                    average_reward = 1 / sels
                    delta_i = math.sqrt(2 * math.log(10) / sels)
                    upper_bound = average_reward + delta_i
                    if upper_bound > max_upper_bound:
                        max_upper_bound = upper_bound
                        arm = i
            return self.neighbour_list[best_i]
        return None """
    
    def find_best(self, exclude=[]):
        if len(self.neighbour_list) > 0:
            best_i = 0
            for i, neighbour in enumerate(self.neighbour_list):
                neighbour["best"] = False
                if neighbour["uid"] not in exclude:
                    if neighbour["cumulative_lqi"] < self.neighbour_list[best_i]["cumulative_lqi"]:
                        # cumulative LQI of this neighbour is better than the previous one
                        # -> save index of this neighbour
                        best_i = i
                    elif neighbour["cumulative_lqi"] == self.neighbour_list[best_i]["cumulative_lqi"]:
                        # See if the LQI is equal -> best route is the lowest number of hops
                        if neighbour["hops"] < self.neighbour_list[best_i]["hops"]:
                            best_i = i
                        elif neighbour["hops"] == self.neighbour_list[best_i]["hops"]:
                            # See if the nr of hops is equal -> best route is the lowest snr to neighbour
                            if neighbour["snr"] > self.neighbour_list[best_i]["snr"]:
                                best_i = i

            self.neighbour_list[best_i]["best"] = True
            return self.neighbour_list[best_i]
        else:
            return None
        
    def find_route(self, exclude=[]):
        return self.find_best(exclude)

    def __str__(self):
        return tabulate(self.neighbour_list, headers="keys")
    
    def get_neighbours_uid(self):
        return [int(x["uid"]) for x in self.neighbour_list]
    
