from .utils import *
from .Timers import TxTimer, TimerType
from .Packets import Message, MessageType
from .Links import LinkTable
from .Routes import Route

import numpy as np
import networkx as nx
import simpy
from aenum import Enum, MultiValue, auto, IntEnum
import collections
from tabulate import tabulate
import simpy
import math
import logging
import statistics

class GatewayState(IntEnum):
    STATE_INIT = auto()
    STATE_ROUTE_DISCOVERY = auto()
    STATE_RX = auto()
    STATE_PROC = auto()


class NodeType(IntEnum):
    GATEWAY = auto()
    SENSOR = auto()


class NodeState(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    STATE_INIT = 0, "INIT"
    STATE_CAD = 2, "CAD"
    STATE_RX = 3, "RX"
    STATE_TX = 6, "TX"
    STATE_PREAMBLE_TX = 5, "P_TX"
    STATE_SLEEP = 1, "ZZZ"
    STATE_SENSING = 4, "SNS"
    STATE_DEPLETED = 7, "DEP"


def power_of_state(settings, s: NodeState):
    if s is NodeState.STATE_INIT: return 0
    if s is NodeState.STATE_CAD: return settings.POWER_CAD_CYCLE_mW
    if s is NodeState.STATE_RX: return settings.POWER_RX_mW
    if s is NodeState.STATE_TX: return settings.POWER_TX_mW
    if s is NodeState.STATE_PREAMBLE_TX: return settings.POWER_TX_mW
    if s is NodeState.STATE_SLEEP: return settings.POWER_SLEEP_mW
    if s is NodeState.STATE_SENSING: return settings.POWER_SENSE_mW
    if s is NodeState.STATE_DEPLETED: return 0
    else:
        ValueError(f"Sensorstate {s} is unknown")


class Node:
    def __init__(self, env: simpy.Environment, _settings, _id, _position, _type, **kwargs):
        self.type = _type

        self.env = env
        self.settings = _settings

        # Statistics
        self.collisions = []
        self.messages_sent = []
        self.messages_for_me = []
        self.own_payloads_sent = []
        self.own_payloads_arrived_at_gateway = []
        self.own_payloads_collided = []
        self.forwarded_payloads = []
        self.forwarded_from = []
        self.number_of_aggregated = []
        self.fairness_factors = []
        self.message_counter_own_data_and_forwarded_data = 0
        self.message_counter_only_forwarded_data = 0
        self.message_counter_only_own_data = 0
        self.message_counter_arrived_at_gateway = 0
        self._pdr = 0
        self._plr = 0

        #MAB
        self._selections = None
        self.rewards = []
        self._arm_rewards = None
        self._last_reward = 0
        self._last_arm = 0
        self.gateway_direct = False
        self._at_gateway = None
        # Buffers
        self.data_buffer = []
        self.data_created_at = 0
        self.forwarded_mgs_buffer = []
        self.route_discovery_forward_buffer = None
        self.messages_seen = collections.deque(maxlen=self.settings.MAX_SEEN_PACKETS)

        # State vars
        self.done_tx = 0
        self.start_tx = 0
        self.states_time = []
        self.states = []
        self.message_in_tx = None

        # Properties
        self.uid = _id
        self.state = None
        self.energy_mJ = 0
        self.min_hops = self.settings["MAX_ROUTE_SIZE"]
        self.initial_energy = self.settings["NODE_ENERGY"]
        self.time_to_sense = None
        self.time_spent_in = {}
        for state in NodeState:
            self.time_spent_in[state.name] = 0

        # Routing and network
        self.link_table = None
        self.route = Route()
        fixed_route = kwargs.get("fixed_route", None)
        if fixed_route is not None:
            self.route.set_fixed(fixed_route[self.uid]["via"], fixed_route[self.uid]["hops"])
        self.nodes = []  # list containing the other nodes in the network

        # Timers
        self.tx_collision_timer = None
        self.tx_route_discovery_timer = None
        self.tx_aggregation_timer = None
        self.sense_timer = None
        self.timers_setup()

        self.aggregation_timer_values = []
        self.aggregation_timer_times = []

        self.sense_until = None

        self.position = _position

        # if self.type == NodeType.GATEWAY:
        #     self.position = Position(0, 0)
        # else:
        #     self.position = Position.random(size=100)

        # Payload
        self.application_counter = 0

    def residual_energy(self):
        if self.type == NodeType.SENSOR:
            return self.initial_energy - self.energy_mJ
        return self.initial_energy

    def add_meta(self, _settings, nodes, link_table):
        self.nodes = nodes
        self.link_table = link_table
        self.settings = _settings

        #self._arm_rewards = []
        if self._selections is None:
            self._selections = np.zeros(len(self.nodes))
            self._at_gateway = np.zeros(len(self.nodes))
            self._arm_rewards = [ [] for x in range(len(self.nodes))]
            self.initial_energy = self.settings["NODE_ENERGY"]
        #self._selections = [0] * len(self.nodes)
            
    def neighbour_average_spent_energy(self):
        spent_energy = 0
        v = 0
        vicini =  [self.nodes[i] for i in self.route.get_neighbours_uid()]
        for n in vicini:
            if n.type != NodeType.GATEWAY:
                spent_energy = spent_energy + n.energy_mJ
                v += 1
        return spent_energy / v

    def state_change(self, state_to):
        if state_to is self.state and state_to is not NodeState.STATE_SLEEP:
            logging.info("mmm not possible, only sleepy can do this")
        # if self.state is None:
        #     logging.info(f"{self.uid}\tState change: None->{state_to.fullname}")
        # else:
        #     logging.info(f"{self.uid}\tState change: {self.state.fullname}->{state_to.fullname}")
        if state_to is not self.state:
            if len(self.states_time) > 0:
                self.energy_mJ += (self.env.now - self.states_time[-1]) * power_of_state(self.settings, self.state)
                self.time_spent_in[self.state.name] += (self.env.now - self.states_time[-1])
            if self.residual_energy() < 1:
                state_to = NodeState.STATE_DEPLETED
                logging.info(f"{self.uid}\tBATTERY DEPLETED")
            self.state = state_to
            self.states.append(state_to)
            self.states_time.append(self.env.now)

    def timers_setup(self):
        # Timers
        self.tx_collision_timer = TxTimer(self.env, self.settings, TimerType.COLLISION)
        if self.type == NodeType.GATEWAY:
            self.tx_route_discovery_timer = TxTimer(self.env, self.settings, TimerType.ROUTE_DISCOVERY)
        else:
            self.tx_route_discovery_timer = None
        self.tx_aggregation_timer = TxTimer(self.env, self.settings, TimerType.AGGREGATION)
        self.sense_timer = TxTimer(self.env, self.settings, TimerType.SENSE)

    def run(self):
        random_wait = np.random.uniform(0, self.settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        yield self.env.timeout(random_wait)
        logging.info(f"{self.uid}\tStarting node {self.uid}")

        self.timers_setup()  # Reset timers

        if self.type == NodeType.GATEWAY:
            self.tx_route_discovery_timer.backoff = 1.1 * self.settings.MAX_DELAY_START_PER_NODE_RANDOM_S
            self.tx_route_discovery_timer.start()
        else:
            self.sense_timer.start()

        self.state_change(NodeState.STATE_INIT)
        self.env.process(self.periodic_wakeup())

    def check_sensing(self):
        if (self.sense_until is None and self.sense_timer.is_expired()) or \
                (self.sense_timer.is_expired() and self.env.now < self.sense_until):
            self.state_change(NodeState.STATE_SENSING)
            yield self.env.timeout(self.settings.MEASURE_DURATION_S)
            # schedule timer for transmit
            logging.info(f"{self.uid}\tSensing")
            self.tx_aggregation_timer.start(restart=False)
            self.sense_timer.start()

            if len(self.data_buffer) == 0:
                self.data_created_at = self.env.now
            self.data_buffer.extend(self.application_counter.to_bytes(self.settings.MEASURE_PAYLOAD_SIZE_BYTE, 'big'))

            self.application_counter = (self.application_counter + 1) % 65535

    def check_transmit(self):
        # Route discovery messages are sent separately
        if self.type == NodeType.GATEWAY:
            if self.tx_route_discovery_timer.is_expired():
                self.route_discovery_forward_buffer = \
                    Message(self.settings, MessageType.TYPE_ROUTE_DISCOVERY, 0, 0, 0, 0, [0x55, 0x55, 0x55], self.env.now, self, [])
                yield self.env.process(self.tx())
                self.tx_route_discovery_timer.backoff = self.settings.ROUTE_DISCOVERY_S
                self.tx_route_discovery_timer.reset()
                #

        elif self.type == NodeType.SENSOR:
            # if route discovery message need to be forwarded (because of collision timer)
            if self.tx_collision_timer.is_expired():
                yield self.env.process(self.tx())
                self.tx_collision_timer.reset()

            # elif to ensure beacon TX is not directly followed by a data TX
            # TODO ensure beacon TX does not throttle data TX
            elif self.tx_aggregation_timer.is_expired() or self.full_buffer():
                yield self.env.process(self.tx())
                self.tx_aggregation_timer.reset()

    def postpone_pending_tx(self):
        # If cad happened before when we should be tx'en, postpone using collision timer
        if self.tx_collision_timer.is_expired() or self.tx_aggregation_timer.is_expired():
            self.tx_collision_timer.start(restart=True)  # Postpone using collision timer
            self.tx_aggregation_timer.reset()  # Reset aggregation timer to stop premature sending

    def periodic_wakeup(self):
        while True:
            if self.state == NodeState.STATE_DEPLETED:
                break
            self.state_change(NodeState.STATE_SLEEP)
            cad_interval = self.settings.CAD_INTERVAL + random(self.settings.CAD_INTERVAL_RANDOM_S)
            yield self.env.timeout(cad_interval)

            cad_detected = yield self.env.process(self.cad())

            if cad_detected:
                packet_for_us = yield self.env.process(self.receiving())
                self.postpone_pending_tx()
            else:
                yield self.env.process(self.check_transmit())

            if self.type != NodeType.GATEWAY:
                yield self.env.process(self.check_sensing())

    def receiving(self):
        """
        After activity during CAD in periodic_cad, we will listen to incoming packets from other nodes

        :return:
        """
        packet_for_us = False
        self.state_change(NodeState.STATE_RX)

        loudest_node = None
        loudest_node_rx_message = None
        rx_message = None
        collision_happened = False
        current_active_nodes = self.get_nodes_in_state([NodeState.STATE_PREAMBLE_TX, NodeState.STATE_TX])
        while len(current_active_nodes) > 0:  # As long as someone is sending

            # Check how is the loudest
            current_loudest_node = self.check_and_handle_collisions(current_active_nodes)

            # If new current loudest node
            # Check if he just started tx'en
            #   Ok great, this is our new champion
            # Else, put None

            # If current loudest node is the same, check if he is done tx
            #   Ok great process rx msg
            if current_loudest_node is not None:
                if loudest_node != current_loudest_node:
                    if current_loudest_node.state == NodeState.STATE_TX and \
                            math.isclose(current_loudest_node.start_tx, self.env.now, abs_tol=0.01):
                        loudest_node = current_loudest_node
                        loudest_node_rx_message = loudest_node.message_in_tx
                    else:
                        loudest_node = None
                elif loudest_node_rx_message is not None \
                        and self.env.now >= loudest_node.done_tx - loudest_node_rx_message.time() / 10 \
                        and self.link_table.get_from_uid(self.uid, loudest_node.uid).in_range():
                    rx_message = loudest_node.message_in_tx

            if rx_message is None:
                # If no message is yet found (thus no nodes yet in TX that have not collided)
                # Go to next tx cycle

                wait = 0
                # Look for nodes in tx
                nodes_in_tx = self.get_nodes_in_state([NodeState.STATE_TX])
                if len(nodes_in_tx) == 0:
                    # No nodes yet in tx, wait until one starts transmitting
                    earliest_transmitting = None
                    for node in current_active_nodes:
                        if earliest_transmitting is None or node.start_tx < earliest_transmitting:
                            earliest_transmitting = node.start_tx
                    wait = max(0.001, earliest_transmitting - self.env.now)
                else:
                    # Someone is in tx, proceed in small steps and preferably until right before tx ends

                    # Standard: until next event with peek(), but at least 1ms wait
                    wait = max(0.001, self.env.peek() - self.env.now)

                    # Look for nodes that are done transmitting, if so: we need to move in time to just before tx_done
                    earliest_done = None
                    for node in nodes_in_tx:
                        if earliest_done is None or node.done_tx < earliest_done:
                            toa = node.message_in_tx.time()
                            earliest_done = node.done_tx - toa / 10

                    # Decide which "wait" to take: lowest value, but at least 1ms
                    wait = max(0.001, min(wait, earliest_done - self.env.now))

                yield self.env.timeout(wait)
                current_active_nodes = self.get_nodes_in_state([NodeState.STATE_PREAMBLE_TX, NodeState.STATE_TX])
            else:
                break

        # Collision did not happen during RX
        if rx_message is not None:
            in_range = False
            if rx_message.payload.own_data.src == self.uid:
                in_range = True
            if rx_message.is_routed():
                in_range = self.link_table.get_from_uid(rx_message.payload.own_data.src, self.uid).in_range()
            elif rx_message.is_route_discovery():
                in_range = self.link_table.get_from_uid(rx_message.header.address, self.uid).in_range()

            if in_range:
                packet_for_us = self.handle_rx_msg(rx_message)

        return packet_for_us

    def tx(self, route_discovery: bool = False):
        if self.route_discovery_forward_buffer is not None:
            route_discovery = True
            self.message_in_tx = self.route_discovery_forward_buffer

        elif len(self.data_buffer) > 0 or len(self.forwarded_mgs_buffer) > 0:
            # Init message for forwarding
            hops = 0
            lqi = 0
            if len(self.forwarded_mgs_buffer) > 0:
                hops = self.forwarded_mgs_buffer[0].header.hops
                lqi = self.forwarded_mgs_buffer[0].header.cumulative_lqi

            self.message_in_tx = Message(self.settings,
                                         MessageType.TYPE_ROUTED,
                                         hops,
                                         0,
                                         0, # Route gets updated later
                                         self.uid,
                                         self.data_buffer,
                                         self.data_created_at,
                                         self,
                                         self.forwarded_mgs_buffer)

            # Get exclude list from all nodes that already forwarded
            exclude = []
            for p in self.message_in_tx.payload.forwarded_data:
                for u in p.trace:
                    if u not in exclude:
                        exclude.append(u)

            route = None
            id_vicini = self.route.get_neighbours_uid()
            max_upper_bound = -10000
            reward = 0
            self.gateway_direct = False
            if 0 in id_vicini:
                self.gateway_direct = True
                route = 0
                self._selections[route] = self._selections[route] + 1
                self._last_arm = route
                self.message_in_tx.header.address = route
            '''else:
                if self._last_arm == 0:
                    lost_pkt = True'''
            if self.type == NodeType.SENSOR and len(self.route.neighbour_list)>0 and not self.gateway_direct:
                if np.sum(self._selections)>0:
                    #print(self.messages_sent)
                    #print(collision)
                    collided = self.messages_sent[-1].collided
                    #print(collided)
                    if not collided:
                        link = self.link_table.get_from_uid(self._last_arm, self.uid)
                        tdr = link.tdr()
                        vicini =  [self.nodes[i] for i in id_vicini]
                        #print("Il mio ultimo arm ",self._last_arm)
                        #hop = self.route.find_node(self._last_arm)
                        #hop = hop["hops"]
                        best_node = self.route.find_best()
                        #best = 0
                        #if int(best_node["uid"]) == self._last_arm:
                         #print("best selected")
                         #best = 1
                        min_hop = 0
                        energy_factor = 1
                        if self.route.find_node(self._last_arm)["hops"] <= best_node["hops"]:
                            #print("WE HAVE NOT SELECTED MIN HOP")
                            min_hop = 1
                        if self._last_arm != 0:
                            res_en = self.nodes[self._last_arm].residual_energy()
                            print("Node", self.uid, "residual energy", self.residual_energy(),"\tSelected node ", self._last_arm, " residual energy:", res_en)
                            print(self.nodes[self._last_arm].route)
                            listone = list(set(self.nodes[self._last_arm].route.get_neighbours_uid()) - {0})
                            print("Listone old:", listone)
                            #listone = list(set(self.nodes[self._last_arm].route.find_closer(self.nodes[self._last_arm].min_hops)) - {0})
                            a = (x for x in listone if 0 not in self.nodes[x].route.get_neighbours_uid())
                            listone = [x for x in a]
                            print("Listone new:", listone)
                            #programPause = input("Press the <ENTER> key to continue...")                
                            if len(listone) > 0:
                            #listone = self.nodes[self._last_arm].route.get_neighbours_uid()
                                energy_factor = 1 - abs(1/self.initial_energy*(res_en - statistics.fmean([self.nodes[x].residual_energy() for x in listone])))
                        print("Fairness factor:", energy_factor)
                        self.fairness_factors.append(energy_factor)
                        pdr_next_hop = 0
                        if self._at_gateway[self._last_arm] > 0:
                            pdr_next_hop = self._at_gateway[self._last_arm]/self._selections[self._last_arm]
                        if self.settings["REWARD"] is None or self.settings["REWARD"] == "magellan":
                            #reward = 0.5*tdr + 0.3*energy_factor - 0.1 * hop
                            reward = 0.4*tdr + 0.5*energy_factor + 0.1 * min_hop
                        elif self.settings["REWARD"] == "mine":
                        #reward = 0.25*tdr + 0.4*energy_factor + 0.1 * min_hop + 0.25*pdr_next_hop
                        reward = min_hop
                        elif reward
                        reward = energy_factor
                        #reward = tdr
                    self._arm_rewards[self._last_arm].append(reward)
                    self.rewards.append(reward)
               #print("Excluded ", exclude)
                if self.settings.ALGORITHM=="random":
                    possibili = list(set(id_vicini) - set(exclude))
                    if len(possibili) < 1:
                        self.message_in_tx.header.address = 0
                        #print(id_vicini)
                        #print(exclude)
                        #print(possibili)
                        #self.state_change(NodeState.STATE_SLEEP)
                        #self.done_tx = None
                        #self.message_in_tx = None
                        #self.forwarded_mgs_buffer = []
                        #self.data_buffer = []
                        #return
                    else:
                        route = rnd.choice(possibili)
                elif self.settings.ALGORITHM=="mab":
                    print("Min Hops:", self.min_hops)
                    for k in range(len(self.route.neighbour_list)):
                        upper_bound = -10000
                        #print(self.route.neighbour_list)
                        i = int(self.route.neighbour_list[k]["uid"])
                        #print("Provo", i)
                        #if i not in exclude and int(self.route.neighbour_list[k]["hops"]) < self.min_hops:
                        if i not in exclude and int(self.route.neighbour_list[k]["hops"]) < self.min_hops:
                            if self._selections[i] > 0 and len(self.rewards) > 0:
                                    delta_i = math.sqrt(3/2 * math.log(len(self.rewards)) / self._selections[i])
                                    upper_bound = statistics.fmean(self._arm_rewards[i]) + delta_i
                            else:
                                route = i
                                print(f"PROVO {i} PER LA PRIMA VOLTA")
                                break
                            if upper_bound > max_upper_bound:
                                max_upper_bound = upper_bound
                                route = i
                    if route == None:
                        print("Sono il nodo ", self.uid)
                        print("Vicini: ", self.route.neighbour_list)
                        print("Escludi: ", exclude)
                        print("Min Hops", self.min_hops)
                        print("Selezioni:", np.sum(self._selections))
                        #programPause = input("Press the <ENTER> key to continue...")                
                elif self.settings.ALGORITHM=="original":
                    route = self.route.find_route(exclude)
                    route = int(route["uid"])
                
                if route is None:
                    self.message_in_tx.header.address = 0
                else:
                    self._selections[route] = self._selections[route] + 1
                    self.message_in_tx.header.address = route
                    self._last_arm = route
                    #print("MAB Route", route)
                print("Scelgo: ",route)
            
            
            if self.type == NodeType.GATEWAY:
                route = self.route.find_route(exclude)
                if route is None:
                    self.message_in_tx.header.address = 0
                else:
                    route = int(route["uid"])
                    self.message_in_tx.header.address = route
                    self._selections[route] = self._selections[route] + 1
            # Increase counters and adjust lqi if forward
            if len(self.message_in_tx.payload.forwarded_data) > 0:
                self.message_in_tx.hop(self)
                self.message_in_tx.header.cumulative_lqi += \
                    self.link_table.get_from_uid(self.uid, self.forwarded_mgs_buffer[0].payload.own_data.src).lqi()

        if self.message_in_tx is not None:
            yield self.env.timeout(rnd.uniform(0,1))
            self.state_change(NodeState.STATE_PREAMBLE_TX)
            packet_time = self.message_in_tx.time()
            self.start_tx = self.env.now + self.settings.PREAMBLE_DURATION_S
            self.done_tx = self.start_tx + packet_time

            yield self.env.timeout(self.settings.PREAMBLE_DURATION_S)

            self.state_change(NodeState.STATE_TX)
            logging.info(
                f"{self.uid}\t Sending packet {self.message_in_tx.header.uid} with size: {self.message_in_tx.size()} bytes")
            logging.info(f"{self.uid}\tTx packet to {self.message_in_tx.header.address} {self.message_in_tx}")

            if self.message_in_tx.is_routed() and len(self.message_in_tx.payload.forwarded_data) == 0:
                self.tx_aggregation_timer.step_down()
                self.aggregation_timer_values.append(self.tx_aggregation_timer.backoff)
                self.aggregation_timer_times.append(self.env.now)

            # Sent statistics
            self.messages_sent.append(self.message_in_tx)
            # Only for routed messages
            if self.message_in_tx.header.type == MessageType.TYPE_ROUTED:
                self.link_table.get_from_uid(self.uid, self.message_in_tx.header.address).use()

                if self.message_in_tx.payload.own_data.len > 0:
                    if self.message_in_tx.payload.own_data.size() - 3 > settings.MEASURE_PAYLOAD_SIZE_BYTE:
                        for i in range(0, math.floor((self.message_in_tx.payload.own_data.size() - 3) / settings.MEASURE_PAYLOAD_SIZE_BYTE)):
                            cp = self.message_in_tx.payload.own_data.copy()
                            cp.clip(i)
                            self.own_payloads_sent.append(cp)
                    else:
                        self.own_payloads_sent.append(self.message_in_tx.payload.own_data)

                for pl in self.message_in_tx.payload.forwarded_data:
                    self.forwarded_payloads.append(pl)

                if self.message_in_tx.payload.own_data.len > 0 and len(self.message_in_tx.payload.forwarded_data) > 0:
                    self.message_counter_own_data_and_forwarded_data += 1
                elif self.message_in_tx.payload.own_data.len > 0:
                    self.message_counter_only_own_data += 1
                elif len(self.message_in_tx.payload.forwarded_data) > 0:
                    self.message_counter_only_forwarded_data += 1

                self.number_of_aggregated.append(len(self.forwarded_mgs_buffer))

            yield self.env.timeout(packet_time)
            self.state_change(NodeState.STATE_SLEEP)  # Go back to sleep after transmit

            self.done_tx = None
            self.message_in_tx = None
            self.forwarded_mgs_buffer = []
            self.data_buffer = []

            if route_discovery:
                self.route_discovery_forward_buffer = None

    def cad(self):
        """
        In the CAD state, the node listens for channel activity
        In the beginning it needs to wake-up and stabilise
        After that all messages sent in the CAD window, will be considered received (if power level is above sensitivity)

        Depending on CAD success the node enters RX state or sleep state
        """
        self.state_change(NodeState.STATE_CAD)
        yield self.env.timeout(self.settings.TIME_CAD_WAKE_S + self.settings.TIME_CAD_STABILIZE_S)

        # check which nodes are now in PREAMBLE_TX state
        nodes_sending_preamble = self.get_nodes_in_state([NodeState.STATE_PREAMBLE_TX])
        active_nodes = []
        for n in nodes_sending_preamble:
            if self.link_table.get(self, n).in_range():
                active_nodes.append(n)

        # check after CAD perform, if it was transmitting during the full window
        yield self.env.timeout(self.settings.TIME_CAD_PERFORM_S)

        cad_detected = False
        for n in active_nodes:
            if n.state is NodeState.STATE_PREAMBLE_TX:
                # OK considered TX and we need to listen
                cad_detected = True
                break
        yield self.env.timeout(self.settings.TIME_CAD_PROC_S)
        return cad_detected

    def full_buffer(self):
        size = 0
        for fw_message in self.forwarded_mgs_buffer:
            size += fw_message.payload.size()

        return size > self.settings.MAX_BUF_SIZE_BYTE*self.settings.MAX_BUF_SIZE_THRESHOLD

    def get_nodes_in_state(self, states):
        nodes = []
        for n in self.nodes:
            if n is not self and n.state in states:
                nodes.append(n)
        return nodes

    def check_and_handle_collisions(self, active_nodes):
        active_node = active_nodes[0]
        if len(active_nodes) > 1:
            # If power higher than power_threshold for all tx nodes, this one will succeed
            power_threshold = self.settings.LORA_POWER_THRESHOLD  # dB
            powers = [(a, self.link_table.get(a, self).rss()) for a in active_nodes]
            powers.sort(key=lambda tup: tup[1], reverse=True)

            # Collisions only when loudest is in TX
            # Only success for the highest power if > power_threshold
            if powers[0][0].state == NodeState.STATE_TX and powers[0][1] >= powers[1][1] + power_threshold:
                # Collision did not happen (due to power threshold)
                active_node = powers[0][0]
            else:
                active_node = None
                # Collision happened, only save for tx nodes and for messages directed at me
                for node in active_nodes:
                    if node.state == NodeState.STATE_TX and node.message_in_tx.is_route_discovery() \
                            and not node.message_in_tx.collided:
                        logging.info(f"{self.uid}\t Route discovery collision detected")
                        self.collisions.append(self.env.now)
                        node.message_in_tx.handle_collision()
                    elif node.state == NodeState.STATE_TX and node.message_in_tx.is_routed() and \
                            node.message_in_tx.header.address == self.uid and not node.message_in_tx.collided:
                        logging.info(f"{self.uid}\t Routed collision detected that was addressed to us")
                        node.message_in_tx.handle_collision()
                        self.link_table.get(node, self).collision()
                        self.collisions.append(self.env.now)

        return active_node

    def collided(self, pl):
        # Callback for packets that were sent by me and have collided
        self.own_payloads_collided.append(pl)

    def handle_rx_msg(self, rx_packet):
        packet_for_us = False
        update_beacon = False

        # Check for routing in all received route discovery messages
        if rx_packet.is_route_discovery():
            # TODO: quick and dirty fix, make better and why is this needed?
            if self.link_table.get_from_uid(self.uid,
                                            rx_packet.header.address).in_range():
                logging.info(f"{self.uid}\tRx packet from {rx_packet.payload.own_data.src} {rx_packet}")
                self.route.update(rx_packet.header.address,
                                  self.link_table.get_from_uid(self.uid,
                                                               rx_packet.header.address).snr(),
                                  rx_packet.header.cumulative_lqi + self.link_table.get_from_uid(self.uid,
                                                                                                 rx_packet.header.address).lqi(),
                                  rx_packet.header.hops)
                self.min_hops = self.route.min_hop()
                logging.info(f"{self.uid}\r\n{self.route}")

        if rx_packet.header.uid in self.messages_seen:
            logging.info(
                f"{self.uid}\tPacket already processed {rx_packet.header.uid}")
            return False
        else:
            if rx_packet.is_route_discovery():
                packet_for_us = True
                self.route_discovery_forward_buffer = rx_packet.copy()
                # Update new packet that needs to be forwarded
                self.route_discovery_forward_buffer.header.address = self.uid
                self.route_discovery_forward_buffer.hop(self)
                self.route_discovery_forward_buffer.header.cumulative_lqi += \
                    self.link_table.get_from_uid(self.uid, rx_packet.header.address).lqi()
                self.tx_collision_timer.start(restart=True)

            elif rx_packet.is_routed() and rx_packet.header.address == self.uid:
                packet_for_us = True
                logging.info(f"{self.uid}\tRx packet from {rx_packet.payload.own_data.src} {rx_packet}")
                # update tx timer
                self.messages_for_me.append(rx_packet)
                if self.type == NodeType.SENSOR:
                    self.tx_aggregation_timer.start(restart=False)
                    self.tx_aggregation_timer.step_up()  # Is only applied for next start

                    self.aggregation_timer_values.append(self.tx_aggregation_timer.backoff)
                    self.aggregation_timer_times.append(self.env.now)

                    self.forwarded_mgs_buffer.append(rx_packet)
                    self.forwarded_from.append(rx_packet.payload.own_data.src)

                elif self.type == NodeType.GATEWAY:
                    rx_packet.arrived_at_gateway()

            self.messages_seen.append(rx_packet.header.uid)

        return packet_for_us

    def arrived_at_gateway(self, payload):
        self.message_counter_arrived_at_gateway += 1
        # Update statistics
        if payload.size()-3 > settings.MEASURE_PAYLOAD_SIZE_BYTE:
            for i in range(0, math.floor((payload.size()-3)/settings.MEASURE_PAYLOAD_SIZE_BYTE)):
                cp = payload.copy()
                cp.clip(i)
                self.own_payloads_arrived_at_gateway.append(cp)
        else:
            self.own_payloads_arrived_at_gateway.append(payload)
        if self.uid in payload.trace:
            #print(self.uid)
            #print(payload)
            #print(payload.src)
            #programPause = input("Press the <ENTER> key to continue...")
            uid = 0
            if len(payload.trace) > 1:
                idx = payload.trace.index(self.uid) + 1
                uid = payload.trace[idx]
            self._at_gateway[uid] = self._at_gateway[uid] + 1

    def pdr(self):
        if len(self.own_payloads_sent) > 0:
            self._pdr = len(self.own_payloads_arrived_at_gateway) / len(self.own_payloads_sent)
        else:
            self._pdr = 1
        return self._pdr

    def plr(self):
        if len(self.own_payloads_sent) > 0:
            self._plr = len(self.own_payloads_collided) / len(self.own_payloads_sent)
        else:
            self._plr = 0
        return self._plr

    def aggregation_efficiency(self):
        if len(self.messages_sent) > 0:
            only_own_payload_sent = 0
            for message in self.messages_sent:
                if len(message.payload.forwarded_data) == 0:
                    only_own_payload_sent += 1
            return 1 - only_own_payload_sent / len(self.messages_sent)
        else:
            return 1

    def energy(self):
        return self.energy_mJ

    def energy_per_byte(self):
        l = 0
        for own in self.own_payloads_sent:
            l += len(own.data)
        for forwarded in self.forwarded_payloads:
            l += len(forwarded.data)
        return self.energy_mJ/l

    def energy_tx_per_byte(self):
        l = 0
        for own in self.own_payloads_sent:
            l += len(own.data)
        for forwarded in self.forwarded_payloads:
            l += len(forwarded.data)

        energy = self.time_spent_in[NodeState["STATE_PREAMBLE_TX"].name] * power_of_state(self.settings, NodeState["STATE_PREAMBLE_TX"])
        energy += self.time_spent_in[NodeState["STATE_TX"].name] * power_of_state(self.settings, NodeState["STATE_PREAMBLE_TX"])
        return energy/l

    def energy_per_state(self):
        ret = {}
        for state in NodeState:
            ret[state.name] = self.time_spent_in[state.name] * power_of_state(self.settings, state)
        return ret

    def latency(self):
        latencies = []
        for payload in self.own_payloads_arrived_at_gateway:
            latencies.append(payload.arrived_at - payload.created_at)
            if payload.arrived_at == 0:
                print("Trouble")

        return latencies

    def plot_states(self, axis=None, plot_labels=True):
        import matplotlib.pyplot as plt

        state_time = []
        states = []

        for i, (t, s) in enumerate(zip(self.states_time, self.states)):
            states.append(int(s.value))
            state_time.append(t)

            if i < len(self.states_time) - 1:
                # state actually till next transition time
                states.append(int(s.value))
                state_time.append(self.states_time[i + 1])

        if axis is None:
            axis = plt.subplot()

        if self.type == NodeType.GATEWAY:
            axis.plot(state_time, states, color="red")
        else:
            axis.plot(state_time, states)
        axis.scatter(self.collisions, [0] * len(self.collisions), edgecolor="green")
        axis.grid()

        # if plot_labels:
        #
        #     states = sorted(self.states, key=lambda tup: tup.value)
        #     y = [s.value for s in states]
        #     labels = [f"{s.fullname}" for s in states]
        #
        #     axis.set_yticks(y, labels)
        if axis is None:
            plt.show(block=False)
