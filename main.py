from matplotlib import pyplot as plt

from Nodes import Gateway, SensorNode
from Links import LinkTable
from config import settings
import simpy

print(settings.POWER_CAD_CYCLE_mW)
print(settings.POWER_SENSE_mW)

simpy_env = simpy.Environment()

nodes = []

gw = Gateway(simpy_env, 0)
nodes.append(gw)

number_of_nodes = 10
for x in range(1, number_of_nodes+1):
    node = SensorNode(simpy_env, x)
    nodes.append(node)

link_table = LinkTable(nodes)

for node in nodes:
    if type(node) is SensorNode:
        node.add_meta(nodes, link_table)

for node in nodes:
    simpy_env.process(node.run())

simpy_env.run(until=10 * 60)

fig, ax = plt.subplots(number_of_nodes+1, sharex=True, sharey=True)
for i, node in enumerate(nodes):
    node.plot_states(ax[i])

ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6], ["INIT", "ZZZ", "CAD", "RX", "SNS", "P_TX", "TX"])
plt.show()
