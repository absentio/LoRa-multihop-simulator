from multihop.Network import *
import matplotlib.pyplot as plt
import random
import logging
import datetime
from multihop.config import settings
from multihop.utils import merge_data
from multihop.preambles import preambles

settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]

logging.getLogger().setLevel(logging.INFO)

#random.seed(5555)
#np.random.seed(19680801)
num_sim = 1
pdrs = []
avg_en = []
std = []
avg_hops = []
start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#network = Network.load("results/2024-02-26_13-06-29_network.dill")
#link = network.link_table.get_from_uid(0,4)
#print(link.in_range())
#network.plot_network()
#network.plot_network()
#network.save()

for i in range(num_sim):
    random.seed(5555 + num_sim)
    np.random.seed(19680801 + num_sim)
    network = Network(settings = settings)
    network.run()
    #network.save()
    pdr = network.pdr()
    pdrs.append(pdr)
    network.tdr()
    hops = network.hops()
    avg_hops.append(hops)
    en, dev = network.energy()
    avg_en.append(en)
    std.append(dev)

end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with open(f"results/{start_time}_{settings['NETWORK_NODES']}_{settings['ALGORITHM']}.txt", "a") as f:
    if settings['ALGORITHM'] == "mab":
                print("Reward:", settings["REWARD"], file=f)
    print("PDR :", statistics.fmean(pdrs), file = f)
    print("hops :", statistics.fmean(avg_hops), file = f)
    print("en :", statistics.fmean(avg_en), file = f)
    print("dev :", statistics.fmean(std), file = f)



#network.plot_network()

#network = Network.load("results/2024-02-14_16-54-37_network.dill")
#network.plot_network()

#network.plot_hops_statistic("pdr", type="cdf")
#network.plot_hops_statistic("plr")
#network.plot_hops_statistic("aggregation_efficiency", type="cdf")
#network.plot_hops_statistic("energy_per_byte", type="cdf")
#network.plot_hops_statistic("energy_tx_per_byte", type="cdf")
#network.plot_hops_statistic("latency")
#network.plot_hops_statistic_energy_per_state()

#network.save_settings()

print("End time:", end_time)
