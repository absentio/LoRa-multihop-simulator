import matplotlib.pyplot as plt
import numpy as np
values10 = [207744.46536318772,232424.6282768704,176802.55765666344,232820.254809559,217778.2425876144,177268.93577462446]
values15 = [240050.3495418825,252684.58680577722,210334.47476256845,249871.92992938598,235368.2852269972,200551.92410014928]
values20 = [270384.8662942459,296488.1036247372,217197.3434265189,278554.8509548547,268484.5432227229,207114.06088328382]
values30 = [319826.2686761372,326615.05317862704,250714.56481802152,321250.04420449917,308627.2461706281,244410.20475914658]
values = [x / 1000 for x in values10]
std10 = [43388.48852460458,50297.74299880352,50349.848957178045, 54454.68558425259,57456.8165717536,51877.21125967522]
std15 = [89843.61002405167,95843.24506947305,84771.97947624973,92948.9576182477,84094.87796259762,75568.00092193292]
std20 = [77086.92361704735, 77523.86553902626,81631.72241908144,82794.98632633842,85811.24726799154,80732.47556725811]
std30 = [58879.59022403443,54412.24178004979,89080.0067156957,59436.10494041607,67020.52347927706,90946.75825767987]
std = [x / 1000 for x in std10]
ymn = [min(values10),min(values15), min(values20)]
ymin = min(ymn) / 1000
ymx = [max(values10),max(values15), max(values20)]
ymax = max(ymx) / 1000
#print(ymin, ymax)
plt.figure()
#plt.ylim(650,InitialEnergy)
y = np.arange(6)
xticks= ['MAGELLAN','MaxF', 'MinE', 'MaxR', 'Random', 'LMHP']
plt.xlabel("Approach")
plt.ylabel("Node average consumed energy [J]")
plt.xticks(y,xticks)
patterns = [ "*" , "x" , "o", "\\", "..","/", "+", "\\" ]
color = ['tab:blue', 'tab:orange', 'tab:green','tab:gray', 'tab:red', 'tab:purple', 'tab:brown']
for i, value in enumerate(values):
  plt.bar(y[i], value, color = color[i], edgecolor='black', hatch = patterns[i])
plt.errorbar(y, values, yerr=std, linestyle='none', color='black', lw = 2, capsize=10, markeredgewidth=2)
plt.ylim(ymin - 90, ymax + 90)
plt.savefig("istogramma_energia15.pdf", bbox_inches='tight')
delivered_packets10 = [2232,1959,2290,2242,2238,2078]
delivered_packets15 = [3387,3267,3430,3383,3252,3162]
delivered_packets = [x / max(delivered_packets10) for x in delivered_packets10]
en_pkt = np.array(values) / np.array(delivered_packets)
print(en_pkt)
plt.figure()
y = np.arange(6)
xticks= ['MAGELLAN','MaxF', 'MinE', 'MaxR', 'Random', 'LMHP']
plt.xlabel("Approach")
plt.ylabel("Node average consumed energy [J]")
plt.xticks(y,xticks)
patterns = [ "*" , "x" , "o", "\\", "..","/", "+", "\\" ]
color = ['tab:blue', 'tab:orange', 'tab:green','tab:gray', 'tab:red', 'tab:purple', 'tab:brown']
for i, value in enumerate(en_pkt):
  plt.bar(y[i], value, color = color[i], edgecolor='black', hatch = patterns[i])
#plt.errorbar(y, values, yerr=std, linestyle='none', color='black', lw = 2, capsize=10, markeredgewidth=2)
plt.savefig("istogramma_energia_per_pacchetto15.pdf", bbox_inches='tight')

