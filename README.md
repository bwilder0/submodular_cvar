# Overview
This repository contains code for the paper:

Bryan Wilder. Risk-sensitive submodular optimization. AAAI Conference on Artificial Intelligence. 2018. 

See http://teamcore.usc.edu/people/bryanwilder/default.htm for the paper. 

```
@inproceedings{wilder2018risk,
 author = {Wilder, Bryan},
 title = {Risk-sensitive submodular optimization},
 booktitle = {Proceedings of the 32nd AAAI Conference on Artificial Intelligence},
 year = {2018}
}
```

Included is code for the RASCAL algorithm, the baseline Frank-Wolfe algorithm for maximizing expected utility, and code for implementing the sensor allocation domain described in the paper.

# Dependencies
* NetworkX, for generating scenarios under the continuous time independent cascade model.
* NumPy
* The BWSN utilities, for generating scenarios for that domain. See http://www.water-simulation.com/wsp/about/bwsn/
* pandas, for parsing the output of the BWSN program.
