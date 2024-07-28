# 2023-07-04
 - 1
     - ucb, just zooming in, '-s 0.025'
 - 2
     - submitted the same value (0.701658-0.102197) to get a noise estimate
 - 3
     - ucb, just zooming in, '-s 0.025'
 - 4
     - lonliest search expansion, might be in a local minima
 - 5
     - stuck on an edge, UCB with a bit of manual assist
 - 6
     - UCB expanded the search , radius = 0.2
 - 7
     - ucb with radius = 0.05
 - 8
     - searaching in the area, dropping the beta to 0.5
     - ucb with radius = 0.05

# 2023-07-07
 - 1
     - last sample dropped quite a bit
     - ucb, zooming in, '-s 0.01’
 - 2
     - submitted the same value (0.701658-0.102197) to get a noise estimate
 - 3
     - ucb, just zooming in, '-s 0.025'
 - 4
     - usb search expansion, might be in a local minima
 - 5
     - usb search, 3 values at ~1.0!
     - hand tweaked it for a void below the max sample in 3 dims
 - 6
     - good result last sample
     - UCB  radius = 0.2 beta = 0.5 
 - 7
     - ucb with radius = 0.05, beta = 0.5
 - 8
    - searaching in the area, dropping the beta to 0.5
     - ucb with radius = 0.05

# 2023-07-11
- 1
   - last sample dropped quite a bit
       - ucb, zooming in, '-s 0.01’, zero beta
       - going off into odd places
       - hand forcing a void near
           - some way of finding the lonliest in a sub section of the space
- 2
   - trying the white noise kernel, not helping
   - ucb’d the whole grid again
- 3
 - ucb’d with -s 0.1 and beta = 0.5
- 4 
   - usb search expansion, might be in a local minima. r = 0.2, beta = 1.96
- 5
   - ucb, 1.96, 0.01
   - hand tweaked it for a void below the max sample in 3 dims
- 6
  - good result last sample
  - ucb radius = 0.15 beta = 0.5
- 7
  - good result last sample
  - ucb radius = 0.05 beta = 0.5
- 8
    - searaching in the area, dropping the beta to 0.5
    - ucb with radius = 0.05

# 2023-07-13
- 1
     - good result
     - moved it into a vacant space by hand
- 2
     - sucked, ucbing
- 3
     - ucb’d with -s 0.1 and beta = 0.5 - 4 
- 4
    - last  search got second best, sampling around that with a tight UCB, 
- 5
    - think I’ve been in a local minima for ages
    - widening it
- 6   
    - mediocre result last sample
    - ucb radius = 0.15 beta = 0.5
- 7
    - mediocre result last sample
    - ucb radius = 0.05 beta = 0.5
- 8
    - searaching in the area, dropping the beta to 0.5
    - ucb with radius = 0.05

# 2024-07-19
- 1
     - bad result
     - using turbo on small area 0.01
- 2
     - sucked, sampled the hole at 0.6/0.6
     - 0.708173-0.010928
- 3
     - sucked!
     - ucb’d window 0.1
- 4
    - last  search got second best, sampling around that with a tight UCB, 
- 5
    - think I’ve been in a local minima for ages
    - widening it to 0.02
- 6   
    - mediocre result last sample
    - turno, R = 0.15
    - t
- 7
    - best result last sample, turning it instead
    - ucb radius = 0.05 beta = 0.5
- 8
    - best result last sample, turbo it instead
    - turbo with radius = 0.05


# 2024-07-25
