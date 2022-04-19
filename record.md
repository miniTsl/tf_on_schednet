### IC3net最后的训练结果（TJ medium）每个epoch打印一次结果
Epoch 1993      Reward [-4.12 -4.21 -4.02 -4.13 -4.24 -4.14 -4.31 -4.24 -4.29 -4.22]    Time 32.32s
Add-Rate: 0.20
Success: 0.56
Steps-taken: 40.00
Epoch 1994      Reward [-4.24 -4.37 -4.26 -4.32 -4.28 -4.19 -4.36 -4.33 -4.35 -4.33]    Time 32.01s
Add-Rate: 0.20
Success: 0.53
Steps-taken: 40.00
Epoch 1995      Reward [-4.21 -4.06 -4.04 -4.08 -4.17 -4.23 -4.13 -4.13 -4.13 -3.94]    Time 32.10s
Add-Rate: 0.20
Success: 0.55
Steps-taken: 40.00
Epoch 1996      Reward [-4.   -4.03 -4.02 -4.1  -4.07 -4.16 -3.93 -3.96 -4.12 -4.29]    Time 31.56s
Add-Rate: 0.20
Success: 0.57
Steps-taken: 40.00
Epoch 1997      Reward [-4.16 -4.14 -4.14 -4.09 -4.38 -4.11 -3.87 -4.11 -4.1  -4.19]    Time 31.38s
Add-Rate: 0.20
Success: 0.56
Steps-taken: 40.00
Epoch 1998      Reward [-4.21 -4.25 -4.19 -4.28 -4.17 -4.08 -4.2  -4.2  -4.12 -4.11]    Time 31.59s
Add-Rate: 0.20
Success: 0.53
Steps-taken: 40.00
Epoch 1999      Reward [-4.09 -4.05 -4.15 -4.05 -4.1  -4.09 -4.09 -4.19 -4.07 -4.05]    Time 31.95s
Add-Rate: 0.20
Success: 0.54
Steps-taken: 40.00
Epoch 2000      Reward [-4.05 -4.18 -4.1  -4.02 -4.07 -4.14 -4.08 -4.19 -4.21 -4.17]    Time 32.05s
Add-Rate: 0.20
Success: 0.54
Steps-taken: 40.00

### tf_on_schednet
#### debug_0
[Train_epoch 1164]
 Total_steps_till_now: 1164000  Success_rate: 0.00  Time: 22.28s  Add_rate: 0.180 
 Ave_reward: [-35.03 -44.12 -25.3  -51.36 -54.81 -45.7  -35.62 -38.28 -45.59 -50.1 ]
[Train_epoch 1165]
 Total_steps_till_now: 1165000  Success_rate: 0.00  Time: 22.31s  Add_rate: 0.180 
 Ave_reward: [-32.64 -52.57 -50.07 -43.76 -46.45 -36.35 -52.24 -47.19 -43.44 -50.44]
[Test_after_epoch 1165]
  Success_rate: 0.00  Time: 2.50s  Add_rate: 0.180 
 Ave_reward: [-52.51 -31.95 -49.07 -47.78 -39.5  -32.42 -46.29 -36.12 -33.31 -33.44]
[Train_epoch 1166]
 Total_steps_till_now: 1166000  Success_rate: 0.00  Time: 22.21s  Add_rate: 0.180 
 Ave_reward: [-43.55 -47.28 -51.34 -38.83 -42.58 -40.51 -55.56 -48.96 -47.32 -41.05]
[Train_epoch 1167]
 Total_steps_till_now: 1167000  Success_rate: 0.00  Time: 22.29s  Add_rate: 0.180 
 Ave_reward: [-36.31 -38.74 -64.02 -54.26 -36.69 -26.07 -72.86 -50.82 -39.25 -55.25]
[Train_epoch 1168]
 Total_steps_till_now: 1168000  Success_rate: 0.00  Time: 21.87s  Add_rate: 0.180 
 Ave_reward: [-41.   -49.43 -54.51 -62.97 -42.43 -34.56 -32.71 -42.51 -48.06 -41.77]
[Train_epoch 1169]
 Total_steps_till_now: 1169000  Success_rate: 0.00  Time: 22.14s  Add_rate: 0.180 
 Ave_reward: [-41.27 -34.23 -51.54 -43.14 -49.88 -49.59 -48.26 -38.37 -35.33 -53.15]
[Train_epoch 1170]
 Total_steps_till_now: 1170000  Success_rate: 0.00  Time: 21.75s  Add_rate: 0.180 
 Ave_reward: [-33.39 -49.04 -69.44 -37.67 -61.73 -25.84 -31.98 -43.08 -34.7  -62.49]
[Test_after_epoch 1170]
  Success_rate: 0.00  Time: 2.58s  Add_rate: 0.180 
 Ave_reward: [-39.15 -45.54 -45.04 -42.73 -53.34 -41.06 -57.3  -43.26 -41.78 -43.67]
Traceback (most recent call last):
  File "main.py", line 153, in <module>
    trainer.learn()
  File "/home/ubuntu/tf_on_schednet/agents/schednet/trainer.py", line 130, in learn
    action_n = self.get_action(obs_n, schedule_n, global_step)
  File "/home/ubuntu/tf_on_schednet/agents/schednet/trainer.py", line 275, in get_action
    predator_action = self._predator_agent.act(predator_obs, schedule_n)
  File "/home/ubuntu/tf_on_schednet/agents/schednet/agent.py", line 70, in act
    raise ValueError('action_prob contains NaN')
ValueError: action_prob contains NaN


#### debug_1
[Train_epoch 279]
 Total_steps_till_now: 11160000  Success_rate: 0.26  Time: 101.74s  Add_rate: 0.05 
 Ave_reward: [-12.73 -12.02 -11.8  -11.83 -10.86 -12.17 -12.43 -11.66 -12.63 -10.6 ]
[Train_epoch 280]
 Total_steps_till_now: 11200000  Success_rate: 0.26  Time: 101.18s  Add_rate: 0.05 
 Ave_reward: [-11.92 -11.04 -11.36 -11.79 -13.09 -13.24 -12.55 -12.21 -12.45 -11.99]
[Test_after_epoch 280]
  Success_rate: 0.26  Time: 114.73s  Add_rate: 0.05 
 Ave_reward: [-11.92 -11.64 -12.87 -11.45 -12.11 -11.06 -12.56 -11.05 -10.73 -10.42]
[Train_epoch 281]
 Total_steps_till_now: 11240000  Success_rate: 0.24  Time: 102.23s  Add_rate: 0.05 
 Ave_reward: [-13.64 -12.96 -11.88 -12.61 -11.97 -12.4  -12.62 -11.87 -12.46 -13.96]
[Train_epoch 282]
 Total_steps_till_now: 11280000  Success_rate: 0.24  Time: 102.15s  Add_rate: 0.05 
 Ave_reward: [-12.36 -11.7  -12.55 -12.41 -13.26 -13.07 -12.06 -11.82 -11.86 -11.53]
[Train_epoch 283]
 Total_steps_till_now: 11320000  Success_rate: 0.23  Time: 101.27s  Add_rate: 0.05 
 Ave_reward: [-13.04 -12.06 -12.09 -11.96 -11.57 -12.12 -11.73 -12.07 -12.18 -12.  ]
[Train_epoch 284]
 Total_steps_till_now: 11360000  Success_rate: 0.25  Time: 101.37s  Add_rate: 0.05 
 Ave_reward: [-13.2  -12.12 -11.51 -11.62 -13.4  -12.9  -12.66 -13.11 -12.72 -12.09]
[Train_epoch 285]
 Total_steps_till_now: 11400000  Success_rate: 0.24  Time: 102.16s  Add_rate: 0.05 
 Ave_reward: [-13.62 -11.93 -12.7  -12.09 -12.   -14.14 -13.21 -13.83 -12.61 -14.01]
[Train_epoch 286]
 Total_steps_till_now: 11440000  Success_rate: 0.27  Time: 102.71s  Add_rate: 0.05 
 Ave_reward: [-13.09 -11.68 -11.05 -12.64 -12.61 -13.   -11.56 -13.18 -11.07 -11.46]
[Train_epoch 287]
 Total_steps_till_now: 11480000  Success_rate: 0.23  Time: 102.72s  Add_rate: 0.05 
 Ave_reward: [-12.12 -13.07 -11.83 -11.89 -12.84 -11.94 -12.72 -13.47 -13.51 -12.86]
[Train_epoch 288]
 Total_steps_till_now: 11520000  Success_rate: 0.23  Time: 101.76s  Add_rate: 0.05 
 Ave_reward: [-12.96 -13.4  -12.86 -13.78 -12.92 -13.22 -12.18 -12.38 -13.45 -11.3 ]
[Train_epoch 289]
 Total_steps_till_now: 11560000  Success_rate: 0.24  Time: 101.92s  Add_rate: 0.05 
 Ave_reward: [-12.67 -13.17 -12.09 -13.09 -13.33 -12.09 -13.36 -13.78 -13.05 -11.83]
[Train_epoch 290]
 Total_steps_till_now: 11600000  Success_rate: 0.25  Time: 102.42s  Add_rate: 0.05 
 Ave_reward: [-11.96 -12.61 -12.75 -12.25 -12.33 -12.23 -12.16 -11.25 -11.46 -11.44]
[Test_after_epoch 290]
 Success_rate: 0.27  Time: 114.86s  Add_rate: 0.05 
 Ave_reward: [-11.39 -10.39 -12.46 -11.05 -12.02 -11.51 -11.62 -11.34 -11.06 -11.86]
[Train_epoch 291]
 Total_steps_till_now: 11640000  Success_rate: 0.26  Time: 101.55s  Add_rate: 0.05 
 Ave_reward: [-12.86 -12.56 -12.76 -12.15 -11.3  -11.66 -12.25 -11.55 -11.96 -12.74]




 [Train_epoch 383]
 Total_steps_till_now: 15320000  Success_rate: 0.15  Time: 97.81s  Add_rate: 0.06 
 Ave_reward: [-17.54 -15.63 -16.18 -15.71 -15.78 -13.44 -14.22 -16.   -16.09 -14.96]
[Train_epoch 384]
 Total_steps_till_now: 15360000  Success_rate: 0.09  Time: 102.16s  Add_rate: 0.07 
 Ave_reward: [-19.03 -19.88 -18.95 -19.97 -19.43 -19.26 -19.52 -20.57 -19.31 -19.95]
[Train_epoch 385]
 Total_steps_till_now: 15400000  Success_rate: 0.10  Time: 99.49s  Add_rate: 0.07 
 Ave_reward: [-20.29 -19.33 -19.59 -19.33 -19.12 -19.84 -20.26 -19.21 -19.97 -19.77]
[Train_epoch 386]
 Total_steps_till_now: 15440000  Success_rate: 0.10  Time: 98.87s  Add_rate: 0.07 
 Ave_reward: [-19.43 -19.7  -21.06 -20.38 -19.75 -18.63 -21.66 -19.3  -19.63 -19.55]
[Train_epoch 387]
 Total_steps_till_now: 15480000  Success_rate: 0.09  Time: 98.41s  Add_rate: 0.07 
 Ave_reward: [-19.65 -18.99 -18.97 -20.04 -19.02 -18.11 -19.55 -18.78 -18.4  -19.8 ]
[Train_epoch 388]
 Total_steps_till_now: 15520000  Success_rate: 0.11  Time: 98.18s  Add_rate: 0.07 
 Ave_reward: [-18.28 -17.78 -18.18 -17.3  -19.05 -17.42 -19.23 -19.23 -19.33 -17.48]
[Train_epoch 389]
 Total_steps_till_now: 15560000  Success_rate: 0.11  Time: 98.12s  Add_rate: 0.07 
 Ave_reward: [-19.4  -20.83 -18.5  -20.78 -20.58 -18.61 -20.62 -19.88 -20.3  -19.01]
[Train_epoch 390]
 Total_steps_till_now: 15600000  Success_rate: 0.10  Time: 99.06s  Add_rate: 0.07 
 Ave_reward: [-19.85 -18.31 -20.4  -19.61 -20.59 -19.57 -20.32 -21.17 -19.79 -20.37]
[Test_after_epoch 390]
  Success_rate: 0.11  Time: 92.42s  Add_rate: 0.07 
 Ave_reward: [-16.36 -17.09 -17.2  -18.2  -17.99 -17.08 -19.82 -18.95 -16.92 -17.84]
[Train_epoch 391]
 Total_steps_till_now: 15640000  Success_rate: 0.09  Time: 99.07s  Add_rate: 0.07 
 Ave_reward: [-20.01 -19.85 -19.48 -19.73 -20.07 -18.65 -20.06 -20.37 -20.23 -19.66]
[Train_epoch 392]
 Total_steps_till_now: 15680000  Success_rate: 0.10  Time: 99.81s  Add_rate: 0.07 
 Ave_reward: [-20.2  -19.66 -19.6  -18.92 -18.82 -18.82 -20.1  -20.91 -19.84 -19.84]
[Train_epoch 393]
 Total_steps_till_now: 15720000  Success_rate: 0.10  Time: 99.06s  Add_rate: 0.07 
 Ave_reward: [-18.99 -20.05 -19.42 -19.17 -19.66 -18.7  -17.77 -20.06 -19.97 -18.77]
[Train_epoch 394]
 Total_steps_till_now: 15760000  Success_rate: 0.11  Time: 100.09s  Add_rate: 0.07 
 Ave_reward: [-19.18 -19.8  -18.69 -18.17 -20.35 -17.99 -18.42 -20.6  -18.1  -17.36]
[Train_epoch 395]
 Total_steps_till_now: 15800000  Success_rate: 0.10  Time: 100.58s  Add_rate: 0.07 
 Ave_reward: [-20.42 -17.84 -17.77 -20.48 -17.9  -17.06 -19.85 -19.07 -17.21 -19.31]
[Train_epoch 396]
 Total_steps_till_now: 15840000  Success_rate: 0.10  Time: 99.87s  Add_rate: 0.07 
 Ave_reward: [-20.42 -18.67 -20.07 -20.29 -19.51 -18.54 -18.74 -20.05 -19.13 -17.61]




 [Test_after_epoch 440]
  Success_rate: 0.14  Time: 93.55s  Add_rate: 0.07 
 Ave_reward: [-16.06 -15.41 -17.25 -16.08 -16.61 -15.34 -17.15 -15.48 -14.7  -15.69]
[Train_epoch 441]
 Total_steps_till_now: 17640000  Success_rate: 0.13  Time: 101.19s  Add_rate: 0.07 
 Ave_reward: [-17.47 -18.12 -18.03 -18.43 -19.02 -17.26 -19.34 -17.93 -18.28 -17.31]
[Train_epoch 442]
 Total_steps_till_now: 17680000  Success_rate: 0.11  Time: 100.12s  Add_rate: 0.07 
 Ave_reward: [-18.51 -16.99 -16.56 -17.15 -17.12 -17.48 -18.54 -18.5  -17.42 -17.4 ]
[Train_epoch 443]
 Total_steps_till_now: 17720000  Success_rate: 0.12  Time: 100.76s  Add_rate: 0.07 
 Ave_reward: [-17.49 -17.15 -18.36 -16.9  -16.54 -16.96 -16.46 -16.74 -15.91 -16.32]
[Train_epoch 444]
 Total_steps_till_now: 17760000  Success_rate: 0.12  Time: 100.52s  Add_rate: 0.07 
 Ave_reward: [-16.96 -17.42 -18.06 -17.04 -18.54 -16.53 -17.12 -17.73 -18.44 -18.81]
[Train_epoch 445]
 Total_steps_till_now: 17800000  Success_rate: 0.12  Time: 101.55s  Add_rate: 0.07 
 Ave_reward: [-17.92 -15.93 -16.04 -17.21 -18.27 -16.17 -16.68 -18.14 -15.85 -16.19]
[Train_epoch 446]
 Total_steps_till_now: 17840000  Success_rate: 0.13  Time: 100.24s  Add_rate: 0.07 
 Ave_reward: [-16.94 -16.73 -16.72 -15.38 -18.78 -15.06 -15.33 -16.21 -16.69 -16.58]
[Train_epoch 447]
 Total_steps_till_now: 17880000  Success_rate: 0.13  Time: 101.40s  Add_rate: 0.07 
 Ave_reward: [-16.6  -16.91 -17.08 -16.16 -17.42 -14.88 -18.2  -16.65 -16.05 -16.92]
[Train_epoch 448]
 Total_steps_till_now: 17920000  Success_rate: 0.11  Time: 101.50s  Add_rate: 0.07 
 Ave_reward: [-17.26 -16.26 -17.66 -17.29 -17.1  -16.21 -16.55 -17.45 -17.86 -17.26]
[Train_epoch 449]
 Total_steps_till_now: 17960000  Success_rate: 0.13  Time: 102.09s  Add_rate: 0.07 
 Ave_reward: [-18.26 -17.2  -16.41 -18.58 -16.8  -15.95 -16.09 -16.27 -16.31 -17.6 ]
[Train_epoch 450]
 Total_steps_till_now: 18000000  Success_rate: 0.11  Time: 102.09s  Add_rate: 0.07 
 Ave_reward: [-16.88 -17.78 -16.56 -16.01 -17.49 -16.6  -17.96 -16.69 -17.16 -16.96]
[Test_after_epoch 450]
  Success_rate: 0.08  Time: 93.84s  Add_rate: 0.08 
 Ave_reward: [-20.17 -19.44 -19.41 -17.86 -19.23 -17.46 -19.51 -18.51 -18.33 -19.2 ]
[Train_epoch 451]
 Total_steps_till_now: 18040000  Success_rate: 0.07  Time: 101.18s  Add_rate: 0.08 
 Ave_reward: [-21.62 -20.92 -20.93 -21.8  -23.46 -19.99 -22.3  -22.28 -20.26 -21.24]
[Train_epoch 452]
 Total_steps_till_now: 18080000  Success_rate: 0.07  Time: 101.98s  Add_rate: 0.08 
 Ave_reward: [-22.47 -19.77 -22.41 -21.7  -20.99 -21.11 -21.82 -20.38 -20.86 -21.36]
[Train_epoch 453]
 Total_steps_till_now: 18120000  Success_rate: 0.08  Time: 100.10s  Add_rate: 0.08 
 Ave_reward: [-22.65 -20.28 -21.61 -19.84 -20.26 -18.46 -22.55 -20.44 -22.38 -20.37]





 [Train_epoch 760]
 Total_steps_till_now: 30400000  Success_rate: 0.02  Time: 108.16s  Add_rate: 0.12 
 Ave_reward: [-23.1  -23.06 -20.43 -23.99 -22.67 -23.09 -23.44 -23.31 -22.41 -21.89]
[Test_after_epoch 760]
  Success_rate: 0.02  Time: 95.18s  Add_rate: 0.12 
 Ave_reward: [-22.77 -21.87 -18.45 -21.71 -21.72 -20.73 -21.65 -21.57 -20.99 -21.83]
[Train_epoch 761]
 Total_steps_till_now: 30440000  Success_rate: 0.02  Time: 108.80s  Add_rate: 0.12 
 Ave_reward: [-24.28 -23.42 -21.56 -23.15 -23.7  -22.38 -21.98 -23.48 -23.78 -20.46]
[Train_epoch 762]
 Total_steps_till_now: 30480000  Success_rate: 0.02  Time: 110.65s  Add_rate: 0.12 
 Ave_reward: [-24.95 -22.16 -21.94 -22.64 -23.35 -23.8  -22.14 -24.53 -22.36 -22.9 ]
[Train_epoch 763]
 Total_steps_till_now: 30520000  Success_rate: 0.02  Time: 110.11s  Add_rate: 0.12 
 Ave_reward: [-24.57 -24.09 -21.9  -23.94 -21.74 -22.42 -22.25 -22.4  -22.69 -23.  ]
[Train_epoch 764]
 Total_steps_till_now: 30560000  Success_rate: 0.01  Time: 111.16s  Add_rate: 0.12 
 Ave_reward: [-25.25 -22.49 -20.84 -23.07 -22.26 -23.09 -22.61 -23.92 -23.03 -22.  ]
[Train_epoch 765]
 Total_steps_till_now: 30600000  Success_rate: 0.02  Time: 110.37s  Add_rate: 0.12 
 Ave_reward: [-24.59 -25.25 -21.03 -22.31 -23.87 -22.89 -22.89 -24.69 -22.46 -23.53]
[Train_epoch 766]
 Total_steps_till_now: 30640000  Success_rate: 0.03  Time: 111.55s  Add_rate: 0.12 
 Ave_reward: [-22.62 -23.08 -21.19 -22.9  -21.36 -21.05 -21.44 -22.1  -22.3  -22.62]
[Train_epoch 767]
 Total_steps_till_now: 30680000  Success_rate: 0.04  Time: 111.05s  Add_rate: 0.12 
 Ave_reward: [-23.16 -21.84 -18.39 -21.68 -21.39 -21.79 -20.18 -20.79 -21.87 -21.71]
[Train_epoch 768]
 Total_steps_till_now: 30720000  Success_rate: 0.03  Time: 110.30s  Add_rate: 0.12 
 Ave_reward: [-22.38 -21.9  -20.45 -21.24 -22.84 -21.46 -21.98 -21.73 -21.17 -20.77]
[Train_epoch 769]
 Total_steps_till_now: 30760000  Success_rate: 0.01  Time: 111.73s  Add_rate: 0.12 
 Ave_reward: [-23.95 -22.36 -21.17 -21.74 -23.24 -22.21 -21.36 -23.16 -21.81 -21.24]
[Train_epoch 770]
 Total_steps_till_now: 30800000  Success_rate: 0.02  Time: 113.15s  Add_rate: 0.12 
 Ave_reward: [-26.58 -24.99 -21.48 -22.88 -22.96 -23.72 -21.96 -24.47 -22.62 -22.1 ]
[Test_after_epoch 770]
  Success_rate: 0.03  Time: 96.53s  Add_rate: 0.12 
 Ave_reward: [-21.5  -20.61 -18.   -20.37 -19.99 -19.87 -21.41 -21.08 -19.37 -19.97]
[Train_epoch 771]
 Total_steps_till_now: 30840000  Success_rate: 0.02  Time: 113.78s  Add_rate: 0.12 
 Ave_reward: [-23.6  -24.5  -20.86 -21.18 -22.04 -20.72 -22.03 -22.93 -21.18 -21.86]
[Train_epoch 772]
 Total_steps_till_now: 30880000  Success_rate: 0.02  Time: 113.69s  Add_rate: 0.12 
 Ave_reward: [-24.47 -23.54 -19.92 -22.25 -23.69 -21.42 -22.73 -22.7  -20.87 -22.35]