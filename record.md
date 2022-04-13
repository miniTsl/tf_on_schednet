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