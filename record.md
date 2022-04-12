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