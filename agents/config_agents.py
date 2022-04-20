# coding=utf8
def config_agent(_flags):
    flags = _flags
    
    # IC3net中的参数
    flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs")
    flags.DEFINE_integer("max_steps", 40, "force to end the game after this many steps")
    flags.DEFINE_integer("epoch_size", 20000, "totla steps for each epoch")
    
    flags.DEFINE_integer("update_gap", 120, "steps between updates")
    
    # schednet中的参数
    flags.DEFINE_string("agent", "schednet", "Agent")
    flags.DEFINE_integer("training_step", 40000000, "Training time step")
    flags.DEFINE_integer("testing_step", 20000, "Testing time step")
    flags.DEFINE_integer("max_step", 40, "Maximum time step per episode")
    flags.DEFINE_boolean("eval_on_train", True, "Evaluation for every eval_step")
    flags.DEFINE_integer("eval_step", 200000, "Number of steps before evaluating = 10 epochs")

    # RL setting
    flags.DEFINE_float("df", 0.9, "Discount factor")
    flags.DEFINE_integer("b_size", 12000, "Size of the replay memory")
    flags.DEFINE_integer("m_size", 120, "Minibatch size, how much memory to replay for each update")
    flags.DEFINE_integer("pre_train_step", 10, "during [m_size * pre_train_step] take random action")

    # Network training setting
    flags.DEFINE_float("a_lr", 0.0001, "Learning rate")
    flags.DEFINE_float("w_lr", 0.0001, "Learning rate")
    flags.DEFINE_float("c_lr", 0.001, "Learning rate")
    flags.DEFINE_float("tau", 0.05, "Learning rate")
    flags.DEFINE_boolean("use_action_in_critic", False, "Use guided samples")
    flags.DEFINE_integer("h_critic", 128, "Width of hidden layer for critic")
    flags.DEFINE_boolean("trainable_encoder", True, "Make the encoder trainable")

    # Basic setting for simulation
    flags.DEFINE_boolean("load_nn", False, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "", "The name of file for loading")
    flags.DEFINE_boolean("train", True, "Training or testing")
    flags.DEFINE_integer("comm", 5, "Communication type")
    flags.DEFINE_boolean("e_share", False, "Share encoder")
    flags.DEFINE_boolean("s_share", False, "Share sender")
    flags.DEFINE_boolean("a_share", False, "Share aggregator")
    flags.DEFINE_string("sched", "schedule", "Scheduler type")
    flags.DEFINE_string("sch_type", "top", "Scheduling algorithm type (top, softmax)")
    
    flags.DEFINE_integer("capa", 2, "Capacity for comm")
    flags.DEFINE_integer("s_num", 1, "Number of agent for sheduling")

    
def get_filename():
    import config
    FLAGS = config.flags.FLAGS
    '''
    eg: -a-schednet-clr-0.0001-alr-1e-05-hc-64-cp-2-ss-False-es-False-as-False-sn-1-sched-schedule-s_type-top
    '''
    return "a-" + FLAGS.agent + "-clr-" + str(FLAGS.c_lr) + "-alr-" + str(FLAGS.a_lr) \
           + "-hc-" + str(FLAGS.h_critic) + "-cp-" + str(FLAGS.capa) + "-ss-" + str(FLAGS.s_share) \
           + "-es-" + str(FLAGS.e_share)  + "-as-" + str(FLAGS.a_share)  + "-sn-" + str(FLAGS.s_num)\
           + "-sched-" + str(FLAGS.sched) + "-s_type-" + str(FLAGS.sch_type) 
