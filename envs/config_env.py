#!/usr/bin/env python
# coding=utf8


def config_env(_flags):
    flags = _flags
    # IC3 initial env
    flags.DEFINE_integer('dim', 14, "Dimension of box (i.e length of road) ")
    flags.DEFINE_integer('vision', 0," Vision of car")
    flags.DEFINE_float('add_rate_min', 0.05, "rate at which to add car (till curr. start)")
    flags.DEFINE_float('add_rate_max',0.2, " max rate at which to add car")
    flags.DEFINE_float('curr_start', 250,"start making harder after this many epochs [0]")
    flags.DEFINE_float('curr_end', 1250, "when to make the game hardest [0]")
    flags.DEFINE_string('difficulty', 'medium',"Difficulty level, easy|medium|hard")
    flags.DEFINE_string('vocab_type', 'bool', "Type of location vector to use, bool|scalar")
    
    # Scenario
    flags.DEFINE_string("scenario", "predator_prey_obs", "Scenario")
    flags.DEFINE_integer("n_predator", 10, "Number of cars (predators)")
    flags.DEFINE_integer("n_prey", 0, "Number of preys = 0, i.e. all cars are the same")
    flags.DEFINE_boolean("obs_diagonal", True, "Whether the agent can see in diagonal directions")
    flags.DEFINE_boolean("moving_prey", True, "Whether the prey is moving")
    flags.DEFINE_integer("obs_range", 1, "Observation range")
    flags.DEFINE_integer("hetero", 1, "Heterogeneity of observation range")
    
    # Observation
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 7, "Size of the map")
    flags.DEFINE_float("render_every", 1000, "Render the nth episode")

    # GUI
    flags.DEFINE_boolean("gui", False, "Activate GUI")


def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "s-"+FLAGS.scenario+"-map-"+str(FLAGS.map_size)+"-or-"+str(FLAGS.obs_range)
