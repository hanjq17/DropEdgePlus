import numpy as np
import scipy.sparse as sp



def uniform_float(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit)
    else:
        return random_state.uniform(lower, upper, number)


def uniform_int(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if not isinstance(lower, int):
        raise ValueError("lower must be of type 'int', got {0} instead"
                         .format(type(lower)))
    if not isinstance(upper, int):
        raise ValueError("upper must be of type 'int', got {0} instead"
                         .format(type(upper)))
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit).astype(np.int32)
    else:
        return random_state.randint(int(lower), int(upper), number)


def generate_random_parameter_settings(search_spaces_dict, num_experiments, seed):
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()

    settings = {}
    for param in search_spaces_dict:
        if search_spaces_dict[param]["format"] == "values":
            settings[param] = random_state.choice(search_spaces_dict[param]["values"], size=num_experiments)

        elif search_spaces_dict[param]["format"] == "range":
            if search_spaces_dict[param]["dtype"] == "int":
                gen_func = uniform_int
            else:
                gen_func = uniform_float
            settings[param] = gen_func(random_state,
                                       lower=search_spaces_dict[param]["min"],
                                       upper=search_spaces_dict[param]["max"],
                                       number=num_experiments,
                                       log_scale=search_spaces_dict[param]["log_scale"])

        else:
            raise ValueError(f"Unknown format {search_spaces_dict[param]['format']}.")

    settings = {key: settings[key].tolist() for key in settings}  # convert to python datatypes since MongoDB cannot
    # serialize numpy datatypes
    return settings

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))
