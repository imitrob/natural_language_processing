import numpy as np
""" TODO! Append all actions and scene objects that are detected!
    This script should read the available action labels and object labels.
    Now the labels are predefined!
"""

A = ['move_up', 'release', 'stop', 'pick_up', 'push', 'unglue', 'pour', 'put', 'stack']
O = ['cup', 'bowl']

def attach_all_labels(output):
    output["action_probs"] = np.zeros(len(A))
    output[output["target_action"]] = 1.0
    output["action"] = A

    output["object_probs"] = np.zeros(len(O))
    output[output["target_object"]] = 1.0
    output["object"] = O

    return output


""" TODO! This script will be moved or reworked
"""
import yaml
import gesture_actions

def map_user_preferences(likelihood_vec_x, user: str):
    links_dict = yaml.safe_load(open(f"{gesture_actions.package_path}/links/{user}_links.yaml", mode='r'))
    A = links_dict['actions']
    A_action_words = links_dict['action_words']
    T = np.zeros((len(A), len(A_action_words)))
    
    for name,link in links_dict['links'].items():
        link['user'] # "melichar"
        link['action_template'] # "push"
        link['object_template'] # "cube_template"
        link['action_word'] # "push"
        # static_action_gesture, dynamic_action_gesture = link['action_gesture'] # [grab, swipe right]

        static_action_word_id = A_action_words.index(link['action_word'])
        action_template_id = A.index(link['action_template'])

        T[action_template_id, static_action_word_id] = 1
    
    # split static and dynamic gestures
    return A, np.max(T * likelihood_vec_x, axis=(1))