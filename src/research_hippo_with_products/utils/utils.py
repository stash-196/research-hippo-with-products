import numpy as np

def compare_stat_signal(signal1, signal2, title1, title2):
    print(f"mean of {title1} signal: ", np.mean(signal1.real))
    print(f"mean of {title2} signal: ", np.mean(signal2.real))
    print("scale factor: ", np.mean(signal2.real) / np.mean(signal1.real))
    # std
    print(f"std of {title1} signal: ", np.std(signal1.real))
    print(f"std of {title2} signal: ", np.std(signal2.real))
    print("scale factor: ", np.std(signal2.real) / np.std(signal1.real))
    # max and min
    print(f"max and min of {title1} signal: ", np.max(signal1.real), np.min(signal1.real))
    print(f"max and min of {title2} signal: ", np.max(signal2.real), np.min(signal2.real))
    print("scale factor: ", np.max(signal2.real) / np.max(signal1.real), np.min(signal2.real) / np.min(signal1.real))
    