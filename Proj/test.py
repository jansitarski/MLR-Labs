import os
import re
import numpy as np

statistics = {
    "adam": [],
    "rmsprop": []
}

count = 0

for method_name in statistics:
    for problem_size in range(10, 200):
            cmndName = "python wine_train.py -o " + method_name + " -e " + str(problem_size)
            print(cmndName)
            result = os.popen(cmndName)
            output = result.read()
            print(output)
            calcTime = re.findall("dt.*", output)
            if (len(calcTime) > 0):
                calcTime = re.findall("[0-9.]+", calcTime[0])
                #print(float(calcTime[0]))
                result_val = max(re.findall("[0-9.]+", re.findall("result.*", output)[0]))
                print(result_val)
            statistics[method_name].append([problem_size, float(result_val), float(calcTime[0])])

with open("result.plt", "w") as gnuplotfile:
    gnuplotfile.write("set term png\n")
    gnuplotfile.write("set output \"result.png\"\n")
    gnuplotfile.write("plot ")
    for method_name in statistics:
        print(method_name)
        print('{:16s}{:18s}{:s}'.format("epochs", "val_accuracy", "time"))
        summary = statistics[method_name]
        #print(summary)
        per_size = {}
        for values in summary:
            if (per_size.get(values[0]) is None):
                per_size[values[0]] = [[values[1], values[2]]]
            else:
                per_size[values[0]].append([values[1], values[2]])
        #print(per_size)
        for s in per_size:
            combined = np.mean(per_size[s], axis=0)
            print('{:s}{:19.2f}{:18.2f}'.format(str(s), combined[0], combined[1]))
            with open("result_" + method_name + ".txt", "a") as myfile:
                myfile.write(str(s) + " " + str(combined[0]) + " " + str(combined[1]) + "\n")
        gnuplotfile.write("'result_" + method_name + ".txt' u 1:2 every 4 w lines, ")
    #gnuplotfile.write("\n")

result = os.popen("gnuplot result.plt")
output = result.read()
