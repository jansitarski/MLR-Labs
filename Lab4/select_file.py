import sys
import os

def select_file(extension, display=True, prompt="Wybierz plik: "):
    '''Allows to select file from a list of files of a given extension'''
    input_filenames = []
    for filename in os.listdir("."):
        if filename.endswith("."+extension):
            input_filenames.append(filename)
    input_filenames.sort()

    if display:
        print("Lista plików:")
        if len(input_filenames) > 8:
            # Display files in columns
            input_filenames_A = input_filenames[:len(input_filenames)//2]
            input_filenames_B = input_filenames[len(input_filenames)//2:]
            print()
            maxlen = len(max(input_filenames_A, key=len))
            nl = 1
            nr = len(input_filenames_A)+1
            for i,j in zip(input_filenames_A, input_filenames_B):
                print("%d. %s\t%d. %s" % (nl,i.ljust(maxlen, " "), nr, j))
                nl = nl+1
                nr = nr+1
            if len(input_filenames_B) > len(input_filenames_A):
                print("%d. %s" % (nr,input_filenames_B[-1:][0]))
        else:
            # One-column
            for i in range(len(input_filenames)):
                print ("{}. {}".format(i+1,input_filenames[i]))

        print("0. Koniec")

    # Collect the filename
    idx = -1
    while idx not in range(len(input_filenames)+1):
        try:
            idx = int(input(prompt))
        except ValueError:
            idx = 0
    if idx == 0:
        print("Koniec...")
        sys.exit()
    filename = input_filenames[idx-1]

    return(filename)
