import os

a = ['K','L','M','N','O','P','Q', 'R','S','T','U','V','W','X','Y','Z']

for file in os.listdir('F:\\chetan\\working_stuff\\Project\\Python\\ASL\\data\\asl_alphabet_train') :
    for j in range(101, 3000):
        if file.endswith(f"{j}.png"):
            print(file)
            os.remove(file)

