import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    for value in range(65 , 91):
        X[chr(value)] = 0
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for row in f:
            for col in row:
                upperVal = col.upper()
                if ord(upperVal) >= 65 and ord(upperVal) < 91:
                    X[upperVal] += 1
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def main():
    print("Q1")
    shreder = shred(sys.argv[1])
    for letter, num in shreder.items():
        print(f"{letter} {num}")
    print("Q2")
    (E, S) = get_parameter_vectors()
    logE = round(shreder["A"] * math.log(E[0]),4)
    logS = round(shreder["A"] * math.log(S[0]),4)
    print("{:.4f}".format(logE))
    print("{:.4f}".format(logS))
    print("Q3")
    Eng = float(sys.argv[2])
    Esp = float(sys.argv[3])
    Eng_res = math.log(Eng)
    Esp_res = math.log(Esp)
    for i in range(26):
       Eng_res += shreder[chr(i + 65)] * math.log(E[i])
    Eng_res = round(Eng_res, 4)
    for j in range(26):
       Esp_res += shreder[chr(j + 65)] * math.log(S[j])
    Esp_res = round(Esp_res, 4)
    print(Eng_res)
    print(Esp_res)
    print("Q4")
    if (Esp_res - Eng_res >= 100):
        Prob = 0
    elif(Esp_res - Eng_res <= -100):
        Prob = 1
    else:
        Prob = round(1/(1 + math.exp(Esp_res - Eng_res)),4)
    print(Prob)
main()