# %%
import numpy as np
import random

path = 'C:/Users/mail/Documents/GitHub/iaateams/'
with open(path + 'data/HDUB.csv', "rb") as f:
    stuDat = [tuple(x.strip(" \r\n").split(",")) for x in f.readlines()]

class student:
    def __init__(self, tup):
        self.id = int(tup[0])
        self.cohort = tup[1].lower()
        self.teams = [int(x) for x in [tup[2], tup[3], tup[4], tup[5], tup[6]]]
        self.modulesLed = [tup[7]]
        self.gender = int(tup[8])   # 0 => Female, 1=> Male
        self.name = tup[9]
        self.pals = []

# %% create an instance of each student
msa17 = [student(x) for x in stuDat[1:]]
studs = [x for x in msa17 if x.cohort == 'o']

# students get a list of 'pals' that have already teamed up with
for stud in studs:
    for s2 in studs:
        if sum([len(set(pair)) for pair in zip(stud.teams, s2.teams)]) < 10:
            stud.pals.append(s2) 

# order students by how many pals they've had
palCounts = [(len(s.pals), s) for s in studs]
palCounts.sort(reverse=True)
studs = [p[1] for p in palCounts]

# identify the leads as students with no modulesLed
leads = [x for x in studs if x.modulesLed == ['']]

# identify the remaining students as having already lead previously
notLeads = [x for x in studs if x.modulesLed != ['']]

def canJoin(stu, team):
    matches = [x for x in stu.pals if x in team]
    if len(matches) > 0:
        return False
    else:
        return True

def byName(tls):
    ls = []
    for t in tls:
        ls.append([s.name for s in t])
    return ls
        
# %%
# list of teams(lists) with a leader on each team
tlist = [[lead] for lead in leads]

teamless = [x for x in notLeads]

for t in tlist:
    for stu in teamless:
        if len(t) < 2 and canJoin(stu, t) and stu.gender != t[0].gender:
            t.append(teamless.pop(teamless.index(stu)))
pairs = [x for x in tlist]

random.seed(33)
random.shuffle(teamless)       
for t in tlist:
    for stu in teamless:
        if len(t) < 3 and canJoin(stu, t) and stu.gender != t[0].gender:
            t.append(teamless.pop(teamless.index(stu)))        
sets = [x for x in tlist]

# %% 

for t in sets:
    for stu in teamless:
        if len(t) < 4 and canJoin(stu, t) and stu.gender == t[0].gender:
            t.append(teamless.pop(teamless.index(stu)))  
quads = [x for x in tlist]

teamOptions = []
for s in teamless:
    joinable = [i for i in range(12) if canJoin(s, quads[i])]
#        print(str(s.name)+': '+str(joinable))
    teamOptions.append(joinable)

studentOptions = []
for i in range(len(quads)):
    able = [x.name for x in teamless if canJoin(x, quads[i])]
#        print(str(i)+': '+str(able))
    studentOptions.append(able)

minTeamOptions = np.min([len(x) for x in teamOptions])
minStudentOptions = np.min([len(x) for x in studentOptions])
remaining = len(teamless)
#    if np.min([len(x) for x in teamOptions]+[len(x) for x in studentOptions]) > 0:
#    maybe.append([r, remaining, minTeamOptions, minStudentOptions])

# %% ===========================================================
teamOptions = []
for s in teamless:
    joinable = [i for i in range(12) if canJoin(s, quads[i])]
    print(str(s.name)+': '+str(joinable))
    teamOptions.append(joinable)

studentOptions = []
for i in range(len(quads)):
    able = [x.name for x in teamless if canJoin(x, quads[i])]
    print(str(i)+': '+str(able))
    studentOptions.append(able)
        
lastPeeps = []
for i in range(len(quads)):
    lastPeeps.append([i] + [x for x in teamless if canJoin(x, quads[i])])
    print([i] + [x.name for x in teamless if canJoin(x, quads[i])])

    
'''
0  Espinal
1  Danchis3  Nishadham
4  Stover
5  Patel
6  Nicholson
7  Valenta
8  Paton
9  Cross
10 Helmer
11 Bryja
'''
    
# view names and choose single instance of each student    
places = [5,1,1,1,1,2,2,1,
   
solution = []
for i in range(12):
    solution.append(lastPeeps[i][places[i]])
    
for i in range(12):
    print(canJoin(solution[i], quads[i]))

# ===========================================================
            
for t in tlist:
    for stu in solution:
        if len(t) < 5 and canJoin(stu, t):
            t.append(teamless.pop(teamless.index(stu)))              
len(teamless)
                

for st in teamless:
    print(str(st.name))
    print(str([x.name for x in st.pals]))
    print('__________')
    
for tm in tlist:
    print(len(tm))
    gender = [x.gender for x in tm]
    print(np.mean(gender))
    
#==============================================================================
# 
# x = [t for t in studs if t.teams[1]==12 ]
# 
# for p in x:
#     print(p.name)
# 
#==============================================================================

    




