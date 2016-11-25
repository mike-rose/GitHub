# 11/13/16
path = 'C:/Users/mail/Documents/GitHub/iaateams/'
with open(path + 'data/teams_with_f3.csv', "rb") as f:
    stuDat = [tuple(x.strip(" \r\n").split(",")) for x in f.readlines()]


class student:
    def __init__(self, tup):
        self.id = int(tup[0])
        self.cohort = tup[1].lower()
        self.teams = [int(x) for x in [tup[2], tup[3], tup[4], tup[5], tup[6]]]
        self.modulesLed = [tup[8]]
        self.gender = int(tup[9])   # 0 => Female, 1=> Male
        self.name = tup[10]
        self.s1=0

    def joinTeam(self, n):
#==============================================================================
        tm = teams[self.cohort][n]
        if len(tm.members) >= 5:
            return False
#==============================================================================
#==============================================================================
#         elif tm.genderMix.count(self.gender) >= 3:
#             return False
        elif [[t for t in self.teams if t in m.teams] for m in tm.members] !=0:
            return False
#==============================================================================
#==============================================================================
#         else:    
#==============================================================================
        self.s1=n
        tm.members.append(self)
        tm.memIDs.append(self.id)
        tm.genderMix.append(self.gender)
        print(str(self.id) + ' joined ' + str(n))
        return True


class team:
    def __init__(self, leader, cohort, module='s3'):
        self.members = [leader]
        self.module = module+cohort
        self.memIDs = [leader.id]
        self.genderMix = [leader.gender]

    def addStudent(self, stud):
        self.members.append(stud)
        self.memIDs.append(stud.id)
        self.genderMix(stud.gender)
        stud.teams.append

# create an instance of each student
msa17 = [student(x) for x in stuDat]

# identify the leads as students with no modulesLed
leads = [x for x in msa17 if x.modulesLed == ['']]

# identify the remaining students as having already lead previously
notLeads = [x for x in msa17 if x.modulesLed != ['']]

# create a set of each cohort label (just 'b' and 'o' in this case)
cohorts = set([x.cohort for x in msa17])

# start each team with a 'lead'
teams = {c: [team(x, c) for x in leads if x.cohort == c] for c in cohorts}

oco = [s for s in msa17 if s.cohort == 'o']

for i in range(len(oco)):
    print(oco[i].name)
    for j in range(len(teams['o'])):
        if len(teams['o'][j].members) < 2:
            k = oco[i].joinTeam(j)
        if k:
            print(str(i)+' '+str(j))



