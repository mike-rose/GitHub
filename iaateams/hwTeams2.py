# 11/13/16

with open('C:/Users/mail/GitHub/iaateams/data/teams_with_f3.csv', "rb") as f:
    stuDat = [tuple(x.strip(" \r\n").split(",")) for x in f.readlines()]


class student:
    def __init__(self, tup):
        self.id = int(tup[0])
        self.cohort = tup[1].lower()
        self.teams = [int(x) for x in [tup[2], tup[3], tup[4], tup[5], tup[6]]]
        self.modulesLed = [tup[8]]
        self.gender = int(tup[9])   # 0 => Female, 1=> Male
        self.name = tup[10]

    def canJoin(self, team):
        for stud in team.members:
            vec = [x is y for (x, y) in zip(mike.teams, stud.teams)]
            if sum(vec) > 0:
                return False


class team:
    def __init__(self, leader, cohort, module='s3'):
        self.members = [leader]
        self.module = module+cohort
        self.memIDs = [leader.id]

    def addStudent(self, stud):
        self.members.append(stud)
        self.memIDs.append(stud.id)

# create an instance of each student
msa17 = [student(x) for x in stuDat]

# identify the leads as students with no modulesLed
leads = [x for x in msa17 if x.modulesLed == ['']]

# identify the remaining students as having already lead previously
notLeads = [x for x in msa17 if x.modulesLed != ['']]

# create a set of each cohort label (just 'b' and 'o' in this case)
cohorts = set([x.cohort for x in msa17])

# start each time with a 'lead'
teams = {c: [team(x, c) for x in leads if x.cohort == c] for c in cohorts}

# >============================================>
# >============================================> TEST OBJECTS
# >============================================>


t3 = teams['o'][3]
info = (234, "O", 23, 2, 2, 3, 9, 0, "f2o", 1, "Rose",)
mike = student(info)

# test if mike can join t3
mike.canJoin(t3)

'''
check gender balance
compare stud with every existing member for past collaboration
'''

# compare mike with stud

mike.gender
mike.teams

# add mike to t3
t3.addStudent(mike)
t3.memIDs


#################################################################
#################################################################
#################################################################

#        maleCount = [1 for stud in t3.members if stud.gender == 1]
#        femaleCount = [-1 for stud in t3.members if stud.gender == 0]
#        if maleCount + self.gender > 3 or femaleCount + self.gender < -3:
#            return False
