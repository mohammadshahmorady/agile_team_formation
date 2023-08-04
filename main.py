from team_formation import *

# data_collection = input("Which data collection do you choose(first letter should be uppercase)? ")
data_collection = 'Java'


def combination(number_of_choice_left, chosen_items, remaining_items, all_instances):
    if number_of_choice_left > len(remaining_items):
        return
    if number_of_choice_left == 0:
        all_instances.append(chosen_items)
        return
    for i in range(len(remaining_items)):
        new_chosen_items = chosen_items.copy()
        new_chosen_items.append(remaining_items[i])
        combination(number_of_choice_left - 1, new_chosen_items, remaining_items[i + 1:], all_instances)


team_formation = TeamFormation(data_collection)
# team_formation.MEM(['B', 'D', 'F', 'G'])

instances = []
for team_size in range(2, 3):
    combination(team_size, [], team_formation.skill_areas_information['SkillArea'].values.tolist(), instances)

coverage = communication = optimality = f_measure = 0
counter = 0
percentage = 0
for skill_areas in instances:
    counter += 1
    if counter > len(instances) * (percentage/100):
        print(percentage, ' % completed')
        percentage += 1
    result = team_formation.RDM(skill_areas, better_optimality=True)
    coverage += result['coverage']
    communication += result['communication']
    optimality += result['optimality']
    f_measure += result['f_measure']

coverage /= len(instances)
communication /= len(instances)
optimality /= len(instances)
print('100 % completed\n')
print('Coverage Average : ', coverage)
print('Communication Average : ', communication)
print('Optimality Average : ', optimality)
print('F_Measure : ', 3/(1/communication + 1/coverage + 1/optimality))
