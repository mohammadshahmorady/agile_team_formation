import pandas as pd

from team_formation import TeamFormation


test = TeamFormation('Android')
# candidates = pd.concat([test.beginner_candidates, test.intermediate_candidates, test.advanced_candidates])
# candidates_skill_areas = candidates.groupby('UserId')['SkillArea'].apply(list).reset_index()


# def _skill_area_documents(row):
#     if sa in row['SkillArea']:
#         return int(candidates[(candidates['UserId'] == row['UserId']) &
#                               (candidates['SkillArea'] == sa)]['AnswerCount'])
#     return 0


# for sa in test.skill_areas_information['SkillArea'].values:
#     candidates_skill_areas[sa] = candidates_skill_areas.apply(lambda row: 0 if sa not in row['SkillArea'] else int(
#         candidates[(candidates['UserId'] == row['UserId']) & (candidates['SkillArea'] == sa)]['AnswerCount']), axis=1)
#
#     print(candidates_skill_areas)
#     print(sa, ' done')
#
# candidates_skill_areas.drop(columns=['SkillArea'], inplace=True)
# candidates_skill_areas['AllDocuments'] = candidates_skill_areas.apply(
#     lambda row: candidates[candidates['UserId'] == row['UserId']]['AnswerCount'].sum(), axis=1)
# candidates_skill_areas.to_csv(f'{test.data_collection}\\candidates.csv', index=False)

test.candidates['MaxDocument'] = test.candidates[test.skill_areas_information['SkillArea']].max(axis=1)
test.candidates['SecondMax'] = test.candidates.apply(lambda row: row[
                test.skill_areas_information['SkillArea']].nlargest(2, keep='all').values[-1], axis=1)
test.candidates.to_csv(f'{test.data_collection}\\{test.data_collection}Candidates.csv', index=False)
