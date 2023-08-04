import pandas as pd
import numpy as np
from enum import Enum

pd.set_option('display.max_columns', 100)


class SetIntMethod(Enum):
    kernel_based = 1
    overlap = 2
    hybrid = 3


class TeamFormation:
    def __init__(self, dataset):
        self.data_collection = dataset
        self.advanced_candidates = pd.read_csv(f"{dataset}\\{dataset}AdvancedLevel.csv").rename(
            {'AdvancedSkillArea': 'SkillArea'}, axis=1)
        self.intermediate_candidates = pd.read_csv(f"{dataset}\\{dataset}IntermediateLevel.csv").rename(
            {'IntermediateSkillArea': 'SkillArea'}, axis=1)
        self.beginner_candidates = pd.read_csv(f"{dataset}\\{dataset}BeginnerLevel.csv").rename(
            {'BeginnerSkillArea': 'SkillArea'}, axis=1)
        self.shape_of_candidates = pd.read_csv(f"{dataset}\\{dataset}Shapes.csv")
        self.candidates = pd.read_csv(f"{self.data_collection}\\{self.data_collection}Candidates.csv")
        self.skill_areas_information = pd.read_csv(f"{dataset}\\{dataset}SkillArea.csv")
        self.lambda_d = 0.9
        self.alpha = 0.5
        self.all_documents = self.skill_areas_information['AnswerCount'].sum()

        self.skill_areas_information['Avg'] = self.skill_areas_information.apply(lambda row: row['AnswerCount'] / len(
            self.candidates[self.candidates[row['SkillArea']] != 0]), axis=1)
        # average is calculated only among those candidates who have at least one document in corresponding skill area
        # that is because, if average < 1 then knowledge value of all users would be negative in that skill area
        # considering equation 18

        self.skill_areas_information['AvgRDM'] = self.skill_areas_information.apply(
            lambda row: row['AnswerCount'] / len(self.candidates[self.candidates[row['SkillArea']] > 1]), axis=1)

    def _expertise_probability(self, candidate_documents, sa_adv):
        return self.lambda_d * (candidate_documents / sa_adv) + (1 - self.lambda_d) * (sa_adv / self.all_documents)
        # calculating expertise probability of a candidate in a specific skill area (Equation 5)

    def SEM(self, skill_areas):
        candidates = pd.concat([self.beginner_candidates, self.intermediate_candidates, self.advanced_candidates])
        candidates = candidates[candidates['SkillArea'].isin(skill_areas)]

        result = {}
        for sa in skill_areas:
            sa_adv = float(self.skill_areas_information[self.skill_areas_information['SkillArea'] == sa]['AnswerCount'])
            # number documents related to this skill area

            sa_candidates = candidates[candidates['SkillArea'] == sa].copy().reset_index()

            sa_candidates['ExpertiseProbability'] = self._expertise_probability(
                sa_candidates['AnswerCount'], sa_adv)

            result[sa] = sa_candidates.loc[sa_candidates['ExpertiseProbability'].idxmax()]['UserId']
            # Equation 3

            candidates = candidates[candidates.UserId != result[sa]]

        # print(result)
        return self.performance_measure(result, skill_areas)

    def MEM(self, skill_areas):
        self.candidates['Score'] = 1.0

        for sa in skill_areas:  # calculating joint expertise probability (Equation 6)
            sa_adv = float(self.skill_areas_information[self.skill_areas_information['SkillArea'] == sa]['AnswerCount'])
            self.candidates['Score'] *= self._expertise_probability(self.candidates[sa], sa_adv)

        candidates = self.candidates[['UserId', 'Score']].copy()
        result = {}
        for sa in skill_areas:
            result[sa] = candidates.loc[candidates['Score'].idxmax()]['UserId']
            # Equation 1

            candidates = candidates[candidates.UserId != result[sa]]

        # print(result)
        return self.performance_measure(result, skill_areas)

    def EBM(self, skill_areas, XEBM=False):
        self.candidates['Entropy'] = 0.0

        for sa in self.skill_areas_information['SkillArea']:
            p_sa_e = self.candidates[sa] / self.candidates['AllDocuments']  # Equation 13
            with np.errstate(divide='ignore'):
                log_p = np.where(p_sa_e != 0, np.log2(p_sa_e), 0)
            self.candidates['Entropy'] -= p_sa_e * log_p  # Equation 12

        self.candidates['T_shapedProbability'] = np.log2(self.candidates['AllDocuments']) / self.candidates['Entropy']
        # Equation 11

        candidates = self.candidates[['UserId', 'T_shapedProbability', 'MaxDocument'] + skill_areas].copy()
        candidates = candidates[~candidates.isin([np.nan, np.inf, -np.inf]).any(1)]
        # getting rid of I-shaped and H_shaped candidates

        min_t_shaped = candidates['T_shapedProbability'].min()
        max_t_shaped = candidates['T_shapedProbability'].max()
        candidates['T_shapedProbability'] = (candidates['T_shapedProbability'] - min_t_shaped) / (
                max_t_shaped - min_t_shaped)  # Min Max normalization to adjust probabilities between 0 and 1

        for sa in skill_areas:
            candidates[sa + '_score'] = candidates['T_shapedProbability'] * (
                    candidates[sa] / candidates['MaxDocument'])  # Equation 10 and 14

        if XEBM:
            return candidates

        result = {}
        for sa in skill_areas:
            result[sa] = candidates.loc[candidates[sa + '_score'].idxmax()]['UserId']
            candidates = candidates[candidates.UserId != result[sa]]

        # print(result)
        return self.performance_measure(result, skill_areas)

    def _P_set_int(self, skill_areas, candidates, set_int_method=SetIntMethod.hybrid, quadratic=True):
        if set_int_method == SetIntMethod.overlap:
            set_int_size = len(skill_areas) - 1
            set_e = candidates[skill_areas].astype(bool).sum(axis=1)
            for sa in skill_areas:
                candidates[sa + '_score'] *= (set_e - candidates[sa].astype(bool)) / set_int_size  # Equation 21

            return

        for sa in skill_areas:
            candidates[sa + '_f_KV'] = np.log2(candidates[sa] + 1) / float(2 * np.log2(
                self.skill_areas_information[self.skill_areas_information['SkillArea'] == sa]['Avg']))  # Equation 18

            if quadratic:
                candidates[sa + '_f_KV'] = np.where(candidates[sa + '_f_KV'] < 1,
                                                    -(2 * candidates[sa + '_f_KV'] - 1) ** 2 + 1, 0)  # Equation 20
            else:
                candidates[sa + '_f_KV'] = np.where(candidates[sa + '_f_KV'] < 1,
                                                    -abs(2 * candidates[sa + '_f_KV'] - 1) + 1, 0)  # Equation 19

        if set_int_method == SetIntMethod.kernel_based:
            for sa in skill_areas:
                for sa_int in skill_areas:
                    if sa_int == sa:
                        continue
                    candidates[sa_int + '_score'] *= candidates[sa + '_f_KV']  # Equation 15 and 17
        else:
            set_int = []
            for sa in skill_areas:
                set_int.append(sa + '_f_KV')
                candidates[sa + '_f_KV'] = candidates[sa + '_f_KV'] >= self.alpha
            set_int_size = len(skill_areas) - 1
            set_int_e = candidates[set_int].astype(bool).sum(axis=1)
            for sa in skill_areas:
                candidates[sa + '_score'] *= (set_int_e - candidates[sa + '_f_KV']) / set_int_size  # Equation 22

    def XEBM(self, skill_areas, set_int_method=SetIntMethod.hybrid, quadratic=True):
        candidates = self.EBM(skill_areas, XEBM=True)
        self._P_set_int(skill_areas, candidates, set_int_method, quadratic)
        result = {}
        for sa in skill_areas:
            result[sa] = candidates.loc[candidates[sa + '_score'].idxmax()]['UserId']
            candidates = candidates[candidates.UserId != result[sa]]

        # print(result)
        return self.performance_measure(result, skill_areas)

    def RDM(self, skill_areas, set_int_method=SetIntMethod.hybrid, quadratic=True, better_optimality=False):
        if better_optimality:
            candidates = self.candidates.copy()
        else:
            candidates = self.candidates[['UserId', 'MaxDocument', 'SecondMax'] + skill_areas].copy()
        for sa in skill_areas:
            candidates[sa + '_score'] = 1.0
        self._P_set_int(skill_areas, candidates, set_int_method, quadratic)

        candidates['SetInt'] = 0

        for sa in self.skill_areas_information['SkillArea']:
            candidates[sa + '_f_KV'] = np.log2(self.candidates[sa] + 1) / float(2 * np.log2(
                self.skill_areas_information[self.skill_areas_information['SkillArea'] == sa]['Avg']))

            if quadratic:
                candidates[sa + '_f_KV'] = np.where(candidates[sa + '_f_KV'] < 1,
                                                    -(2 * candidates[sa + '_f_KV'] - 1) ** 2 + 1, 0)
            else:
                candidates[sa + '_f_KV'] = np.where(candidates[sa + '_f_KV'] < 1,
                                                    -abs(2 * candidates[sa + '_f_KV'] - 1) + 1, 0)

            candidates['SetInt'] += candidates[sa + '_f_KV'] >= self.alpha  # Equation 23

        result = {}
        for sa in skill_areas:
            sa_candidates = candidates[(candidates[sa] == candidates['MaxDocument']) &
                                       (candidates[sa] > 1)].copy()  # Equation 26

            with np.errstate(divide='ignore'):
                log_second_max = np.where(sa_candidates['SecondMax'] != 0, np.log2(sa_candidates['SecondMax']), 0)

            sa_candidates[sa + '_score'] *= sa_candidates['SetInt'] * (
                    np.log2(sa_candidates[sa]) - log_second_max)  # Equation 27 and 28

            sa_candidates.sort_values(by=[sa + '_score'], ascending=False, inplace=True)
            sa_candidates = sa_candidates[:int(len(sa_candidates) * 0.5)]
            #  Keeping top 50% of candidates

            if better_optimality:
                top_skill_sum = sa_candidates['MaxDocument'].sum()
                average = top_skill_sum / len(sa_candidates)
                for _, sa_int in self.skill_areas_information.iterrows():
                    if sa_int['SkillArea'] == sa:
                        continue
                    sa_candidates.drop(sa_candidates[
                                    (sa_candidates[sa_int['SkillArea']] == sa_candidates['SecondMax']) &
                                    (sa_candidates[sa_int['SkillArea']] > average)].index, inplace=True)
                    # removing candidates that their number of documents in their second best skill area is greater -
                    # than the average of top skill documents of candidates

            result[sa] = sa_candidates.iloc[0]['UserId']
            candidates = candidates[candidates.UserId != result[sa]]

        # print(result)
        return self.performance_measure(result, skill_areas)

    def performance_measure(self, team, skill_areas):
        advanced = self.advanced_candidates[self.advanced_candidates['UserId'].isin(list(team.values()))].copy()
        intermediate = self.intermediate_candidates[self.intermediate_candidates['SkillArea'].isin(skill_areas)]
        intermediate = intermediate[intermediate['UserId'].isin(list(team.values()))].copy()
        coverage = 0.0
        communication = 0.0
        optimality = 0.0
        for sa in skill_areas:
            if not advanced[(advanced['SkillArea'] == sa) & (advanced['UserId'] == team[sa])].empty:
                coverage += 1

            for i in skill_areas:
                if i == sa:
                    continue
                if not intermediate[(intermediate['SkillArea'] == i) & (intermediate['UserId'] == team[sa])].empty \
                        or not advanced[(advanced['SkillArea'] == i) & (advanced['UserId'] == team[sa])].empty:
                    # Equation 31
                    communication += 1

            opt_e = len(advanced[advanced['UserId'] == team[sa]])
            if opt_e:
                optimality += 1.0 / opt_e ** 2  # Equation 33

        coverage /= len(skill_areas)  # Equation 29
        communication /= len(skill_areas) * (len(skill_areas) - 1) * 0.5  # Equation 30
        # Equation 30 in the article is wrong and might result in communication measures greater than 1
        # the right equation is communication/n(n-1) not communication/(n(n-1)/2)

        optimality /= len(skill_areas)  # Equation 32

        result: dict[str, float] = {'coverage': coverage, 'communication': communication, 'optimality': optimality}

        if coverage and communication and optimality:
            result['f_measure'] = 3.0 / (1.0/coverage + 1.0/communication + 1.0/optimality)  # Equation 34
        else:
            result['f_measure'] = 0

        return result
