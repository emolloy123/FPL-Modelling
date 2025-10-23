import pulp
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from .BaseOptimizer import BaseOptimizer

class TeamOptimizer(BaseOptimizer):
    def solve(self, budget=100, debug=True):
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

        x = pulp.LpVariable.dicts("squad", self.all_players, cat='Binary')
        s = pulp.LpVariable.dicts("start", self.all_players, cat='Binary')
        c = pulp.LpVariable.dicts("captain", self.all_players, cat='Binary')
        v = pulp.LpVariable.dicts("vice_captain", self.all_players, cat='Binary')
        formation_vars = pulp.LpVariable.dicts("formation", self.FORMATIONS.keys(), cat='Binary')

        prob += self.create_objective(s, c)
        prob += pulp.lpSum([
            self.player_info[p]['cost'] * x[p] for p in self.all_players
        ]) <= budget

        self.add_squad_constraints(prob, x)
        self.add_starting_xi_constraints(prob, x, s, formation_vars)
        self.add_captain_constraints(prob, s, c, v)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"❌ Solver status: {pulp.LpStatus[prob.status]}")
            return None

        chosen_formation = [f for f in self.FORMATIONS.keys() if formation_vars[f].value() == 1][0]
        result = self.extract_solution(x, s, c,  chosen_formation)
        result['squad_ranking'] = self.rank_squad(result['squad'])
        self.print_team_solution(result)
        return result

    def print_team_solution(self, result):
        print("\n=== OPTIMAL TEAM ===")
        print(f"Total Cost: £{result['total_cost']}M")
        print(f"Expected Points: {result['expected_points']:.1f}")
        print(f"Formation: {result['formation']}")
        print(f"Captain: {result['captain']}")
        print(f"Vice Captain: {result['vice_captain']}")
        print("\nStarters:")
        for p in result['starters']:
            info = self.player_info[p]
            cap_marker = " (C)" if p == result['captain'] else " (VC)" if p == result['vice_captain'] else ""
            print(f"  {p} ({info['position']}){cap_marker} - £{info['cost']}M - {info['points']:.1f} pts")
        print("\nBench:")
        for p in result['bench']:
            info = self.player_info[p]
            print(f"  {p} ({info['position']}) - £{info['cost']}M - {info['points']:.1f} pts")
