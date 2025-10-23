from .BaseOptimizer import BaseOptimizer
import pulp
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class TransferOptimizer(BaseOptimizer):
    def __init__(self, df, current_squad, kpi_col):
        super().__init__(df, kpi_col)
        self.current_squad = current_squad
        self.current_players = set(current_squad['player_name'].values)

        self.selling_prices = {}
        for idx, row in current_squad.iterrows():
            purchase_price = row.get('purchase_price', row['now_cost'])
            current_price = row['now_cost']
            if current_price > purchase_price:
                profit = current_price - purchase_price
                half_profit = int(profit * 10 / 2) / 10
                selling_price = purchase_price + half_profit
            else:
                selling_price = current_price
            self.selling_prices[row['player_name']] = selling_price

    def solve_transfers(self, 
                         budget=0,
                         free_transfers=1,
                         max_transfers=6,
                         players_to_remove=None,
                         players_to_keep=None,
                         points_penalty_per_transfer=4,
                         debug=True):

        players_to_remove = players_to_remove or []
        players_to_keep = players_to_keep or []

        players_to_remove = [p for p in players_to_remove if p in self.current_players]
        players_to_keep = [p for p in players_to_keep if p in self.current_players]

        if set(players_to_remove) & set(players_to_keep):
            print("Error: Players cannot be in both remove and keep lists")
            return None

        if debug:
            print(f"\n=== TRANSFER OPTIMIZATION ===")
            print(f"Current squad: {len(self.current_players)} players")
            print(f"Budget available: £{budget}M")
            print(f"Free transfers: {free_transfers}")
            print(f"Max transfers to consider: {max_transfers}")
            print(f"Points penalty per extra transfer: -{points_penalty_per_transfer}")
            if players_to_remove:
                print(f"Forced removals: {players_to_remove}")
            if players_to_keep:
                print(f"Protected players: {len(players_to_keep)}")

        prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

        x = pulp.LpVariable.dicts("in_squad", self.all_players, cat='Binary')
        transfer_out = pulp.LpVariable.dicts("transfer_out", self.all_players, cat='Binary')
        transfer_in = pulp.LpVariable.dicts("transfer_in", self.all_players, cat='Binary')
        s = pulp.LpVariable.dicts("start", self.all_players, cat='Binary')
        c = pulp.LpVariable.dicts("captain", self.all_players, cat='Binary')
        v = pulp.LpVariable.dicts("vice_captain", self.all_players, cat='Binary')
        formation_vars = pulp.LpVariable.dicts("formation", self.FORMATIONS.keys(), cat='Binary')

        num_transfers = pulp.LpVariable("num_transfers", lowBound=0, cat='Integer')
        extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat='Integer')

        points_from_team = self.create_objective(s, c)
        prob += points_from_team - (points_penalty_per_transfer * extra_transfers)

        for p in self.all_players:
            if p in self.current_players:
                prob += x[p] + transfer_out[p] == 1
                prob += transfer_in[p] == 0
            else:
                prob += x[p] == transfer_in[p]
                prob += transfer_out[p] == 0

        prob += num_transfers == pulp.lpSum([transfer_out[p] for p in self.all_players])
        prob += num_transfers <= max_transfers
        prob += extra_transfers >= num_transfers - free_transfers
        prob += extra_transfers >= 0
        prob += pulp.lpSum([transfer_out[p] for p in self.all_players]) == \
                pulp.lpSum([transfer_in[p] for p in self.all_players])

        for p in players_to_remove:
            prob += transfer_out[p] == 1
        for p in players_to_keep:
            prob += transfer_out[p] == 0

        transfer_cost = pulp.lpSum([
            self.player_info[p]['cost'] * transfer_in[p] for p in self.all_players
        ]) - pulp.lpSum([
            self.selling_prices.get(p, 0) * transfer_out[p] for p in self.all_players
        ])
        prob += transfer_cost <= budget

        self.add_squad_constraints(prob, x)
        self.add_starting_xi_constraints(prob, x, s, formation_vars)
        self.add_captain_constraints(prob, s, c, v)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"❌ Solver status: {pulp.LpStatus[prob.status]}")
            return None

        chosen_formation = [f for f in self.FORMATIONS.keys() if formation_vars[f].value() == 1][0]
        result = self.extract_solution(x, s, c, chosen_formation)
        result['transferred_out'] = [p for p in self.all_players if transfer_out[p].value() == 1]
        result['transferred_in'] = [p for p in self.all_players if transfer_in[p].value() == 1]
        result['num_transfers'] = int(num_transfers.value())
        result['free_transfers'] = free_transfers
        result['extra_transfers'] = int(extra_transfers.value()) if extra_transfers.value() else 0
        result['points_penalty'] = result['extra_transfers'] * points_penalty_per_transfer
        result['net_expected_points'] = result['expected_points'] - result['points_penalty']

        money_spent = sum(self.player_info[p]['cost'] for p in result['transferred_in'])
        money_gained = sum(self.selling_prices.get(p, 0) for p in result['transferred_out'])
        result['net_spent'] = money_spent - money_gained
        result['remaining_budget'] = budget - result['net_spent']
        result['squad_ranking'] = self.rank_squad(result['squad'])

        self.print_transfer_solution(result)
        return result

    def print_transfer_solution(self, result):
        print("\n=== TRANSFER PLAN ===")
        print(f"Formation chosen: {result['formation']}")
        print(f"Expected points: {result['net_expected_points']:.1f}")
        print(f"Free transfers: {result['free_transfers']}")
        print(f"Extra transfers: {result['extra_transfers']} (-{result['points_penalty']} pts penalty)")
        print(f"Net spent: £{result['net_spent']}M, Remaining budget: £{result['remaining_budget']}M")
        print("\nTransfers In:", result['transferred_in'])
        print("Transfers Out:", result['transferred_out'])
        print("\nSquad:")
        for p in result['squad']:
            info = self.player_info[p]
            print(f"  {p} ({info['position']}) - £{info['cost']}M - {info['points']:.1f} pts")
