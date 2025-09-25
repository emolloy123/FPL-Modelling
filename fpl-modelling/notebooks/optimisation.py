import pulp
import numpy as np
import pandas as pd

class TeamOptimizer:
    def __init__(self, df, kpi_col):
        '''
        df: DataFrame with columns:
            - now_cost: player cost
            - position: 'Goalkeeper', 'Defender', 'Midfielder', 'Forward'
            - points metric (kpi_col)
            - player_name
        '''
        self.df = df
        self.costs = df['now_cost'].values
        self.points = df[kpi_col].values
        self.positions = df['position_name'].values
        self.players = df['player_name'].values
        self.n_players = len(df)
        
    def check_data_feasibility(self, budget=100):
        """Check if the data allows for a feasible solution"""
        print("=== DATA FEASIBILITY CHECK ===")
        
        # Check position counts
        pos_counts = {}
        for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            pos_counts[pos] = sum(1 for p in self.positions if p == pos)
            print(f"Available {pos}: {pos_counts[pos]}")
        
        # Check if we have enough players in each position
        required = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
        feasible = True
        for pos, req in required.items():
            if pos_counts[pos] < req:
                print(f"❌ INFEASIBLE: Need {req} {pos} players, only have {pos_counts[pos]}")
                feasible = False
            else:
                print(f"✅ {pos}: {pos_counts[pos]} >= {req} (OK)")
        
        # Check budget feasibility
        min_cost_team = 0
        for pos, req in required.items():
            pos_costs = [self.costs[i] for i in range(self.n_players) if self.positions[i] == pos]
            if len(pos_costs) >= req:
                pos_costs.sort()
                min_cost_team += sum(pos_costs[:req])
                print(f"Min cost for {req} {pos} players: {sum(pos_costs[:req])}")
            
        print(f"Minimum possible team cost: {min_cost_team}")
        print(f"Budget: {budget}")
        
        if min_cost_team > budget:
            print(f"❌ INFEASIBLE: Minimum team cost ({min_cost_team}) exceeds budget ({budget})")
            feasible = False
        else:
            print(f"✅ Budget: {min_cost_team} <= {budget} (OK)")
            
        return feasible

    def solve(self, budget=100, team_size=15, debug=True):
        if debug:
            if not self.check_data_feasibility(budget):
                print("Data is not feasible for the given constraints!")
                return None
        
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

        # Decision variables
        x = pulp.LpVariable.dicts("squad", self.players, cat='Binary')       # in squad
        s = pulp.LpVariable.dicts("start", self.players, cat='Binary')       # in starting 11
        c = pulp.LpVariable.dicts("captain", self.players, cat='Binary')     # captain
        v = pulp.LpVariable.dicts("vice_captain", self.players, cat='Binary')# vice-captain

        # Formation choices
        formations = {
            "3-4-3": {"Defender": 3, "Midfielder": 4, "Forward": 3},
            "3-5-2": {"Defender": 3, "Midfielder": 5, "Forward": 2},
            "4-4-2": {"Defender": 4, "Midfielder": 4, "Forward": 2},
            "4-3-3": {"Defender": 4, "Midfielder": 3, "Forward": 3},
            "5-3-2": {"Defender": 5, "Midfielder": 3, "Forward": 2}
        }
        formation_vars = pulp.LpVariable.dicts("formation", formations.keys(), cat='Binary')

        # Objective: maximize expected points + captain bonus
        prob += pulp.lpSum([self.points[i]*s[self.players[i]] for i in range(self.n_players)]) + \
                pulp.lpSum([self.points[i]*c[self.players[i]] for i in range(self.n_players)])

        # Budget constraint
        prob += pulp.lpSum([self.costs[i]*x[self.players[i]] for i in range(self.n_players)]) <= budget

        # Squad size constraint
        prob += pulp.lpSum([x[p] for p in self.players]) == team_size

        # Position constraints (full squad) 'Goalkeeper', 'Defender', 'Midfielder', 'Forward']
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 2
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == 5
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == 5
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == 3

        # Only one formation allowed
        prob += pulp.lpSum([formation_vars[f] for f in formations.keys()]) == 1

        # Starting lineup constraints
        prob += pulp.lpSum([s[p] for p in self.players]) == 11
        for p in self.players:
            prob += s[p] <= x[p]  # must be in squad to start

        # Formation position constraints - THIS IS A POTENTIAL ISSUE
        for pos in ['Defender', 'Midfielder', 'Forward']:
            prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == pos]) == \
                    pulp.lpSum([formations[f][pos] * formation_vars[f] for f in formations.keys()])

        # Starting goalkeeper constraint
        prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 1

        # Captain constraints
        prob += pulp.lpSum([c[p] for p in self.players]) == 1
        prob += pulp.lpSum([v[p] for p in self.players]) == 1
        for p in self.players:
            prob += c[p] <= s[p]
            prob += v[p] <= s[p]
            prob += c[p] + v[p] <= 1  # Can't be both captain and vice-captain

        # Solve
        if debug:
            print(f"\n=== SOLVING OPTIMIZATION ===")
            print(f"Players: {self.n_players}")
            print(f"Budget: {budget}")
            print(f"Team size: {team_size}")
        
        prob.solve(pulp.PULP_CBC_CMD(msg=debug))

        # Check if solver found an optimal solution
        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"❌ Solver status: {pulp.LpStatus[prob.status]}")
            
            if debug:
                print("\n=== DEBUGGING INFEASIBLE SOLUTION ===")
                # Try solving with relaxed constraints
                self.debug_constraints(budget, team_size)
            
            return None

        # Safely choose formation
        chosen_formations = [f for f in formations.keys() if formation_vars[f].value() == 1]
        if not chosen_formations:
            print("No formation chosen — solver failed to set formation variable.")
            return None
        chosen_formation = chosen_formations[0]

        # Output results
        selected = [p for p in self.players if x[p].value() == 1]
        starters = [p for p in self.players if s[p].value() == 1]
        captain = [p for p in self.players if c[p].value() == 1][0] if any(c[p].value() == 1 for p in self.players) else None
        vice_captain = [p for p in self.players if v[p].value() == 1][0] if any(v[p].value() == 1 for p in self.players) else None

        total_cost = sum(self.costs[i] for i in range(self.n_players) if x[self.players[i]].value() == 1)
        total_points = sum(self.points[i] for i in range(self.n_players) if s[self.players[i]].value() == 1)
        total_points += sum(self.points[i] for i in range(self.n_players) if c[self.players[i]].value() == 1)  # captain bonus

        print("\n=== SOLUTION ===")
        print("Selected Squad:", selected)
        print("Starters:", starters)
        print("Formation:", chosen_formation)
        print("Captain:", captain)
        print("Vice Captain:", vice_captain)
        print("Total Cost:", total_cost)
        print("Total Points:", total_points)

        return {
            "squad": selected,
            "starters": starters,
            "formation": chosen_formation,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_cost": total_cost,
            "total_points": total_points
        }
    
    def debug_constraints(self, budget, team_size):
        """Try to identify which constraint is causing infeasibility"""
        print("Testing individual constraint feasibility...")
        
        # Test 1: Can we select any valid squad (ignoring starting lineup)?
        prob1 = pulp.LpProblem("Test_Squad_Only", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("squad", self.players, cat='Binary')
        
        prob1 += pulp.lpSum([self.points[i]*x[self.players[i]] for i in range(self.n_players)])
        prob1 += pulp.lpSum([self.costs[i]*x[self.players[i]] for i in range(self.n_players)]) <= budget
        prob1 += pulp.lpSum([x[p] for p in self.players]) == team_size
        prob1 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 2
        prob1 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == 5
        prob1 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == 5
        prob1 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == 3
        
        prob1.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob1.status] == "Optimal":
            print("✅ Squad selection constraints are feasible")
        else:
            print("❌ Squad selection constraints are infeasible")
            return
        
        # Test 2: Try with a simpler formation constraint
        print("Testing with single formation (4-3-3)...")
        prob2 = pulp.LpProblem("Test_Single_Formation", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("squad", self.players, cat='Binary')
        s = pulp.LpVariable.dicts("start", self.players, cat='Binary')
        
        prob2 += pulp.lpSum([self.points[i]*s[self.players[i]] for i in range(self.n_players)])
        prob2 += pulp.lpSum([self.costs[i]*x[self.players[i]] for i in range(self.n_players)]) <= budget
        prob2 += pulp.lpSum([x[p] for p in self.players]) == team_size
        
        # Squad constraints
        prob2 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 2
        prob2 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == 5
        prob2 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == 5
        prob2 += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == 3
        
        # Starting lineup
        prob2 += pulp.lpSum([s[p] for p in self.players]) == 11
        for p in self.players:
            prob2 += s[p] <= x[p]
            
        # Fixed formation 4-3-3
        prob2 += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 1
        prob2 += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == 4
        prob2 += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == 3
        prob2 += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == 3
        
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob2.status] == "Optimal":
            print("✅ Single formation (4-3-3) is feasible")
            print("❌ Issue is likely with the formation variable constraints")
        else:
            print("❌ Even single formation is infeasible")

    def solve_simple(self, budget=100, formation="4-3-3"):
        """Simplified solver with fixed formation"""
        print(f"=== SOLVING WITH FIXED FORMATION: {formation} ===")
        
        formations = {
            "3-4-3": {"Defender": 3, "Midfielder": 4, "Forward": 3},
            "3-5-2": {"Defender": 3, "Midfielder": 5, "Forward": 2},
            "4-4-2": {"Defender": 4, "Midfielder": 4, "Forward": 2},
            "4-3-3": {"Defender": 4, "Midfielder": 3, "Forward": 3},
            "5-3-2": {"Defender": 5, "Midfielder": 3, "Forward": 2}
        }
        
        if formation not in formations:
            print(f"Unknown formation: {formation}")
            return None
            
        prob = pulp.LpProblem("FPL_Team_Selection_Simple", pulp.LpMaximize)

        # Decision variables
        x = pulp.LpVariable.dicts("squad", self.players, cat='Binary')
        s = pulp.LpVariable.dicts("start", self.players, cat='Binary')
        c = pulp.LpVariable.dicts("captain", self.players, cat='Binary')
        v = pulp.LpVariable.dicts("vice_captain", self.players, cat='Binary')

        # Objective
        prob += pulp.lpSum([self.points[i]*s[self.players[i]] for i in range(self.n_players)]) + \
                pulp.lpSum([self.points[i]*c[self.players[i]] for i in range(self.n_players)])

        # Budget constraint
        prob += pulp.lpSum([self.costs[i]*x[self.players[i]] for i in range(self.n_players)]) <= budget

        # Squad constraints
        prob += pulp.lpSum([x[p] for p in self.players]) == 15
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 2
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == 5
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == 5
        prob += pulp.lpSum([x[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == 3

        # Starting lineup
        prob += pulp.lpSum([s[p] for p in self.players]) == 11
        for p in self.players:
            prob += s[p] <= x[p]

        # Fixed formation
        prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Goalkeeper']) == 1
        prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Defender']) == formations[formation]['Defender']
        prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Midfielder']) == formations[formation]['Midfielder']
        prob += pulp.lpSum([s[self.players[i]] for i in range(self.n_players) if self.positions[i] == 'Forward']) == formations[formation]['Forward']

        # Captain constraints
        prob += pulp.lpSum([c[p] for p in self.players]) == 1
        prob += pulp.lpSum([v[p] for p in self.players]) == 1
        for p in self.players:
            prob += c[p] <= s[p]
            prob += v[p] <= s[p]
            prob += c[p] + v[p] <= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"❌ Solver status: {pulp.LpStatus[prob.status]}")
            return None

        # Extract results
        selected = [p for p in self.players if x[p].value() == 1]
        starters = [p for p in self.players if s[p].value() == 1]
        captain = [p for p in self.players if c[p].value() == 1][0]
        vice_captain = [p for p in self.players if v[p].value() == 1][0]

        total_cost = sum(self.costs[i] for i in range(self.n_players) if x[self.players[i]].value() == 1)
        total_points = sum(self.points[i] for i in range(self.n_players) if s[self.players[i]].value() == 1)
        total_points += sum(self.points[i] for i in range(self.n_players) if c[self.players[i]].value() == 1)

        print("\n=== SOLUTION ===")
        print("Selected Squad:", selected)
        print("Starters:", starters)
        print("Formation:", formation)
        print("Captain:", captain)
        print("Vice Captain:", vice_captain)
        print("Total Cost:", total_cost)
        print("Total Points:", total_points)

        return {
            "squad": selected,
            "starters": starters,
            "formation": formation,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_cost": total_cost,
            "total_points": total_points
        }