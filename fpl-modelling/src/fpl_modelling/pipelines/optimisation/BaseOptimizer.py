class BaseOptimizer:
    """Base class with shared functionality for FPL optimizers."""

    FORMATIONS = {
        "3-4-3": {"Defender": 3, "Midfielder": 4, "Forward": 3},
        "3-5-2": {"Defender": 3, "Midfielder": 5, "Forward": 2},
        "4-4-2": {"Defender": 4, "Midfielder": 4, "Forward": 2},
        "4-3-3": {"Defender": 4, "Midfielder": 3, "Forward": 3},
        "5-3-2": {"Defender": 5, "Midfielder": 3, "Forward": 2}
    }

    POSITION_REQUIREMENTS = {
        'Goalkeeper': 2,
        'Defender': 5,
        'Midfielder': 5,
        'Forward': 3
    }

    def __init__(self, df, kpi_col):
        self.df = df
        self.kpi_col = kpi_col
        self.all_players = df['player_name'].values
        self.n_players = len(df)

        self.player_info = {}
        for idx, row in df.iterrows():
            self.player_info[row['player_name']] = {
                'cost': row['now_cost'],
                'points': row[kpi_col],
                'position': row['position_name'],
                'team': row['team_id']
            }

    def validate_formation(self, formation):
        if formation not in self.FORMATIONS:
            print(f"Error: Unknown formation '{formation}'")
            print(f"Available formations: {list(self.FORMATIONS.keys())}")
            return False
        return True

    def add_squad_constraints(self, prob, x, team_size=15):
        prob += pulp.lpSum([x[p] for p in self.all_players]) == team_size

        for pos, required in self.POSITION_REQUIREMENTS.items():
            prob += pulp.lpSum([
                x[p] for p in self.all_players
                if self.player_info[p]['position'] == pos
            ]) == required

        all_teams = set(self.player_info[p]['team'] for p in self.all_players)
        for team in all_teams:
            prob += pulp.lpSum([
                x[p] for p in self.all_players
                if self.player_info[p]['team'] == team
            ]) <= 3

    def add_starting_xi_constraints(self, prob, x, s, formation_vars):
        """Flexible formation constraint (automatically chooses best formation)."""
        prob += pulp.lpSum([s[p] for p in self.all_players]) == 11

        for p in self.all_players:
            prob += s[p] <= x[p]

        prob += pulp.lpSum([
            s[p] for p in self.all_players
            if self.player_info[p]['position'] == 'Goalkeeper'
        ]) == 1

        prob += pulp.lpSum([formation_vars[f] for f in self.FORMATIONS.keys()]) == 1

        for pos in ['Defender', 'Midfielder', 'Forward']:
            prob += pulp.lpSum([
                s[p] for p in self.all_players
                if self.player_info[p]['position'] == pos
            ]) == pulp.lpSum([
                self.FORMATIONS[f][pos] * formation_vars[f]
                for f in self.FORMATIONS.keys()
            ])

    def add_captain_constraints(self, prob, s, c, v):
        prob += pulp.lpSum([c[p] for p in self.all_players]) == 1
        prob += pulp.lpSum([v[p] for p in self.all_players]) == 1

        for p in self.all_players:
            prob += c[p] <= s[p]
            prob += v[p] <= s[p]
            prob += c[p] + v[p] <= 1

    def create_objective(self, s, c):
        return pulp.lpSum([
            self.player_info[p]['points'] * s[p] for p in self.all_players
        ]) + pulp.lpSum([
            self.player_info[p]['points'] * c[p] for p in self.all_players
        ])

    def extract_solution(self, x, s, c, v, formation):
        squad = [p for p in self.all_players if x[p].value() == 1]
        starters = [p for p in self.all_players if s[p].value() == 1]
        bench = [p for p in squad if p not in starters]
        captain = [p for p in self.all_players if c[p].value() == 1][0]
        vice = [p for p in self.all_players if v[p].value() == 1][0]

        total_cost = sum(self.player_info[p]['cost'] for p in squad)
        expected_points = sum(self.player_info[p]['points'] for p in starters)
        expected_points += self.player_info[captain]['points']

        return {
            "squad": squad,
            "starters": starters,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice,
            "formation": formation,
            "total_cost": total_cost,
            "expected_points": expected_points
        }

    def rank_squad(self, squad_players):
        squad_df = self.df[self.df['player_name'].isin(squad_players)].copy()
        squad_df = squad_df.sort_values(by=self.kpi_col, ascending=False)
        squad_df['rank'] = range(1, len(squad_df) + 1)
        return squad_df[['player_name', self.kpi_col, 'position_name', 'team_id', 'rank']]


