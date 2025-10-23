import streamlit as st
import pandas as pd
import numpy as np
from fpl_modelling.__main__ import run_pipeline_programmatically

st.set_page_config(page_title="FPL Wildcard Team Optimizer (Real Data)", layout="wide")
st.title("⚽ FPL Wildcard Team Optimizer")
st.markdown("This app predicts a squad for the next gameweek in FPL by using machine learning to predict points in the next gameweek and choosing a team that maximisea "
"this expected points while adhering to the rules of FPL team building")

# --- Embedded real `res` data (taken from the message you provided) ---
if st.button("Generate Team (Table View)"):
    res = run_pipeline_programmatically(pipeline_name="gameweek_prediction", params={"current_gameweek": 8, "model_num": 4})['optimal_team'].load()

    # Convert squad_ranking to DataFrame
    squad_ranking_df = pd.DataFrame(res['squad_ranking'])

    # Helper: build team DataFrame from res lists + ranking table (to get position)
    all_players = squad_ranking_df.set_index('player_name').to_dict(orient='index')

    rows = []
    for p in res['squad']:
        info = all_players.get(p, {})
        rows.append({
            'Player Name': p,
            'Player Position': info.get('position_name', 'Unknown'),
            'Team': info.get('team_id', ''),
            'Expected Points': info.get('points_per_game', np.nan),
            'Role': ('Captain' if p == res['captain'] else 'Vice-Captain' if p == res['vice_captain'] else ('Starter' if p in res['starters'] else 'Bench'))
        })

    team_df = pd.DataFrame(rows).sort_values(by='Expected Points', ascending=False)

    team_df.loc[team_df['Role'] == 'Captain', 'Expected Points'] *= 2


    # position colours for display
    position_colors = {
        'GK': '#FFD700',
        'Goalkeeper': '#FFD700',
        'DEF': '#00BFFF',
        'Defender': '#00BFFF',
        'MID': '#32CD32',
        'Midfielder': '#32CD32',
        'FWD': '#FF4500',
        'Forward': '#FF4500'
    }

    # --- Layout ---
    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader(f'🏆 Selected Squad: Expected Points = {team_df['Expected Points'].sum():.2f} ')

        def highlight_roles(row):
            if row.Role == 'Captain':
                return ['background-color: #FFF4CC']*len(row)
            elif row.Role == 'Vice-Captain':
                return ['background-color: #E8F4F9']*len(row)
            elif row.Role == 'Bench':
                return ['background-color: #D3D3D3']*len(row)
            else:
                return ['']*len(row)
            
        team_df['Player Name'] = team_df.apply(
            lambda x: f"🏆 {x['Player Name']}" if x['Role']=='Captain' else
                    f"🎖️ {x['Player Name']}" if x['Role']=='Vice-Captain' else x['Player Name'], axis=1
        )

        styled = team_df.style.apply(highlight_roles, axis=1)
        st.dataframe(styled, use_container_width=True)

        # st.markdown(f"**Formation:** {res['formation']} — **Expected points:** {team_df['Expected Points'].sum():.2f}")
        st.markdown(f"**Formation:** {res['formation']}")
        st.markdown('### Starters / Bench')
        st.write('**Starters:**', ', '.join(res['starters']))
        st.write('**Bench:**', ', '.join(res['bench']))
    with col2:
        st.markdown("\n")
        st.markdown("\n")
        

    # with col2:
    #     st.subheader('📊 Squad Ranking (points per game)')
    #     st.dataframe(squad_ranking_df.sort_values('points_per_game', ascending=False).reset_index(drop=True), use_container_width=True)

    #     st.markdown('### Visual: top players by PPG')
    #     chart_df = squad_ranking_df.sort_values('points_per_game', ascending=False).head(12)
    #     st.bar_chart(chart_df.set_index('player_name')['points_per_game'])

    # Additional: quick comparison summary (text)
    # st.markdown('---')
    # st.header('Model Performance Summary')
    # st.write(f"If you had used this model in previous gameweeks, the expected points for this wildcard would be **{res['expected_points']:.2f}**.")

    # Allow export of selected team to CSV
    @st.cache_data
    def make_export(df):
        return df.to_csv(index=False)

    csv = make_export(team_df)
    st.download_button('Download selected squad as CSV', data=csv, file_name='selected_squad.csv', mime='text/csv')

    st.markdown('---')

