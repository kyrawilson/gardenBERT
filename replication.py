import pandas as pd
import plotly.express as px

qa = pd.read_csv("Garden path stimuli - 4.csv")

cols = [str(l) for l in list(range(19))]
keep = ["9"]+[str(n) for n in range(12,18)]
ds = pd.read_csv("results1-74.csv", on_bad_lines="skip", names=cols, keep_default_na=False, na_values=[""], usecols=keep).dropna()
answers = ds[ds['9'].isin(["Final"])]

answers = answers[~answers['13'].isin(["FS", "FRC"])]

#Make answer, context, and question lowercase
answers['14'] = answers['14'].str.lower()
answers['17'] = answers['17'].str.lower()
#print(answers)

#print(answers['12'].value_counts())
#print(answers['13'].value_counts())
#print(answers.groupby('16')['12'].value_counts())
#print(answers.groupby('16')['12'].count())

#Merge
answers = pd.merge(answers, qa, left_on=["14", "15"], right_on=["context", "question"])

#Score
def score(row):
    if row['17'].strip(". ") in row['answers.all.B'].split('\n'):
        return 1
    else:
        return 0

answers['Correct'] = answers.apply(score, axis=1)
answers.to_csv("results1-74_scored.csv", index=False)
#print(answers)

#Average of all question types
flat = answers.groupby('13')['Correct'].mean()
#print(answers.groupby('13')['Correct'].mean())

#Edit answers by hand....

scored = pd.read_csv("results1-74_scored_edit.csv")
groups = scored.groupby(['code', 'question', 'context'])['Correct_edit'].mean()
#print(groups.groupby('code').mean())
answers_df = pd.DataFrame(groups.groupby('code').mean()).reset_index()
print(answers_df)


names = {"AM": "Agent Matrix", "PM": "Patient Matrix", "PE": "Ambiguous Argument", "A1": "Matrix Action", "A2": "Embedded Action", "U": "Unambiguous", "A": "Ambiguous", "GP": "Garden Path", "LC": "Local Coherence"}
answers_df['Question Type'] = answers_df['code'].str.extract(r'-([A-Z0-9]*)$')
answers_df['Ambiguity'] = answers_df['code'].str.extract(r'-([A-Z0-9]*)-')
answers_df['Structure'] = answers_df['code'].str.extract(r'^([A-Z0-9]*)-')
answers_df['Question Type'] = answers_df['Question Type'].map(names)
answers_df['Ambiguity'] = answers_df['Ambiguity'].map(names)
answers_df['Structure'] = answers_df['Structure'].map(names)
answers_df['Question Type'] = pd.Categorical(answers_df['Question Type'], ["Agent Matrix", "Patient Matrix", "Ambiguous Argument", "Matrix Action", "Embedded Action"])

print(answers_df)
ambiguity_df = pd.DataFrame(answers_df.groupby(['Question Type', 'Ambiguity']).mean()).reset_index()
print(ambiguity_df)

fig = px.bar(ambiguity_df, x="Question Type", y="Correct_edit", color="Ambiguity", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1),
    font_family="Times New Roman",
    font_color ="black",
    font_size=36,
    #plot_bgcolor='rgba(0,0,0,0)'
)

fig.show()

structure_df = pd.DataFrame(answers_df.groupby(['Question Type', 'Structure']).mean()).reset_index()
print(structure_df)

fig = px.bar(structure_df, x="Question Type", y="Correct_edit", color="Structure", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1),
    font_family="Times New Roman",
    font_color ="black",
    font_size=36,
    #plot_bgcolor='rgba(0,0,0,0)'
)

fig.show()

print(answers_df)
print(ambiguity_df)
print(structure_df)
