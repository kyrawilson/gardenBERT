from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def stacked_plot(answers_df):
    colors = {"Garden Path": "#DDAA33", "Local Coherence": "#BB5566"}
    patterns = {"Ambiguous": "", "Unambiguous": "."}

    #print(answers_df)
    answers_df['Question Type'] = pd.Categorical(answers_df['Question Type'], ["Agent Matrix", "Patient Matrix", "Ambiguous Argument", "Matrix Action", "Embedded Action"])
    answers_df = answers_df.sort_values(['Question Type', 'Correct'],ascending=[True, True])
    print(answers_df)

    temp = answers_df.iloc[::4][['Correct', 'Question Type', 'Ambiguity', 'Structure']]
    # fig = go.Figure(go.Bar(x=temp['Question Type'], y = temp['Correct'], color=temp['Structure'], color_discrete_sequence=temp['Structure'].map(colors), pattern_shape=temp['Ambiguity'], pattern_shape_sequence=temp['Ambiguity'].map(patterns)))

    fig = go.Figure(go.Bar(x=temp['Question Type'], y = temp['Correct'], marker_color=temp['Structure'].map(colors), marker_pattern={"shape": temp['Ambiguity'].map(patterns)}))

    for i in range(1,4):
        temp_prev = answers_df.iloc[i-1::4][['Correct', 'Question Type', 'Ambiguity', 'Structure']].reset_index()
        temp = answers_df.iloc[i::4][['Correct', 'Question Type', 'Ambiguity', 'Structure']].reset_index()
        temp['Correct_edit'] = temp['Correct']-temp_prev['Correct']
        #fig.add_trace(go.Bar(x=temp['Question Type'], y = temp['Correct_edit'],
        #marker_color=temp['Structure'].map(colors), marker_pattern={"shape": temp['Ambiguity'].map(patterns)}))
        fig.add_trace(go.Bar(x=temp['Question Type'], y = temp['Correct'],
        marker_color=temp['Structure'].map(colors), marker_pattern={"shape": temp['Ambiguity'].map(patterns)}))

    fig.update_layout(
        #barmode='stack',
        font_family="Times New Roman",
        font_color ="black",
        font_size=24,
        #plot_bgcolor='rgba(0,0,0,0)',
        #gridcolor='gray'
    )

    fig.update_xaxes(title_text='Question Type',
        title_font_family="Times New Roman",
        title_font_color ="black",
        title_font_size=24,
    )

    fig.update_yaxes(title_text='Score',
        title_font_family="Times New Roman",
        title_font_color ="black",
        title_font_size=24,
    )


    fig.show()


#model_name = "deepset/roberta-base-squad2"
model_name = "deepset/bert-base-cased-squad2"
models = ["ahotrod/electra_large_discriminator_squad2_512", "deepset/roberta-base-squad2", "deepset/bert-base-cased-squad2", "distilbert-base-cased-distilled-squad"]

#What do we want to do?
#load in Q/A DataFrame
#Loop through questions/answers
#save results
#average score for each kind of question
#should also look at whether the model was actually correct or not

df = pd.read_csv("Garden path stimuli - 4.csv")
scores = {v: [] for v in df.code.unique()}
answers = {v: [] for v in df.code.unique()}
out = []

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

for index, row in df.iterrows():
    QA_input = {
        'question': row['question'],
        'context': row['context']
    }
    res = nlp(QA_input)

    scores[row['code']].append(res['score'])

    if res['answer'].strip(".") in row['answers.all.B'].split('\n'):
        correct = 1
    else:
        correct = 0
    answers[row['code']].append(correct)
    out.append([row['code'], row['context'], row['question'], row['answers.all.B'], res['answer'], res['score'], correct])

final = {v: statistics.mean(scores[v]) for v in scores.keys()}
final_ans = {v: statistics.mean(answers[v]) for v in answers.keys()}
answers_df = pd.DataFrame(final_ans.items(), columns=['Type', 'Correct']).sort_values(by=['Correct'])
scores_df = pd.DataFrame(final.items(), columns=['Type', 'Score']).sort_values(by=['Score'])

names = {"AM": "Agent Matrix", "PM": "Patient Matrix", "PE": "Ambiguous Argument", "A1": "Matrix Action", "A2": "Embedded Action", "U": "Unambiguous", "A": "Ambiguous", "GP": "Garden Path", "LC": "Local Coherence"}
answers_df['Question Type'] = answers_df['Type'].str.extract(r'-([A-Z0-9]*)$')
answers_df['Ambiguity'] = answers_df['Type'].str.extract(r'-([A-Z0-9]*)-')
answers_df['Structure'] = answers_df['Type'].str.extract(r'^([A-Z0-9]*)-')
answers_df['Question Type'] = answers_df['Question Type'].map(names)
answers_df['Ambiguity'] = answers_df['Ambiguity'].map(names)
answers_df['Structure'] = answers_df['Structure'].map(names)
answers_df['Question Type'] = pd.Categorical(answers_df['Question Type'], ["Agent Matrix", "Patient Matrix", "Ambiguous Argument", "Matrix Action", "Embedded Action"])

stacked_plot(answers_df)

#print(answers_df)
amb_avg = answers_df.groupby(['Question Type', 'Ambiguity'])['Correct'].mean().reset_index()
print(amb_avg)
fig = px.bar(amb_avg, x="Question Type", y="Correct", color="Ambiguity", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])

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

str_avg = answers_df.groupby(['Question Type', 'Structure'])['Correct'].mean().reset_index()
print(str_avg)
fig1 = px.bar(str_avg, x="Question Type", y="Correct", color="Structure", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])

fig1.update_layout(legend=dict(
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

fig1.show()

scores_df['Question Type'] = scores_df['Type'].str.extract(r'-([A-Z0-9]*)$')
scores_df['Ambiguity'] = scores_df['Type'].str.extract(r'-([A-Z0-9]*)-')
scores_df['Structure'] = scores_df['Type'].str.extract(r'^([A-Z0-9]*)-')
scores_df['Question Type'] = scores_df['Question Type'].map(names)
scores_df['Ambiguity'] = scores_df['Ambiguity'].map(names)
scores_df['Structure'] = scores_df['Structure'].map(names)
scores_df['Correct'] = scores_df['Score']
scores_df['Question Type'] = pd.Categorical(scores_df['Question Type'], ["Agent Matrix", "Patient Matrix", "Ambiguous Argument", "Matrix Action", "Embedded Action"])

stacked_plot(scores_df)

amb_avg = scores_df.groupby(['Question Type', 'Ambiguity'])['Score'].mean().reset_index()
fig = px.bar(amb_avg, x="Question Type", y="Score", color="Ambiguity", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])
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

str_avg = scores_df.groupby(['Question Type', 'Structure'])['Score'].mean().reset_index()
fig1 = px.bar(str_avg, x="Question Type", y="Score", color="Structure", barmode='group', color_discrete_sequence=["#DDAA33", "#BB5566"])
fig1.update_layout(legend=dict(
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

fig1.show()


my_df = pd.DataFrame(out)
my_df.to_csv('GP_BERT.csv', index=False, header=['Code', 'Context', 'Question', 'Answer', 'Model_Ans', 'Score', 'Correct'])



#Matplotlib plotting
# ax = pd.DataFrame(final.items(), columns=['Type', 'Score']).plot(x='Type', y='Score') #xticks=pd.DataFrame(final.items(), columns=['Type', 'Score'])['Type'])
# ax.set_xticks(pd.DataFrame(final.items(), columns=['Type', 'Score']).index)
#
# ax.set_xticklabels(pd.DataFrame(final.items(), columns=['Type', 'Score']).Type, rotation=0)
#
# pd.DataFrame(final_ans.items(), columns=['Type', 'Correct']).plot(ax=ax, x='Type', y='Correct')
# plt.show(block=True)

# b) Load model & tokenizer
#model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
