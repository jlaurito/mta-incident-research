''' Cleaning MTA My Alert data post-feature-generation for analysis
	
	This analysis looks at data from September 2014 to August 2017 for a 
	clean, recent set of subway alerts
'''

import pandas as pd

from datetime import datetime
from sklearn import tree


raw_data = pd.read_csv('/Users/josh.laurito/src/subwayservice/clean_data/data_for_analysis.csv')

START_DATE = '2014-09-01'
END_DATE = '2017-09-01' 


# Filter subway alerts & create features
recent_subway_alerts = raw_data[
    (raw_data['system'].isin(['NYC', 'NYCT'])) & 
    (raw_data['time'] > START_DATE) & 
    (raw_data['time'] < END_DATE) &
    # filter out buses
    (raw_data['title'].str.contains('M[0-9]|B[mM][0-9]|B[0-9]|B[xX][0-9]|Q[0-9]|X[0-9]|S[0-9]') == False) &
    (raw_data['msg'].str.contains('M[0-9]|B[mM][0-9]|B[0-9]|B[xX][0-9]|Q[0-9]|X[0-9]|S[0-9]') == False) & 
    (raw_data['title'].str.contains('((ario|xpres|ocal).*[Bb]us)') == False) &
    (raw_data['msg'].str.contains('((ario|xpres|ocal).*[Bb]us)') == False)
]


features = recent_subway_alerts.drop(
    ['Unnamed: 0', 'count', 'title', 'body', 'msg', 'system', 'time', 'hex_y'],
    axis=1
)


# create sample for manual classification where there are multiple potential classes
# excluding updates as noise. Add time to name to prevent over-writing work
manual_sample = recent_subway_alerts[
    (recent_subway_alerts['update'] == False)
].sample(1000)
file_name = '../data/mta_manually_classified_' + datetime.now().strftime('%H-%M-%S') + '.csv'
manual_sample.to_csv(file_name, index=False)

# load back manual sample once done to build classifier
classified = pd.read_csv('../data/mta_manually_classified_08-08-41.csv', encoding='latin1')


# train the decision tree off the classified data
y = classified['classified']
x = classified.drop(['classified','count', 'hex_x', 'title','body', 
                    'msg', 'system', 'time', 'hex_y'], axis=1)
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x, y)


# write results to predicted values
predicted_values = decision_tree.predict(features.drop('hex_x', axis=1))
recent_subway_alerts['estimated'] = predicted_values


# over-write values for 'Test' & 'Update'
recent_subway_alerts.loc[recent_subway_alerts['update'] == True, 'estimated'] = 'update'
recent_subway_alerts.loc[recent_subway_alerts['test'] == True, 'estimated'] = 'test'


# save classified data for analysis
recent_subway_alerts.to_csv( '../data/my_mta_data_for_analysis.csv',index=False)
