from kowalsky.opt import optimize
from kowalsky.opt import models

for model in models:
    if model[-1] == 'C': continue
    print()
    optimize(model,
             path='./feed_baseline.csv',
             y_label='count',
             direction='minimize',
             scorer='rmsle',
             trials=2)

