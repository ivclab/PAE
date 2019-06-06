import matplotlib.pyplot as plt
import csv

x = []
y_mean = []
y_std = []
pruned_gender_csv = 'csv/experiment2/chalearn/gender/expand_final_05.csv'

with open(pruned_gender_csv, 'r') as csvfile:
  rows = csv.reader(csvfile, delimiter=',')
  for row in rows:
    try:
      x.append(float(row[0]))
      y_mean.append(float(row[1]))
      y_std.append(0.0)
    except:
      continue
plt.figure()
plt.errorbar(x, y_mean, yerr=y_std, fmt='--o')
plt.annotate('accuracy', xy=(x, y_mean))
plt.title('Pruning facenet (Gender dataset)')
plt.show()