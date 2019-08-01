import matplotlib.pyplot as plt
import csv

x = []
y_mean = []
y_std = []

with open('csv/facenet.csv', 'r') as csvfile:
  rows = csv.reader(csvfile, delimiter=',')
  for row in rows:
    try:
      x.append(float(row[0]))
      y_mean.append(float(row[1]))
      y_std.append(float(row[2]))
    except:
      continue
plt.figure()
plt.errorbar(x, y_mean, yerr=y_std, fmt='--o')
plt.title('Pruning FaceNet')
plt.show()
