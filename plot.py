import matplotlib.pyplot as plt
import csv


def plot(loss, em, mode, img_name):
    plt.plot(
        loss,
        color="orange",
        label="loss",
    )
    plt.title("learning curve")
    plt.xlabel("iter")
    plt.ylabel(mode)
    plt.legend()
    plt.savefig(img_name)
    plt.clf()

loss = []
em = []
mode = ["loss", "em"]
with open('./plot.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i in rows:
      loss.append(i[0])
      em.append(i[1])

plot(loss[1:], em[1:], mode, "plot.png")