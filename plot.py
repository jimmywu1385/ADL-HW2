import matplotlib.pyplot as plt
import csv


def plot(loss, step, img_name):
    plt.plot(
        step,
        loss,
        color="orange",
        label=img_name[:-4],
    )
    plt.title("learning curve")
    plt.xlabel("iter")
    plt.legend()
    plt.savefig(img_name)
    plt.clf()

loss = []
em = []
steps = []
step = 0
mode = ["loss", "em"]
with open('./plot.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i in rows:
    if step != 0:
        loss.append(float(i[0]))
        em.append(float(i[1]))
    steps.append(step)
    step += 100


plot(loss, steps[1:], "loss.png")
plot(em, steps[1:], "em.png")