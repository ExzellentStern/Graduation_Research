import pandas as pd
import matplotlib.pyplot as plt

# CSV 読み込み
server = pd.read_csv("logs/server_metrics.csv")

# ===== Accuracy =====
plt.figure()
plt.plot(server["round"], server["accuracy"], marker="o")
plt.xlabel("Server Round")
plt.ylabel("Accuracy")
plt.title("Server-side Global Accuracy")
plt.grid()
plt.show()

# ===== Loss =====
plt.figure()
plt.plot(server["round"], server["loss_func"], marker="o")
plt.xlabel("Server Round")
plt.ylabel("Loss")
plt.title("Server-side Global Loss")
plt.grid()
plt.show()
