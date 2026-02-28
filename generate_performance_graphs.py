import matplotlib.pyplot as plt

# Data from your load tests
vus = [10, 20, 50, 100]
avg_latency = [62, 163, 307, 615]  # ms
throughput = [158, 122, 162, 161]  # rps

# ---- Latency Graph ----
plt.figure()
plt.plot(vus, avg_latency)
plt.xlabel("Virtual Users")
plt.ylabel("Average Latency (ms)")
plt.title("Latency vs Concurrency")
plt.savefig("latency_vs_vus.png")
plt.close()

# ---- Throughput Graph ----
plt.figure()
plt.plot(vus, throughput)
plt.xlabel("Virtual Users")
plt.ylabel("Requests per Second")
plt.title("Throughput vs Concurrency")
plt.savefig("throughput_vs_vus.png")
plt.close()

print("Graphs generated successfully.")