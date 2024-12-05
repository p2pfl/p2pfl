# 🕹️ Simulations

> To disable ray, set `sdc akcakj sxaray=False` in the `p2pfl.config` file.

In the context of **p2pfl**, the **Simulation** section refers to the process of training and evaluating machine learning models in a federated learning environment where tasks such as training, evaluation, and aggregation are distributed across a cluster of computing nodes using **Ray**. This approach enhances the scalability and efficiency of federated learning by allowing tasks to be executed in parallel, improving throughput and reducing overall training time.

**Ray** is a powerful distributed computing framework that allows you to parallelize tasks.  With **Ray's actor pool**, individual tasks are broken down and assigned to different actors, which are isolated, long-running processes that handle specific tasks like training and evaluation.

Here's an explanation of the key elements in the **Simulation** process using Ray:

### 🌐 Ray Integration in your experiments

Once Ray dependencies are installed, it is used automatically without requiring any additional configuration from the user, allowing for efficient parallelism and task management across the network of nodes.

#### 🧩 Cluster Setup

To get started with Ray, the user can set up a cluster of nodes. This setup typically involves configuring the available machines or instances to communicate with each other. After this, Ray automatically takes care of distributing tasks among the nodes, freeing the user from the complexity of manual task management.

Here are the basic steps to set up the cluster:

1. **Start the Ray cluster**: On the head node (the central coordinator):
   ```bash
   ray start --head
   ```
   On worker nodes (additional machines):
   ```bash
   ray start --address='head_node_ip:6379'
   ```

2. **Verify the cluster**: Check the status of the cluster using the following command:
   ```bash
   ray status
   ```

3. **Run p2pfl with Ray**: Once the cluster is set up, Ray will automatically manage the distribution of tasks, and the federated learning process can begin by running the appropriate commands or starting the Node processes as described in the previous examples.


🌟 Ready? **You can view next**: > [Logger](docs-logger.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-logger.md">Logger</a> </div>