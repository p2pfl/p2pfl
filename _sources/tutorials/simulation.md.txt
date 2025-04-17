# 🕹️ Simulations

P2PFL leverages **[Ray](https://www.ray.io/)**, a powerful distributed computing framework, to enable efficient simulations of large-scale federated learning scenarios. This allows you to train and evaluate models across a cluster of machines or multiple processes on a single machine, significantly accelerating the process and overcoming the limitations of single-machine setups.

## 🌐 Ray Integration for Scalability

P2PFL seamlessly integrates with Ray to distribute the learning process. When Ray is installed, P2PFL automatically creates a pool of **actors**, which are independent Python processes that can be distributed across your cluster. Each actor hosts a `Learner` instance, allowing for parallel training and evaluation.

### 🧩 Actor Pool

The core of P2PFL's simulation capabilities is the `SuperActorPool`. This pool manages the lifecycle of `VirtualNodeLearner` actors. Each `VirtualNodeLearner` wraps a standard `Learner`, enabling it to be executed remotely by Ray. This means that each node in your federated learning simulation can be run as an independent actor, managed by the pool.

### 🚀 Benefits of Using Ray

*   **Scalability:** Distribute the learning process across multiple machines or processes, enabling larger-scale simulations.
*   **Efficiency:** Parallelize training and evaluation, significantly reducing overall experiment time.
*   **Fault Tolerance:** Ray's actor model provides fault tolerance. If an actor fails, Ray can automatically restart it.
*   **Resource Management:** Ray intelligently manages the allocation of resources (CPUs, GPUs) to actors.

### ⚙️ Setting Up a Ray Cluster

> To disable ray (even if installed), export an environment var `export DISABLE_RAY=1`.
To run P2PFL simulations with Ray, you need to set up a Ray cluster. This can be done on a single machine (for smaller simulations) or across multiple machines (for larger simulations).

#### Single Machine Setup

For simulations on a single machine, you don't need to explicitly start a Ray cluster. P2PFL will automatically initialize Ray in local mode when you start your experiment.

#### Multi-Machine Setup

For larger simulations, you'll need to set up a Ray cluster across multiple machines:

1. **Start the head node:** On the machine designated as the head node, run:

    ```bash
    ray start --head --port=6379
    ```

2. **Start worker nodes:** On each additional machine, run:

    ```bash
    ray start --address='<head_node_ip>:6379'
    ```

    Replace `<head_node_ip>` with the IP address of the head node.

3. **Verify the cluster:** Check the status of your Ray cluster using:

    ```bash
    ray status
    ```

Once the cluster is set up, you can run your P2PFL experiment as usual. P2PFL will automatically distribute the `VirtualNodeLearner` actors across the available nodes in the cluster.
