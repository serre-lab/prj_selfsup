

# What is kubernetes?
https://kubernetes.io/

# Using kubernetes with GCP
https://cloud.google.com/tpu/docs/tutorials/kubernetes-engine-resnet
https://cloud.google.com/kubernetes-engine

Minju:
1. Edit `create_cluster.sh` CLUSTER_NAME
2. Edit `kub_job.yaml` TPU type + experiment name
3. Create a new experiment (experiment dims are hardcoded in prepare_experiments.py for now -- feel free to make this dynamic).
4. Change out_dir in `jobs/pretrain_ilsvrc.sh`