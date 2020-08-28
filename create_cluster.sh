# Create kubernetes cluster with 4 nodes for self-sup experiments
CLUSTER_NAME=self-sup
NUM_NODES=1

gcloud beta container --project "beyond-dl-1503610372419" \
clusters create "$CLUSTER_NAME" \
--zone "europe-west4-a" --no-enable-basic-auth \
--cluster-version "1.15.12-gke.2" \
--machine-type "n1-standard-2" \
--image-type "UBUNTU" \
--disk-type "pd-standard" \
--disk-size "20" \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/cloud-platform" --preemptible \
--num-nodes "$NUM_NODES" --enable-stackdriver-kubernetes --enable-ip-alias \
--network "projects/beyond-dl-1503610372419/global/networks/default" \
--subnetwork "projects/beyond-dl-1503610372419/regions/europe-west4/subnetworks/default" \
--default-max-pods-per-node "110" \
--no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing \
--enable-autoupgrade --enable-autorepair \
--max-surge-upgrade 1 --max-unavailable-upgrade 0 --enable-tpu

