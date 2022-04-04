BASTION := bastion			# Name of the instance for SSH tunnel
TPU := atpu					# TPU name
TYPE := v2-8  				# TPU type (just v2-8 or v3-8)
TENSORFLOW_RUNTIME := 2.8.0
NETWORK := default
ZONE := us-central1-f   # Must be one of us-central1-[abc], europe-west4-a, or asia-east1-c
REGION := $(shell echo $(ZONE) | cut --fields=-2 --delimiter=-)
PROJECT := transformer-wisdom

# On google cloud there are TPUs, TPU VMs, and TPU execution groups. A TPU
# means that a physical TPU is allocated and we have access to it via a gRPC
# endpoint. A TPU VM means that additionally we get SSH access to the worker
# that is physically attached to the TPU. A TPU execution group means that we
# get a TPU plus a separate compute instance that is set up with network
# access to the TPU.
#
# Here we only really want a TPU, but a TPU never gets an external IP address,
# so in order to access it we would need to set up a separate bastion compute
# instance, which is an operational pain. Therefore we allocate a TPU VM just
# so that we get an external IP address.
#
# On the other hand raw TPUs seem to be more readily available than TPU VMs,
# based on my experience trying to create both.

create-tpu:
	gcloud compute tpus create $(TPU) \
		--zone $(ZONE) \
		--version $(TENSORFLOW_RUNTIME) \
		--network $(NETWORK) \
		--accelerator-type $(TYPE) \
		--project $(PROJECT)

delete-tpu:
	gcloud compute tpus delete $(TPU) \
		--zone $(ZONE) \
		--project $(PROJECT)

describe-tpu:
	gcloud compute tpus describe $(TPU) \
		--zone=$(ZONE) \
		--project $(PROJECT)

list-tpus:
	gcloud compute tpus list \
		--zone=$(ZONE) \
		--project $(PROJECT)

create-bastion:
	gcloud compute instances create $(BASTION) \
		--zone $(ZONE) \
		--machine-type e2-micro \
		--network $(NETWORK) \
		--project $(PROJECT)

delete-bastion:
	gcloud compute instances delete $(BASTION) \
		--project $(PROJECT)

# create a tunnel to the TPU worker grpc endpoint
tunnel:
	gcloud compute ssh $(BASTION) \
		--zone=$(ZONE) \
		-- \
		-N \
		-L 19870:$(shell \
			gcloud compute tpus describe $(TPU) \
				--zone $(ZONE) \
				--project $(PROJECT) \
				--format 'get(ipAddress,port)[separator=:]')

get-tpu-endpoint:
	gcloud compute tpus describe $(TPU) \
		--zone $(ZONE) \
		--project $(PROJECT) \
		--format 'get(ipAddress,port)[separator=:]'

#
# OLD: TPU VMs
#

TPU_VM := my-tpu-vm
TPU_RUNTIME := tpu-vm-tf-$(TENSORFLOW_RUNTIME)
SRC := mnist/train_with_keras_model_fit.py

create-tpu-vm:
	gcloud alpha compute tpus tpu-vm create $(TPU_VM) \
		--zone $(ZONE) \
		--version $(TPU_RUNTIME) \
		--network $(NETWORK) \
		--accelerator-type $(TYPE) \
		--project $(PROJECT)

ssh-tpu-vm:
	gcloud alpha compute tpus tpu-vm ssh $(TPU_VM) \
		--zone $(ZONE) \
		--project $(PROJECT)

scp-tpu-vm:
	gcloud alpha compute tpus tpu-vm scp \
		--zone $(ZONE) \
		--project $(PROJECT) \
		$(SRC) \
		my-tpu-vm:

run-on-tpu-vm:
	gcloud alpha compute tpus tpu-vm scp \
		--zone $(ZONE) \
		--project $(PROJECT) \
		$(SRC) \
		my-tpu-vm:
	gcloud alpha compute tpus tpu-vm ssh $(TPU_VM) \
		--zone $(ZONE) \
		--project $(PROJECT) \
		-- \
		python3 $(shell basename $(SRC)) --tpu local


list-tensorflow-versions:
	gcloud alpha compute tpus tpu-vm versions list \
		--zone=$(ZONE) \
		--project $(PROJECT)

#
# OLD: TPU execution groups
#

EG := my-execution-group

create-eg:
	gcloud compute tpus execution-groups create \
		--name $(EG) \
		--zone $(ZONE) \
		--tf-version $(TENSORFLOW_VERSION) \
		--accelerator-type $(TYPE) \
		--project $(PROJECT)

delete-eg:
	gcloud compute tpus execution-groups delete $(EG) \
		--zone $(ZONE) \
		--project $(PROJECT)

ssh:
	gcloud compute ssh $(EG) \
		--zone $(ZONE) \
		--project $(PROJECT)


# create firewall ruleset

# gcloud compute firewall-rules create $(RULESET_NAME) \
#    --network=NETWORK \
#    --allow=tcp:22

# use identity-aware proxy to tunnel to a raw TPU?
# https://cloud.google.com/iap/docs/tcp-forwarding-overview



#create an instance with docker pre-installed:
create-instance:
	gcloud compute instances create foo2 \
		--boot-disk-size 1TB \
		--image-family cos-stable \
		--image-project cos-cloud \
		--zone $(ZONE) \
		--project $(PROJECT)

# sudo systemctl stop docker.service docker.socket
# sudo mv /var/lib/docker /mnt/stateful_partition/   -- ignore the error
# sudo vi /etc/docker/daemon.json
#   add "data-root": "/mnt/stateful_partition/docker"
# sudo systemctl start docker.service docker.socket

# docker pull gcr.io/deeplearning-platform-release/tf-cpu.2-8
#  it's 5.56 GB

connect-to-tpu-from-docker:
	docker run \
		--rm \
		-it \
		-v $(pwd)/src:/src \
		-w /src \
		tensorflow/tensorflow:2.8.0 \
		python3 test-tpu.py --tpu grpc://10.81.197.122:8470

#another possible docker image:
#  us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8


# install and run simpleproxy on bastion
#   sudo apt update
#   sudo apt install -y simpleproxy
#   nohup simpleproxy -L 19870 -R 10.81.197.122:8470 > simpleproxy.log &

create-grpc-firewall-rule:
	gcloud compute firewall-rules create allow-19870 \
		--direction=INGRESS \
		--action=ALLOW \
		--rules=tcp:19870 \
		--source-ranges=0.0.0.0/0
